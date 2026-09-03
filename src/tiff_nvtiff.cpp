// nvTIFF backend: decodes TIFF strips/tiles directly into device memory.
// Compiled only with SIRIUS_ENABLE_NVTIFF.

#include "tiff_internal.hpp"
#include "cuda_check.hpp"

#include <nvtiff.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::detail {

    namespace {

        const char* statusName(nvtiffStatus_t s) {
            switch (s) {
                case NVTIFF_STATUS_SUCCESS:                 return "NVTIFF_STATUS_SUCCESS";
                case NVTIFF_STATUS_NOT_INITIALIZED:         return "NVTIFF_STATUS_NOT_INITIALIZED";
                case NVTIFF_STATUS_INVALID_PARAMETER:       return "NVTIFF_STATUS_INVALID_PARAMETER";
                case NVTIFF_STATUS_BAD_TIFF:                return "NVTIFF_STATUS_BAD_TIFF";
                case NVTIFF_STATUS_TIFF_NOT_SUPPORTED:      return "NVTIFF_STATUS_TIFF_NOT_SUPPORTED";
                case NVTIFF_STATUS_ALLOCATOR_FAILURE:       return "NVTIFF_STATUS_ALLOCATOR_FAILURE";
                case NVTIFF_STATUS_EXECUTION_FAILED:        return "NVTIFF_STATUS_EXECUTION_FAILED";
                case NVTIFF_STATUS_ARCH_MISMATCH:           return "NVTIFF_STATUS_ARCH_MISMATCH";
                case NVTIFF_STATUS_INTERNAL_ERROR:          return "NVTIFF_STATUS_INTERNAL_ERROR";
                case NVTIFF_STATUS_NVCOMP_NOT_FOUND:        return "NVTIFF_STATUS_NVCOMP_NOT_FOUND";
                case NVTIFF_STATUS_NVJPEG_NOT_FOUND:        return "NVTIFF_STATUS_NVJPEG_NOT_FOUND";
                case NVTIFF_STATUS_TAG_NOT_FOUND:           return "NVTIFF_STATUS_TAG_NOT_FOUND";
                case NVTIFF_STATUS_PARAMETER_OUT_OF_BOUNDS: return "NVTIFF_STATUS_PARAMETER_OUT_OF_BOUNDS";
                case NVTIFF_STATUS_NVJPEG2K_NOT_FOUND:      return "NVTIFF_STATUS_NVJPEG2K_NOT_FOUND";
                case NVTIFF_STATUS_BATCH_INCOMPATIBLE:      return "NVTIFF_STATUS_BATCH_INCOMPATIBLE";
                default:                                    return "NVTIFF_STATUS_UNKNOWN";
            }
        }

        std::string describe(nvtiffStatus_t s) {
            std::string msg = statusName(s);
            switch (s) {
                case NVTIFF_STATUS_NVCOMP_NOT_FOUND:
                    msg += " (Deflate/ZIP decoding needs libnvcomp.so.5 on the loader path; "
                           "build with SIRIUS_ENABLE_NVCOMP or add nvCOMP to LD_LIBRARY_PATH)";
                    break;
                case NVTIFF_STATUS_TIFF_NOT_SUPPORTED:
                    msg += " (codec, predictor or sample layout not supported by nvTIFF; "
                           "e.g. the floating-point predictor)";
                    break;
                case NVTIFF_STATUS_BATCH_INCOMPATIBLE:
                    msg += " (pages use different codecs or strip/tile layouts)";
                    break;
                default:
                    break;
            }
            return msg;
        }

        void check(nvtiffStatus_t s, const char* what) {
            if (s != NVTIFF_STATUS_SUCCESS)
                throw std::runtime_error(std::string("nvTIFF ") + what + " failed: " + describe(s));
        }

        struct ParamsGuard {
            nvtiffDecodeParams_t p = nullptr;
            ParamsGuard() { check(nvtiffDecodeParamsCreate(&p), "nvtiffDecodeParamsCreate"); }
            ~ParamsGuard() { if (p) (void)nvtiffDecodeParamsDestroy(p); }
            ParamsGuard(const ParamsGuard&) = delete;
            ParamsGuard& operator=(const ParamsGuard&) = delete;
        };

        // Output descriptors for one nvtiffDecode call: page i of the batch is
        // written densely (pitch = region width * bytes per pixel) at
        // base + i * pageBytes, so a whole stack decodes into one contiguous
        // {pages, height, width} block without any post-copy.
        struct Batch {
            std::vector<nvtiffDecodeRegion_t> regions;
            std::vector<void*> planes;
            std::vector<std::size_t> pitches;
            std::vector<nvtiffImage_t> images;

            void build(const std::uint64_t* ifds, std::size_t n, const Region& r,
                       std::uint8_t* base, std::size_t pageBytes, std::size_t pitch) {
                regions.resize(n);
                planes.resize(n);
                pitches.assign(n, pitch);
                images.resize(n);
                for (std::size_t i = 0; i < n; ++i) {
                    regions[i] = nvtiffDecodeRegion_t{static_cast<std::size_t>(ifds[i]),
                                                      static_cast<std::int32_t>(r.x), static_cast<std::int32_t>(r.y),
                                                      r.width, r.height};
                    planes[i] = base + i * pageBytes;
                    images[i].plane_data = &planes[i];
                    images[i].plane_pitch_bytes = &pitches[i];
                    images[i].num_planes = 1;
                }
            }
        };

        // Pages per nvtiffDecode call. Large batches amortize per-call overhead
        // and keep the GPU busy; the cap bounds nvTIFF's internal scratch and
        // our conversion staging buffer for multi-gigabyte stacks.
        std::size_t pagesPerBatch(std::size_t pageBytes, std::size_t total) {
            constexpr std::size_t kTargetBytes = std::size_t{512} << 20;
            constexpr std::size_t kMaxPages = 1024;
            const std::size_t byBytes = std::max<std::size_t>(1, kTargetBytes / std::max<std::size_t>(pageBytes, 1));
            return std::min({byBytes, total, kMaxPages});
        }

        nvtiffDecoder_t decoderFor(NvTiffSession& s, int deviceIndex);

    } // namespace

    struct NvTiffSession {
        nvtiffStream_t stream = nullptr;
        std::map<int, nvtiffDecoder_t> decoders;   // keyed by CUDA device ordinal

        ~NvTiffSession() {
            int previous = -1;
            (void)cudaGetDevice(&previous);
            for (auto& [index, decoder] : decoders) {
                (void)cudaSetDevice(index);
                (void)nvtiffDecoderDestroy(decoder, nullptr);
                (void)cudaStreamSynchronize(nullptr);
            }
            if (previous >= 0) (void)cudaSetDevice(previous);
            if (stream) (void)nvtiffStreamClose(stream);
            (void)cudaGetLastError();
        }
    };

    namespace {

        // Caller holds impl.nvMutex.
        NvTiffSession& session(TiffFile::Impl& impl) {
            if (!impl.nv) {
                auto s = std::make_shared<NvTiffSession>();
                const nvtiffStatus_t st = nvtiffStreamOpenFromFile(impl.path.c_str(), &s->stream);
                if (st != NVTIFF_STATUS_SUCCESS)
                    throw std::runtime_error("nvTIFF failed to open " + impl.path + ": " + describe(st));
                impl.nv = std::move(s);
            }
            return *impl.nv;
        }

        // Current device must already be `deviceIndex`.
        nvtiffDecoder_t decoderFor(NvTiffSession& s, int deviceIndex) {
            auto it = s.decoders.find(deviceIndex);
            if (it != s.decoders.end()) return it->second;
            nvtiffDecoder_t d = nullptr;
            check(nvtiffDecoderCreateSimple(&d, nullptr), "nvtiffDecoderCreateSimple");
            s.decoders.emplace(deviceIndex, d);
            return d;
        }

        // Dry run over the whole request so an unsupported page anywhere in
        // the file is found before any output is written.
        bool supported(NvTiffSession& s, nvtiffDecoder_t dec, const DecodeJob& job, std::string& reason) {
            ParamsGuard params;
            Batch probe;
            probe.build(job.ifds->data(), job.ifds->size(), job.region, nullptr, 0, 0);
            check(nvtiffDecodeParamsSetRegions(params.p, probe.regions.data(),
                                               static_cast<std::uint32_t>(probe.regions.size())),
                  "nvtiffDecodeParamsSetRegions");
            const nvtiffStatus_t st = nvtiffDecodeCheckSupported(s.stream, dec, params.p, nullptr);
            if (st == NVTIFF_STATUS_SUCCESS) return true;
            reason = describe(st);
            return false;
        }

    } // namespace

    bool nvTiffSupports(TiffFile::Impl& impl, const DecodeJob& job, Device device, std::string& reason) {
        std::lock_guard<std::mutex> lock(impl.nvMutex);
        cuda::DeviceGuard guard(device.index);
        NvTiffSession& s = session(impl);
        return supported(s, decoderFor(s, device.index), job, reason);
    }

    bool decodeWithNvTiff(TiffFile::Impl& impl, const DecodeJob& job, void* dstDevice, Device device,
                          const Stream& stream, std::string& reason) {
        std::lock_guard<std::mutex> lock(impl.nvMutex);
        cuda::DeviceGuard guard(device.index);
        NvTiffSession& s = session(impl);
        nvtiffDecoder_t dec = decoderFor(s, device.index);
        if (!supported(s, dec, job, reason)) return false;

        const Region r = job.region;
        const PixelType native = job.geometry->pixelType;
        const std::size_t n = job.ifds->size();
        const std::size_t pixels = static_cast<std::size_t>(r.width) * r.height;
        const std::size_t nativePitch = static_cast<std::size_t>(r.width) * bytesPerPixel(native);
        const std::size_t nativePageBytes = pixels * bytesPerPixel(native);
        const std::size_t dstPageBytes = pixels * bytesPerPixel(job.dstType);
        const bool needConvert = native != job.dstType;
        const std::size_t perBatch = pagesPerBatch(nativePageBytes, n);
        const cudaStream_t cs = cuda::handle(stream);

        // When the caller wants another pixel type, nvTIFF writes the native
        // type into a device staging block and a conversion kernel finishes
        // the job on the GPU -- no host round trip.
        Buffer<std::uint8_t> staging;
        if (needConvert)
            staging = Buffer<std::uint8_t>(Shape{static_cast<Index>(perBatch * nativePageBytes)}, device,
                                           HostMemory::Pageable, stream);

        ParamsGuard params;
        Batch batch;
        auto* dst = static_cast<std::uint8_t*>(dstDevice);
        for (std::size_t first = 0; first < n; first += perBatch) {
            const std::size_t count = std::min(perBatch, n - first);
            std::uint8_t* out = needConvert ? staging.data() : dst + first * dstPageBytes;
            batch.build(job.ifds->data() + first, count, r, out, nativePageBytes, nativePitch);
            check(nvtiffDecodeParamsSetRegions(params.p, batch.regions.data(), static_cast<std::uint32_t>(count)),
                  "nvtiffDecodeParamsSetRegions");
            const nvtiffStatus_t st = nvtiffDecode(s.stream, dec, params.p, batch.images.data(), cs);
            // nvTIFF requires stream completion after every nvtiffDecode before
            // the decoder is reused, whatever the status.
            cuda::check(cudaStreamSynchronize(cs), "cudaStreamSynchronize after nvtiffDecode");
            if (st != NVTIFF_STATUS_SUCCESS)
                throw std::runtime_error("nvTIFF decode of " + impl.path + " failed: " + describe(st));
            if (needConvert)
                convertPixels(staging.data(), native, dst + first * dstPageBytes, job.dstType,
                              static_cast<Index>(count * pixels), device, stream);
        }
        if (needConvert) stream.synchronize();   // staging must outlive the conversion kernels
        return true;
    }

} // namespace sirius::detail
