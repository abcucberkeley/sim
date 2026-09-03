#include "sirius/fft.hpp"
#include "fft_backend.hpp"
#include "fftw_internal.hpp"

#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <fftw3.h>

// TODO: detect/handle int overflow and use
//       fftw_plan_guru64_dft instead of fftw_plan_many_dft

namespace sirius {
    namespace {
        // RAII for fft plan
        // Note: fftw_plan_s is a struct and fftw_plan is a pointer to that struct
        struct FFTWPlanDeleter {
            void operator()(fftw_plan plan) const {fftw_destroy_plan(plan);}
        };
        using PlanPtr = std::unique_ptr<fftw_plan_s, FFTWPlanDeleter>;

        // safe fft buffers
        struct FftwFree { void operator()(void* p) const { fftw_free(p); } };
        using FftwBuf = std::unique_ptr<fftw_complex[], FftwFree>;

        // FFTW's planner modifies global state — must be serialized across all instances
        std::mutex s_planner_mutex;
        int s_fftw_thread_count = 1;
        bool s_fftw_threads_initialized = false;

        // safe execution in case unaligned buffers with offset are passed
        void execute_safe(fftw_plan plan, int plan_alignment, int full_size,
            const std::complex<double>* in, std::complex<double>* out)
        {
            // case to fftw_complex*
            auto* in_ptr  = reinterpret_cast<fftw_complex*>(const_cast<std::complex<double>*>(in));
            auto* out_ptr = reinterpret_cast<fftw_complex*>(out);

            // check if aligned
            bool aligned = fftw_alignment_of(reinterpret_cast<double*>(in_ptr))  == plan_alignment &&
                           fftw_alignment_of(reinterpret_cast<double*>(out_ptr)) == plan_alignment;

            // if aligned, simply execute, otherwise need to copy
            if (aligned) {
                fftw_execute_dft(plan, in_ptr, out_ptr);
            } else {
                FftwBuf tmp_in (static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * full_size)));
                FftwBuf tmp_out(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * full_size)));
                std::memcpy(tmp_in.get(),  in_ptr,  sizeof(fftw_complex) * full_size);
                fftw_execute_dft(plan, tmp_in.get(), tmp_out.get());
                std::memcpy(out_ptr, tmp_out.get(), sizeof(fftw_complex) * full_size);
            }
        }

        // FFTW wisdom is global state no needs shared mutex
        void loadWisdomImpl(const std::string& path) {
            std::lock_guard<std::mutex> lock(s_planner_mutex);
            fftw_import_wisdom_from_filename(path.c_str()); // returns 0 on missing file, silently ok
        }

        void saveWisdomImpl(const std::string& path) {
            std::lock_guard<std::mutex> lock(s_planner_mutex);
            if (!fftw_export_wisdom_to_filename(path.c_str()))
                throw std::runtime_error("Failed to save FFTW wisdom to: " + path);
        }

    } // anonymous namespace

    namespace detail {
        std::mutex& fftwPlannerMutex() {
            return s_planner_mutex;
        }

        // map plan rigor to fftw flags
        unsigned int toFFTWFlag(PlanRigor r) {
            switch (r) {
                case PlanRigor::Estimate:   return FFTW_ESTIMATE;
                case PlanRigor::Measure:    return FFTW_MEASURE;
                case PlanRigor::Patient:    return FFTW_PATIENT;
                case PlanRigor::Exhaustive: return FFTW_EXHAUSTIVE;
            }
            throw std::invalid_argument("Unknown PlanRigor value");
        }

        int checkedProduct(const std::vector<int>& dims, const char* what) {
            long long total = 1;
            for (int d : dims) {
                if (d <= 0)
                    throw std::invalid_argument(std::string(what) + " dimensions must be positive");
                if (total > std::numeric_limits<int>::max() / d)
                    throw std::overflow_error(std::string(what) + " dimensions overflow int");
                total *= d;
            }
            return static_cast<int>(total);
        }

        int checkedMultiply(int a, int b, const char* what) {
            if (a < 0 || b < 0 || a > std::numeric_limits<int>::max() / b)
                throw std::overflow_error(std::string(what) + " size overflows int");
            return a * b;
        }

        // Caller must hold fftwPlannerMutex().
        void ensureDoubleThreadsInitializedLocked() {
            if (!s_fftw_threads_initialized) {
                if (fftw_init_threads() == 0)
                    throw std::runtime_error("FFTW failed to initialize double-precision threading");
                s_fftw_threads_initialized = true;
            }
            fftw_plan_with_nthreads(s_fftw_thread_count);
        }

        void* checkedFftwMalloc(std::size_t bytes) {
            if (bytes == 0) return nullptr;
            void* p = fftw_malloc(bytes);
            if (!p) throw std::bad_alloc();
            return p;
        }
    } // namespace detail

    void setFFTWThreadCount(int nthreads) {
        if (nthreads < 1)
            throw std::invalid_argument("FFTW thread count must be >= 1");

        std::lock_guard<std::mutex> lock(s_planner_mutex);
        s_fftw_thread_count = nthreads;
        detail::ensureDoubleThreadsInitializedLocked();
    }

    int getFFTWThreadCount() {
        std::lock_guard<std::mutex> lock(s_planner_mutex);
        return s_fftw_thread_count;
    }

    void* fftwAlignedMalloc(std::size_t bytes) {
        return detail::checkedFftwMalloc(bytes);
    }

    void fftwAlignedFree(void* p) noexcept {
        fftw_free(p);
    }

    // --- FFTW backend --------------------------------------------------------

    namespace {
        class FftwBackend final : public detail::FftBackend {
        public:
            FftwBackend(const std::vector<int>& dims, int howmany, PlanRigor rigor) {
                const int total = detail::checkedProduct(dims, "FFT");
                full_size_ = detail::checkedMultiply(total, howmany, "FFT");

                FftwBuf buf_in (static_cast<fftw_complex*>(detail::checkedFftwMalloc(sizeof(fftw_complex) * full_size_)));
                FftwBuf buf_out(static_cast<fftw_complex*>(detail::checkedFftwMalloc(sizeof(fftw_complex) * full_size_)));
                alignment_ = fftw_alignment_of(reinterpret_cast<double*>(buf_in.get()));

                const unsigned flags = detail::toFFTWFlag(rigor);

                std::lock_guard<std::mutex> lock(s_planner_mutex);
                detail::ensureDoubleThreadsInitializedLocked();

                forward_plan_ = PlanPtr(
                    fftw_plan_many_dft(
                        static_cast<int>(dims.size()), dims.data(), howmany,
                        buf_in.get(), nullptr, 1, total,
                        buf_out.get(), nullptr, 1, total,
                        FFTW_FORWARD, flags
                    )
                );

                inverse_plan_ = PlanPtr(
                    fftw_plan_many_dft(
                        static_cast<int>(dims.size()), dims.data(), howmany,
                        buf_in.get(), nullptr, 1, total,
                        buf_out.get(), nullptr, 1, total,
                        FFTW_BACKWARD, flags
                    )
                );

                if (!forward_plan_ || !inverse_plan_)
                    throw std::runtime_error("FFTW failed to create plan.");
            }

            void execute(const std::complex<double>* in, std::complex<double>* out, bool forward,
                         const Stream&) const override {
                execute_safe(forward ? forward_plan_.get() : inverse_plan_.get(), alignment_, full_size_, in, out);
            }

            void scale(std::complex<double>* out, std::size_t n, double s, const Stream&) const override {
                Eigen::Map<Eigen::VectorXcd>(out, static_cast<Eigen::Index>(n)) *= s;
            }

        private:
            PlanPtr forward_plan_;
            PlanPtr inverse_plan_;
            int full_size_ = 0;
            int alignment_ = 0;
        };
    } // anonymous namespace

    namespace detail {
        std::unique_ptr<FftBackend> makeFftwBackend(const std::vector<int>& dims, int howmany, PlanRigor rigor) {
            return std::make_unique<FftwBackend>(dims, howmany, rigor);
        }
    } // namespace detail

    // --- FFT -------------------------------------------------------------------

    struct FFT::Impl {
        std::unique_ptr<detail::FftBackend> backend;
        std::vector<int> dims;
        Device device;
        int howmany = 1;
        int total_size = 0; // product of dims
        Index full_size = 0; // total_size * howmany
    };

    FFT::FFT(std::vector<int> dims, int howmany, PlanRigor rigor, Device device)
        : impl_(std::make_unique<Impl>())
    {
        if (dims.empty() || dims.size() > 3)
            throw std::invalid_argument("Only ranks 1, 2 and 3 are supported.");
        if (howmany < 1)
            throw std::invalid_argument("howmany must be >= 1");

        // Both planner APIs (FFTW and cuFFT) take int sizes: validate once here.
        const int total = detail::checkedProduct(dims, "FFT");
        impl_->dims = std::move(dims);
        impl_->device = device;
        impl_->howmany = howmany;
        impl_->total_size = total;
        impl_->full_size = detail::checkedMultiply(total, howmany, "FFT");

        requireDevice(device);
        if (device.isCuda())
            impl_->backend = detail::makeCufftBackend(impl_->dims, howmany, device);
        else
            impl_->backend = detail::makeFftwBackend(impl_->dims, howmany, rigor);
    }

    FFT::~FFT() = default;
    FFT::FFT(FFT&&) noexcept = default;
    FFT& FFT::operator=(FFT&&) noexcept = default;

    Device FFT::device() const noexcept { return impl_->device; }
    const std::vector<int>& FFT::dims() const noexcept { return impl_->dims; }
    int FFT::howmany() const noexcept { return impl_->howmany; }
    Index FFT::size() const noexcept { return impl_->full_size; }

    Shape FFT::shape() const {
        Shape s;
        if (impl_->howmany > 1) s.push(impl_->howmany);
        for (int d : impl_->dims) s.push(d);
        return s;
    }

    // Raw interface
    void FFT::fft(const std::complex<double>* in, std::complex<double>* out, const Stream& stream) const {
        impl_->backend->execute(in, out, true, stream);
    }

    void FFT::ifft(const std::complex<double>* in, std::complex<double>* out, const Stream& stream) const {
        impl_->backend->execute(in, out, false, stream);
    }

    namespace {
        void checkView(const char* what, Index expected, Device device, Index got, Device viewDevice) {
            if (got != expected)
                throw std::invalid_argument(std::string(what) + ": view has " + std::to_string(got) +
                                            " elements, plan expects " + std::to_string(expected));
            if (viewDevice != device)
                throw std::invalid_argument(std::string(what) + ": view lives on " + toString(viewDevice) +
                                            " but the plan targets " + toString(device));
        }
    } // namespace

    void FFT::fft(BufferView<const std::complex<double>> in, BufferView<std::complex<double>> out,
                  const Stream& stream) const {
        checkView("FFT::fft in", impl_->full_size, impl_->device, in.size(), in.device());
        checkView("FFT::fft out", impl_->full_size, impl_->device, out.size(), out.device());
        impl_->backend->execute(in.data(), out.data(), true, stream);
    }

    void FFT::ifft(BufferView<const std::complex<double>> in, BufferView<std::complex<double>> out,
                   bool normalize, const Stream& stream) const {
        checkView("FFT::ifft in", impl_->full_size, impl_->device, in.size(), in.device());
        checkView("FFT::ifft out", impl_->full_size, impl_->device, out.size(), out.device());
        impl_->backend->execute(in.data(), out.data(), false, stream);
        if (normalize)
            impl_->backend->scale(out.data(), static_cast<std::size_t>(impl_->full_size),
                                  1.0 / static_cast<double>(impl_->total_size), stream);
    }

    // Convenience functions for eigen
    template<int Rank>
    void FFT::fft(const TensorXcd<Rank>& in, TensorXcd<Rank>& out) const {
        fft(toConstView(in), toView(out));
    }

    template<int Rank>
    void FFT::ifft(const TensorXcd<Rank>& in, TensorXcd<Rank>& out, bool normalize) const {
        ifft(toConstView(in), toView(out), normalize);
    }

    // explicit instantiations to avoid linker errors (templates defined in .cpp must be instantiated there)
    template void FFT::fft(const TensorXcd<1>&, TensorXcd<1>&) const;
    template void FFT::fft(const TensorXcd<2>&, TensorXcd<2>&) const;
    template void FFT::fft(const TensorXcd<3>&, TensorXcd<3>&) const;
    template void FFT::ifft(const TensorXcd<1>&, TensorXcd<1>&, bool) const;
    template void FFT::ifft(const TensorXcd<2>&, TensorXcd<2>&, bool) const;
    template void FFT::ifft(const TensorXcd<3>&, TensorXcd<3>&, bool) const;


    void FFT::loadWisdom(const std::string& path) { loadWisdomImpl(path); }

    void FFT::saveWisdom(const std::string& path) { saveWisdomImpl(path); }

#ifndef SIRIUS_HAS_CUDA
    namespace detail {
        std::unique_ptr<FftBackend> makeCufftBackend(const std::vector<int>&, int, Device device) {
            throw std::runtime_error("SIRIUS was built without CUDA support; cannot plan an FFT on " +
                                     toString(device));
        }
    } // namespace detail
#endif

} // namespace sirius
