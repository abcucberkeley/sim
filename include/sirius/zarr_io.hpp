#ifndef SIRIUS_ZARR_IO_HPP
#define SIRIUS_ZARR_IO_HPP

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/tiff_io.hpp"   // PixelType

// Chunked array stores (zarr v2, zarr v3, N5) through TensorStore. Only
// compiled with SIRIUS_ENABLE_TENSORSTORE; without it every function throws
// std::runtime_error("built without TensorStore") and zarrSupported() is
// false, so callers can offer the feature conditionally.

namespace sirius {

    bool zarrSupported() noexcept;

    struct ZarrArrayInfo {
        std::string path;                 // store root ("/data/x.zarr" or ".../x.zarr/0")
        std::string driver;               // "zarr", "zarr3", "n5"
        std::vector<Index> shape;         // as stored
        std::vector<Index> chunks;
        PixelType pixelType = PixelType::Float32;
        std::vector<std::string> axes;    // OME-NGFF axis names when present ("t","c","z","y","x")
        std::vector<double> scale;        // OME-NGFF coordinate scale per axis (voxel size) when present
        std::vector<std::string> channelNames;
        std::vector<std::string> multiscalePaths;   // OME-NGFF pyramid datasets, level 0 first
        std::string codec;                // "blosc(zstd,3)", "gzip(6)", ...
        std::uint64_t bytesOnDisk = 0;
    };

    // Inspect a zarr / N5 store (an OME-NGFF group resolves to its level-0 array).
    ZarrArrayInfo inspectZarr(const std::string& path);

    // Read a hyper-rectangle [origin, origin + shape) of the array (level 0 or
    // one of multiscalePaths) into a host buffer of the requested type, in
    // the stored axis order. `shape` entries of 0 extend to the end.
    template <typename T>
    Buffer<T> readZarr(const std::string& path, const std::vector<Index>& origin, const std::vector<Index>& shape,
                       const std::string& levelPath = {});

    struct ZarrWriteOptions {
        int zarrVersion = 3;                            // 2, 3, or 0 for N5
        std::vector<Index> chunks;                      // per axis; empty = TensorStore default
        std::string codec = "blosc-zstd";               // "blosc-zstd", "blosc-lz4", "zstd", "gzip", "none"
        int level = 3;
        bool shard = false;                             // zarr 3: group chunks into shards of shardFactor^n chunks
        int shardFactor = 4;
        std::vector<std::string> axes;                  // OME-NGFF axis names (size == rank)
        std::vector<double> scale;                      // voxel size per axis
        std::vector<std::string> channelNames;
        int pyramidLevels = 1;                          // > 1 writes "0", "1", ... with OME-NGFF multiscales
        int downsample = 2;                             // per spatial axis, per level
        bool omeNgff = true;
        bool deleteExisting = true;
    };

    // Write a host array (any rank <= 5 given explicitly in `shape`) as a
    // store at `path`; `progress` receives 0..1.
    template <typename T>
    void writeZarr(const std::string& path, const T* data, const std::vector<Index>& shape,
                   const ZarrWriteOptions& options,
                   const std::function<void(double)>& progress = {});

} // namespace sirius

#endif // SIRIUS_ZARR_IO_HPP
