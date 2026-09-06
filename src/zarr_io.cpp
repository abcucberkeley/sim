// zarr / N5 through TensorStore. Placeholder until the I/O module lands: every
// entry point reports the missing build option.
#include "sirius/zarr_io.hpp"
#include <stdexcept>
namespace sirius {
    bool zarrSupported() noexcept { return false; }
    ZarrArrayInfo inspectZarr(const std::string&) { throw std::runtime_error("built without TensorStore"); }
    template <typename T>
    Buffer<T> readZarr(const std::string&, const std::vector<Index>&, const std::vector<Index>&, const std::string&) {
        throw std::runtime_error("built without TensorStore");
    }
    template <typename T>
    void writeZarr(const std::string&, const T*, const std::vector<Index>&, const ZarrWriteOptions&,
                   const std::function<void(double)>&) {
        throw std::runtime_error("built without TensorStore");
    }
#define SIRIUS_ZARR_INSTANTIATE(T) \
    template Buffer<T> readZarr<T>(const std::string&, const std::vector<Index>&, const std::vector<Index>&, const std::string&); \
    template void writeZarr<T>(const std::string&, const T*, const std::vector<Index>&, const ZarrWriteOptions&, const std::function<void(double)>&);
    SIRIUS_ZARR_INSTANTIATE(std::uint8_t) SIRIUS_ZARR_INSTANTIATE(std::int8_t) SIRIUS_ZARR_INSTANTIATE(std::uint16_t)
    SIRIUS_ZARR_INSTANTIATE(std::int16_t) SIRIUS_ZARR_INSTANTIATE(std::uint32_t) SIRIUS_ZARR_INSTANTIATE(std::int32_t)
    SIRIUS_ZARR_INSTANTIATE(float) SIRIUS_ZARR_INSTANTIATE(double)
}
