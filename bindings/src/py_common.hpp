#ifndef SIRIUS_PY_COMMON_HPP
#define SIRIUS_PY_COMMON_HPP

// Shared helpers for the nanobind modules: pixel-type dispatch and the
// Buffer <-> Python bridge (numpy on the host, DLPack-capable object on CUDA).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <sirius/buffer.hpp>
#include <sirius/device.hpp>
#include <sirius/tiff_io.hpp>

#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace nb = nanobind;

namespace sirius_py {

    // Element types a sirius.Buffer can hold: the TIFF pixel types
    // (sirius::AnyBuffer) plus the complex scalars of the FFT API.
    using AnyPyBuffer = std::variant<sirius::Buffer<std::uint8_t>, sirius::Buffer<std::int8_t>,
                                     sirius::Buffer<std::uint16_t>, sirius::Buffer<std::int16_t>,
                                     sirius::Buffer<std::uint32_t>, sirius::Buffer<std::int32_t>,
                                     sirius::Buffer<float>, sirius::Buffer<double>,
                                     sirius::Buffer<std::complex<float>>, sirius::Buffer<std::complex<double>>>;

    // numpy dtype name of an element type.
    template <typename T>
    constexpr const char* dtypeNameOf() {
        if constexpr (std::is_same_v<T, std::uint8_t>)               return "uint8";
        else if constexpr (std::is_same_v<T, std::int8_t>)           return "int8";
        else if constexpr (std::is_same_v<T, std::uint16_t>)         return "uint16";
        else if constexpr (std::is_same_v<T, std::int16_t>)          return "int16";
        else if constexpr (std::is_same_v<T, std::uint32_t>)         return "uint32";
        else if constexpr (std::is_same_v<T, std::int32_t>)          return "int32";
        else if constexpr (std::is_same_v<T, float>)                 return "float32";
        else if constexpr (std::is_same_v<T, double>)                return "float64";
        else if constexpr (std::is_same_v<T, std::complex<float>>)   return "complex64";
        else if constexpr (std::is_same_v<T, std::complex<double>>)  return "complex128";
        else static_assert(sizeof(T) == 0, "unsupported element type");
    }

    // Call f(T{}) for the scalar type behind a PixelType.
    template <typename F>
    decltype(auto) withPixelType(sirius::PixelType t, F&& f) {
        using sirius::PixelType;
        switch (t) {
            case PixelType::UInt8:   return f(std::uint8_t{});
            case PixelType::Int8:    return f(std::int8_t{});
            case PixelType::UInt16:  return f(std::uint16_t{});
            case PixelType::Int16:   return f(std::int16_t{});
            case PixelType::UInt32:  return f(std::uint32_t{});
            case PixelType::Int32:   return f(std::int32_t{});
            case PixelType::Float32: return f(float{});
            case PixelType::Float64: return f(double{});
        }
        throw std::invalid_argument("unknown pixel type");
    }

    // numpy dtype (object, string, or None -> nullopt) to PixelType.
    std::optional<sirius::PixelType> pixelTypeFromDtype(nb::handle dtype);
    nb::object dtypeObject(sirius::PixelType t);
    nb::object dtypeObject(const char* name);

    inline AnyPyBuffer widen(sirius::AnyBuffer&& buffer) {
        return std::visit([](auto&& b) -> AnyPyBuffer { return std::move(b); }, std::move(buffer));
    }

    // Python-side owner of a type-erased buffer. Exposed as sirius.Buffer.
    class PyBuffer {
    public:
        explicit PyBuffer(AnyPyBuffer buffer) : buffer_(std::move(buffer)) {}
        explicit PyBuffer(sirius::AnyBuffer buffer) : buffer_(widen(std::move(buffer))) {}

        AnyPyBuffer& any() { return buffer_; }
        const AnyPyBuffer& any() const { return buffer_; }

        const char* dtypeName() const;
        nb::object dtype() const;
        sirius::Shape shape() const;
        sirius::Device device() const;
        std::size_t nbytes() const;
        std::size_t size() const;
        bool pinned() const;

        PyBuffer to(sirius::Device device) const;
        // Host copy as a numpy array (always a copy; safe to keep after the buffer dies).
        nb::object numpy() const;
        // DLPack ("dltensor") capsule sharing this buffer's memory without a
        // copy; the capsule holds a reference to `self`.
        nb::object ndarray(nb::handle self) const;

    private:
        AnyPyBuffer buffer_;
    };

    // Move an owning buffer to Python: numpy (zero-copy) for host memory,
    // sirius.Buffer for device memory.
    nb::object toPython(AnyPyBuffer&& buffer);
    inline nb::object toPython(sirius::AnyBuffer&& buffer) { return toPython(widen(std::move(buffer))); }

} // namespace sirius_py

#endif // SIRIUS_PY_COMMON_HPP
