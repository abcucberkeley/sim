#ifndef SIRIUS_PY_COMMON_HPP
#define SIRIUS_PY_COMMON_HPP

// Shared helpers for the nanobind modules: pixel-type dispatch and the
// Buffer <-> Python bridge (numpy on the host, DLPack-capable object on CUDA).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <sirius/buffer.hpp>
#include <sirius/device.hpp>
#include <sirius/tiff_io.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace sirius_py {

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

    // Python-side owner of a type-erased buffer. Exposed as sirius.Buffer.
    class PyBuffer {
    public:
        explicit PyBuffer(sirius::AnyBuffer buffer) : buffer_(std::move(buffer)) {}

        sirius::AnyBuffer& any() { return buffer_; }
        const sirius::AnyBuffer& any() const { return buffer_; }

        sirius::PixelType pixelType() const;
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
        sirius::AnyBuffer buffer_;
    };

    // Move an owning buffer to Python: numpy (zero-copy) for host memory,
    // sirius.Buffer for device memory.
    nb::object toPython(sirius::AnyBuffer&& buffer);

} // namespace sirius_py

#endif // SIRIUS_PY_COMMON_HPP
