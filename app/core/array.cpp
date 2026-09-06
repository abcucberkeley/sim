#include "core/array.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace sirius::app {

    char axisLetter(Axis a) noexcept {
        switch (a) {
            case Axis::C: return 'c';
            case Axis::T: return 't';
            case Axis::Z: return 'z';
            case Axis::Y: return 'y';
            case Axis::X: return 'x';
        }
        return '?';
    }

    std::optional<Axis> axisFromLetter(char letter) noexcept {
        switch (letter) {
            case 'c': case 'C': return Axis::C;
            case 't': case 'T': return Axis::T;
            case 'z': case 'Z': return Axis::Z;
            case 'y': case 'Y': return Axis::Y;
            case 'x': case 'X': return Axis::X;
            default: return std::nullopt;
        }
    }

    Index Dims5::operator[](Axis a) const noexcept {
        switch (a) {
            case Axis::C: return c;
            case Axis::T: return t;
            case Axis::Z: return z;
            case Axis::Y: return y;
            case Axis::X: return x;
        }
        return 0;
    }

    Index& Dims5::operator[](Axis a) noexcept {
        switch (a) {
            case Axis::C: return c;
            case Axis::T: return t;
            case Axis::Z: return z;
            case Axis::Y: return y;
            default: return x;
        }
    }

    std::string Dims5::toString() const {
        return "c" + std::to_string(c) + " t" + std::to_string(t) + " z" + std::to_string(z) + " y" +
               std::to_string(y) + " x" + std::to_string(x);
    }

    std::string Dims5::toProduct() const {
        return std::to_string(c) + " × " + std::to_string(t) + " × " + std::to_string(z) + " × " +
               std::to_string(y) + " × " + std::to_string(x);
    }

    Array5::Array5(Dims5 dims) : dims_(dims) {
        if (dims.c < 1 || dims.t < 1 || dims.z < 1 || dims.y < 1 || dims.x < 1)
            throw std::invalid_argument("Array5: every extent must be >= 1, got " + dims.toString());
        data_ = Buffer<float>(Shape{dims.planes(), dims.y, dims.x});
    }

    Array5 Array5::zeros(Dims5 dims) { return filled(dims, 0.0f); }

    Array5 Array5::filled(Dims5 dims, float value) {
        Array5 a(dims);
        std::fill(a.data(), a.data() + a.numel(), value);
        return a;
    }

    Array5 Array5::fromBuffer(Buffer<float> buffer, std::optional<Dims5> dims) {
        if (buffer.empty()) throw std::invalid_argument("Array5::fromBuffer: empty buffer");
        if (!buffer.device().isCpu()) throw std::invalid_argument("Array5::fromBuffer: buffer must be on the host");
        const Shape s = buffer.shape().asStack();
        Dims5 d;
        if (dims) {
            d = *dims;
            if (d.planes() != s[0] || d.y != s[1] || d.x != s[2])
                throw std::invalid_argument("Array5::fromBuffer: dims " + d.toString() + " do not match buffer " +
                                            buffer.shape().toString());
        } else {
            d = Dims5{1, 1, s[0], s[1], s[2]};
        }
        Array5 a;
        a.dims_ = d;
        a.data_ = std::move(buffer);
        a.data_.reshape(Shape{d.planes(), d.y, d.x});
        return a;
    }

    float* Array5::plane(Index c, Index t, Index z) noexcept {
        return data_.data() + dims_.planeIndex(c, t, z) * dims_.planeSize();
    }
    const float* Array5::plane(Index c, Index t, Index z) const noexcept {
        return data_.data() + dims_.planeIndex(c, t, z) * dims_.planeSize();
    }
    float& Array5::at(Index c, Index t, Index z, Index y, Index x) noexcept { return plane(c, t, z)[y * dims_.x + x]; }
    float Array5::at(Index c, Index t, Index z, Index y, Index x) const noexcept {
        return plane(c, t, z)[y * dims_.x + x];
    }

    BufferView<float> Array5::volume(Index c, Index t) noexcept {
        return BufferView<float>(plane(c, t, 0), Shape{dims_.z, dims_.y, dims_.x}, Device::cpu());
    }
    BufferView<const float> Array5::volume(Index c, Index t) const noexcept {
        return BufferView<const float>(plane(c, t, 0), Shape{dims_.z, dims_.y, dims_.x}, Device::cpu());
    }

    Array5 Array5::clone() const {
        Array5 a;
        a.dims_ = dims_;
        a.data_ = data_.clone();
        return a;
    }

    std::pair<float, float> minMax(const Array5& a) noexcept {
        float lo = std::numeric_limits<float>::infinity(), hi = -std::numeric_limits<float>::infinity();
        const float* p = a.data();
        const Index n = a.numel();
        for (Index i = 0; i < n; ++i) {
            const float v = p[i];
            if (std::isnan(v)) continue;
            lo = v < lo ? v : lo;
            hi = v > hi ? v : hi;
        }
        if (!(lo <= hi)) return {0.0f, 0.0f};
        return {lo, hi};
    }

} // namespace sirius::app
