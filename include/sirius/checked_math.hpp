#ifndef SIRIUS_CHECKED_MATH_HPP
#define SIRIUS_CHECKED_MATH_HPP

// Overflow-checked size arithmetic. Element counts and byte sizes are formed
// from user-supplied extents (a TIFF header, a zarr shape, a Python tuple)
// right before an allocation; a silently wrapped product would allocate far
// too little and every later access would run off the end. These helpers
// throw std::overflow_error, naming the extents, instead.

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace sirius::detail {

    // a * b for non-negative counts, or std::overflow_error mentioning `what`.
    inline std::ptrdiff_t checkedMul(std::ptrdiff_t a, std::ptrdiff_t b, const char* what) {
        if (a < 0 || b < 0)
            throw std::invalid_argument(std::string(what) + ": negative extent in " + std::to_string(a) + " * " +
                                        std::to_string(b));
        if (b != 0 && a > std::numeric_limits<std::ptrdiff_t>::max() / b)
            throw std::overflow_error(std::string(what) + ": " + std::to_string(a) + " * " + std::to_string(b) +
                                      " overflows a 64-bit element count");
        return a * b;
    }

    // Product of the extents in [first, last) (1 when empty), or
    // std::overflow_error listing them.
    template <typename It>
    std::ptrdiff_t checkedProduct(It first, It last, const char* what) {
        std::ptrdiff_t total = 1;
        for (It it = first; it != last; ++it) {
            const std::ptrdiff_t d = static_cast<std::ptrdiff_t>(*it);
            if (d < 0) throw std::invalid_argument(std::string(what) + ": negative extent " + std::to_string(d));
            if (d != 0 && total > std::numeric_limits<std::ptrdiff_t>::max() / d) {
                std::string dims;
                for (It jt = first; jt != last; ++jt)
                    dims += (dims.empty() ? "" : " x ") + std::to_string(static_cast<std::ptrdiff_t>(*jt));
                throw std::overflow_error(std::string(what) + ": extents " + dims +
                                          " overflow a 64-bit element count");
            }
            total *= d;
        }
        return total;
    }

    // Byte size of `count` elements of `elemBytes` each, or std::overflow_error.
    inline std::size_t checkedBytes(std::ptrdiff_t count, std::size_t elemBytes, const char* what) {
        if (count < 0)
            throw std::invalid_argument(std::string(what) + ": negative element count " + std::to_string(count));
        const std::size_t n = static_cast<std::size_t>(count);
        if (elemBytes != 0 && n > std::numeric_limits<std::size_t>::max() / elemBytes)
            throw std::overflow_error(std::string(what) + ": " + std::to_string(n) + " elements of " +
                                      std::to_string(elemBytes) + " bytes overflow a size_t");
        return n * elemBytes;
    }

} // namespace sirius::detail

#endif // SIRIUS_CHECKED_MATH_HPP
