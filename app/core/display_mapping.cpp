#include "core/display_mapping.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sirius::app {

    DisplayRange minMaxRange(const double* data, Index n) noexcept {
        double lo = std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        for (Index i = 0; i < n; ++i) {
            const double v = data[i];
            if (std::isnan(v)) continue;
            lo = v < lo ? v : lo;
            hi = v > hi ? v : hi;
        }
        if (!(lo <= hi)) return {0.0, 0.0};   // empty or all NaN
        return {lo, hi};
    }

    DisplayRange percentileRange(const double* data, Index n, double lowFrac, double highFrac,
                                 Index maxSamples) {
        if (n <= 0) return {0.0, 0.0};
        lowFrac = std::clamp(lowFrac, 0.0, 1.0);
        highFrac = std::clamp(highFrac, lowFrac, 1.0);
        maxSamples = std::max<Index>(maxSamples, 1);

        // fixed-stride subsample: bounded work, deterministic result
        const Index stride = std::max<Index>(1, (n + maxSamples - 1) / maxSamples);
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(n / stride + 1));
        for (Index i = 0; i < n; i += stride)
            if (!std::isnan(data[i])) samples.push_back(data[i]);
        if (samples.empty()) return {0.0, 0.0};

        auto rank = [&](double frac) {
            return static_cast<std::ptrdiff_t>(std::llround(frac * static_cast<double>(samples.size() - 1)));
        };
        const std::ptrdiff_t kLo = rank(lowFrac);
        const std::ptrdiff_t kHi = rank(highFrac);
        // Two partial selections instead of a full sort. After the first,
        // everything before kLo is <= samples[kLo], so the second only has to
        // partition the tail [kLo, end).
        std::nth_element(samples.begin(), samples.begin() + kLo, samples.end());
        const double lo = samples[static_cast<std::size_t>(kLo)];
        std::nth_element(samples.begin() + kLo, samples.begin() + kHi, samples.end());
        const double hi = samples[static_cast<std::size_t>(kHi)];
        if (hi > lo) return {lo, hi};
        return minMaxRange(data, n);   // flat quantiles (e.g. mostly-zero data)
    }

    void mapToGray8(const double* src, Index n, DisplayRange range, std::uint8_t* dst) noexcept {
        if (!range.valid()) {
            std::fill(dst, dst + n, std::uint8_t{0});
            return;
        }
        const double scale = 255.0 / (range.hi - range.lo);
        const double lo = range.lo;
        for (Index i = 0; i < n; ++i) {
            // std::max(0.0, NaN) is 0.0 (the first argument wins when the
            // comparison is false), which also gives NaN pixels a value
            const double t = std::min(255.0, std::max(0.0, (src[i] - lo) * scale));
            dst[i] = static_cast<std::uint8_t>(t + 0.5);
        }
    }

} // namespace sirius::app
