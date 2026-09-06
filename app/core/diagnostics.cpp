#include "core/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include <sirius/fft.hpp>

namespace sirius::app {

    DiagnosticImage spectrumImage(const float* plane, Index rows, Index cols, std::string title, std::string meta) {
        DiagnosticImage img;
        img.title = std::move(title);
        img.meta = std::move(meta);
        img.logScale = true;
        if (rows <= 0 || cols <= 0) return img;
        // Diagnostics are thumbnails: cap the transform size so a 4096²
        // plane does not cost a full FFT per parameter change.
        constexpr Index kMax = 512;
        const Index f = std::max<Index>(1, std::max((rows + kMax - 1) / kMax, (cols + kMax - 1) / kMax));
        const Index r = rows / f, c = cols / f;
        if (r <= 1 || c <= 1) return img;
        std::vector<std::complex<double>> in(static_cast<std::size_t>(r * c)), out(in.size());
        for (Index y = 0; y < r; ++y)
            for (Index x = 0; x < c; ++x) {
                double acc = 0.0;
                for (Index dy = 0; dy < f; ++dy)
                    for (Index dx = 0; dx < f; ++dx) acc += plane[(y * f + dy) * cols + x * f + dx];
                in[static_cast<std::size_t>(y * c + x)] = acc / static_cast<double>(f * f);
            }
        FFT fft(std::vector<int>{static_cast<int>(r), static_cast<int>(c)}, 1, PlanRigor::Estimate);
        fft.fft(in.data(), out.data());
        img.rows = r;
        img.cols = c;
        img.values.resize(static_cast<std::size_t>(r * c));
        for (Index y = 0; y < r; ++y) {
            const Index yo = (y + r / 2) % r;
            for (Index x = 0; x < c; ++x)
                img.values[static_cast<std::size_t>(yo * c + (x + c / 2) % c)] =
                    static_cast<float>(std::log10(std::abs(out[static_cast<std::size_t>(y * c + x)]) + 1e-12));
        }
        return img;
    }

    DiagnosticImage thumbnail(const float* plane, Index rows, Index cols, Index maxSide, std::string title,
                              std::string meta) {
        DiagnosticImage img;
        img.title = std::move(title);
        img.meta = std::move(meta);
        if (rows <= 0 || cols <= 0) return img;
        const Index f = std::max<Index>(1, (std::max(rows, cols) + maxSide - 1) / maxSide);
        const Index r = std::max<Index>(1, rows / f), c = std::max<Index>(1, cols / f);
        img.rows = r;
        img.cols = c;
        img.values.resize(static_cast<std::size_t>(r * c));
        for (Index y = 0; y < r; ++y)
            for (Index x = 0; x < c; ++x) {
                double acc = 0.0;
                Index n = 0;
                for (Index dy = 0; dy < f && y * f + dy < rows; ++dy)
                    for (Index dx = 0; dx < f && x * f + dx < cols; ++dx, ++n)
                        acc += plane[(y * f + dy) * cols + x * f + dx];
                img.values[static_cast<std::size_t>(y * c + x)] = n ? static_cast<float>(acc / n) : 0.0f;
            }
        return img;
    }

    DiagnosticHistogram histogramOf(const float* values, Index n, int bins) {
        DiagnosticHistogram h;
        bins = std::max(bins, 1);
        h.bins.assign(static_cast<std::size_t>(bins), 0.0);
        float lo = std::numeric_limits<float>::infinity(), hi = -lo;
        for (Index i = 0; i < n; ++i) {
            const float v = values[i];
            if (std::isnan(v)) continue;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        if (!(lo <= hi)) return h;
        h.binLo = lo;
        h.binHi = hi;
        h.lo = lo;
        h.hi = hi;
        const double scale = hi > lo ? bins / static_cast<double>(hi - lo) : 0.0;
        for (Index i = 0; i < n; ++i) {
            const float v = values[i];
            if (std::isnan(v)) continue;
            int b = static_cast<int>((v - lo) * scale);
            b = std::clamp(b, 0, bins - 1);
            h.bins[static_cast<std::size_t>(b)] += 1.0;
        }
        return h;
    }

} // namespace sirius::app
