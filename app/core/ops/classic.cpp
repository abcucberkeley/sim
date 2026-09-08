// Classical segmentation: the conventional recipe for nuclei and blobs
// without a model -- flatten the background (white top-hat), smooth
// (Gaussian), cut with a global (Otsu, percentile, manual) or local-mean
// threshold, clean the mask (binary opening, hole filling) and split it into
// instances by connected components or a distance watershed. Everything is
// per plane in (y, x) except the instance step, which is 3D.
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <utility>
#include <limits>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <vector>

#include <sirius/image_ops.hpp>

namespace sirius::app {

    namespace {

        inline Index reflect(Index i, Index n) {
            if (n <= 1) return 0;
            while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - i - 2;
            return i;
        }

        // Separable Gaussian on one (y, x) plane, borders reflected.
        void gaussianPlane(float* plane, Index y, Index x, double sigma, std::vector<float>& tmp) {
            if (sigma <= 0.0) return;
            const Index r = std::max<Index>(1, static_cast<Index>(std::ceil(3.0 * sigma)));
            std::vector<float> k(static_cast<std::size_t>(2 * r + 1));
            double sum = 0.0;
            for (Index i = -r; i <= r; ++i) {
                k[static_cast<std::size_t>(i + r)] = static_cast<float>(std::exp(-0.5 * (i * i) / (sigma * sigma)));
                sum += k[static_cast<std::size_t>(i + r)];
            }
            for (float& v : k) v = static_cast<float>(v / sum);
            tmp.resize(static_cast<std::size_t>(y * x));
            for (Index yy = 0; yy < y; ++yy) {
                const float* src = plane + yy * x;
                float* dst = tmp.data() + yy * x;
                for (Index xx = 0; xx < x; ++xx) {
                    float acc = 0.0f;
                    for (Index i = -r; i <= r; ++i) acc += src[reflect(xx + i, x)] * k[static_cast<std::size_t>(i + r)];
                    dst[xx] = acc;
                }
            }
            for (Index yy = 0; yy < y; ++yy) {
                float* dst = plane + yy * x;
                for (Index xx = 0; xx < x; ++xx) {
                    float acc = 0.0f;
                    for (Index i = -r; i <= r; ++i) acc += tmp[static_cast<std::size_t>(reflect(yy + i, y) * x + xx)] * k[static_cast<std::size_t>(i + r)];
                    dst[xx] = acc;
                }
            }
        }

        // Sliding-window min or max over one line (stride between elements),
        // window 2r+1, borders clamped: monotonic deque, O(n).
        template <typename T, bool Max>
        void slideLine(const T* src, T* dst, Index n, Index stride, Index r) {
            std::deque<Index> q;
            auto better = [](T a, T b) { return Max ? a >= b : a <= b; };
            Index next = 0;
            for (Index i = 0; i < n; ++i) {
                const Index hi = std::min(n - 1, i + r);
                for (; next <= hi; ++next) {
                    while (!q.empty() && better(src[next * stride], src[q.back() * stride])) q.pop_back();
                    q.push_back(next);
                }
                while (!q.empty() && q.front() < i - r) q.pop_front();
                dst[i * stride] = src[q.front() * stride];
            }
        }

        template <typename T, bool Max>
        void boxFilterPlane(T* plane, Index y, Index x, Index r, std::vector<T>& tmp) {
            if (r <= 0) return;
            tmp.resize(static_cast<std::size_t>(y * x));
            for (Index yy = 0; yy < y; ++yy) slideLine<T, Max>(plane + yy * x, tmp.data() + yy * x, x, 1, r);
            for (Index xx = 0; xx < x; ++xx) slideLine<T, Max>(tmp.data() + xx, plane + xx, y, x, r);
        }

        // White top-hat: image minus its opening (erode, then dilate) with a
        // (2r+1)² box: what is smaller than the box survives, the rest is background.
        void topHatPlane(float* plane, Index y, Index x, Index r, std::vector<float>& work, std::vector<float>& tmp) {
            if (r <= 0) return;
            work.assign(plane, plane + y * x);
            boxFilterPlane<float, false>(work.data(), y, x, r, tmp);
            boxFilterPlane<float, true>(work.data(), y, x, r, tmp);
            for (Index i = 0; i < y * x; ++i) plane[i] = std::max(0.0f, plane[i] - work[static_cast<std::size_t>(i)]);
        }

        // Mean over a (2r+1)² window through an integral image, borders clamped.
        void localMeanPlane(const float* plane, Index y, Index x, Index r, float* out, std::vector<double>& integral) {
            const Index W = x + 1;
            integral.assign(static_cast<std::size_t>((y + 1) * W), 0.0);
            for (Index yy = 0; yy < y; ++yy) {
                double row = 0.0;
                for (Index xx = 0; xx < x; ++xx) {
                    row += plane[yy * x + xx];
                    integral[static_cast<std::size_t>((yy + 1) * W + xx + 1)] = integral[static_cast<std::size_t>(yy * W + xx + 1)] + row;
                }
            }
            for (Index yy = 0; yy < y; ++yy) {
                const Index y0 = std::max<Index>(0, yy - r), y1 = std::min(y, yy + r + 1);
                for (Index xx = 0; xx < x; ++xx) {
                    const Index x0 = std::max<Index>(0, xx - r), x1 = std::min(x, xx + r + 1);
                    const double s = integral[static_cast<std::size_t>(y1 * W + x1)] - integral[static_cast<std::size_t>(y0 * W + x1)] -
                                     integral[static_cast<std::size_t>(y1 * W + x0)] + integral[static_cast<std::size_t>(y0 * W + x0)];
                    out[yy * x + xx] = static_cast<float>(s / static_cast<double>((y1 - y0) * (x1 - x0)));
                }
            }
        }

        // Background connected to the plane border stays background; every
        // other 0 (enclosed) becomes foreground.
        void fillHolesPlane(std::uint8_t* mask, Index y, Index x, std::vector<std::uint8_t>& seen, std::vector<Index>& stack) {
            seen.assign(static_cast<std::size_t>(y * x), 0);
            stack.clear();
            auto push = [&](Index i) {
                if (!seen[static_cast<std::size_t>(i)] && mask[i] == 0) {
                    seen[static_cast<std::size_t>(i)] = 1;
                    stack.push_back(i);
                }
            };
            for (Index xx = 0; xx < x; ++xx) {
                push(xx);
                push((y - 1) * x + xx);
            }
            for (Index yy = 0; yy < y; ++yy) {
                push(yy * x);
                push(yy * x + x - 1);
            }
            while (!stack.empty()) {
                const Index i = stack.back();
                stack.pop_back();
                const Index yy = i / x, xx = i % x;
                if (xx > 0) push(i - 1);
                if (xx + 1 < x) push(i + 1);
                if (yy > 0) push(i - x);
                if (yy + 1 < y) push(i + x);
            }
            for (Index i = 0; i < y * x; ++i)
                if (mask[i] == 0 && !seen[static_cast<std::size_t>(i)]) mask[i] = 1;
        }

        // Local mean and standard deviation over a (2r+1)^2 window, both from
        // integral images of the value and its square: O(1) per pixel.
        void localStatsPlane(const float* plane, Index y, Index x, Index r, float* mean, float* stddev,
                             std::vector<double>& sum, std::vector<double>& sumSq) {
            const Index W = x + 1;
            sum.assign(static_cast<std::size_t>((y + 1) * W), 0.0);
            sumSq.assign(static_cast<std::size_t>((y + 1) * W), 0.0);
            for (Index yy = 0; yy < y; ++yy) {
                double row = 0.0, rowSq = 0.0;
                for (Index xx = 0; xx < x; ++xx) {
                    const double v = plane[yy * x + xx];
                    row += v;
                    rowSq += v * v;
                    sum[static_cast<std::size_t>((yy + 1) * W + xx + 1)] = sum[static_cast<std::size_t>(yy * W + xx + 1)] + row;
                    sumSq[static_cast<std::size_t>((yy + 1) * W + xx + 1)] = sumSq[static_cast<std::size_t>(yy * W + xx + 1)] + rowSq;
                }
            }
            auto box = [&](const std::vector<double>& in, Index y0, Index y1, Index x0, Index x1) {
                return in[static_cast<std::size_t>(y1 * W + x1)] - in[static_cast<std::size_t>(y0 * W + x1)] -
                       in[static_cast<std::size_t>(y1 * W + x0)] + in[static_cast<std::size_t>(y0 * W + x0)];
            };
            for (Index yy = 0; yy < y; ++yy) {
                const Index y0 = std::max<Index>(0, yy - r), y1 = std::min(y, yy + r + 1);
                for (Index xx = 0; xx < x; ++xx) {
                    const Index x0 = std::max<Index>(0, xx - r), x1 = std::min(x, xx + r + 1);
                    const double count = static_cast<double>((y1 - y0) * (x1 - x0));
                    const double m = box(sum, y0, y1, x0, x1) / count;
                    const double var = std::max(0.0, box(sumSq, y0, y1, x0, x1) / count - m * m);
                    mean[yy * x + xx] = static_cast<float>(m);
                    stddev[yy * x + xx] = static_cast<float>(std::sqrt(var));
                }
            }
        }

        // Two Otsu thresholds (three classes) by exhaustive search over a
        // 128-bin histogram: the upper cut keeps only the brightest class,
        // which separates objects from a bright halo the single cut merges.
        std::pair<float, float> multiOtsuThresholds(const float* v, Index n) {
            float mn = std::numeric_limits<float>::infinity(), mx = -mn;
            for (Index i = 0; i < n; ++i) {
                if (std::isnan(v[i])) continue;
                mn = std::min(mn, v[i]);
                mx = std::max(mx, v[i]);
            }
            if (!(mx > mn)) return {mn, mn};
            constexpr int bins = 128;
            const std::vector<double> h = histogram(v, n, bins, mn, mx);
            std::array<double, bins + 1> w{}, m{};
            for (int i = 0; i < bins; ++i) {
                w[static_cast<std::size_t>(i + 1)] = w[static_cast<std::size_t>(i)] + h[static_cast<std::size_t>(i)];
                m[static_cast<std::size_t>(i + 1)] = m[static_cast<std::size_t>(i)] + i * h[static_cast<std::size_t>(i)];
            }
            const double total = w[bins], mean = m[bins];
            double best = -1.0;
            int bestA = bins / 3, bestB = 2 * bins / 3;
            if (!(total > 0.0)) return {mn, mx};
            for (int a = 1; a < bins - 1; ++a)
                for (int b = a + 1; b < bins; ++b) {
                    const double w0 = w[static_cast<std::size_t>(a)], w1 = w[static_cast<std::size_t>(b)] - w0, w2 = total - w[static_cast<std::size_t>(b)];
                    if (w0 <= 0.0 || w1 <= 0.0 || w2 <= 0.0) continue;
                    const double m0 = m[static_cast<std::size_t>(a)] / w0;
                    const double m1 = (m[static_cast<std::size_t>(b)] - m[static_cast<std::size_t>(a)]) / w1;
                    const double m2 = (mean - m[static_cast<std::size_t>(b)]) / w2;
                    const double between = w0 * (m0 - mean / total) * (m0 - mean / total) +
                                           w1 * (m1 - mean / total) * (m1 - mean / total) +
                                           w2 * (m2 - mean / total) * (m2 - mean / total);
                    if (between > best) {
                        best = between;
                        bestA = a;
                        bestB = b;
                    }
                }
            const float lo = mn + (mx - mn) * static_cast<float>(bestA + 1) / bins;
            const float hi = mn + (mx - mn) * static_cast<float>(bestB + 1) / bins;
            return {lo, hi};
        }

        // Difference of Gaussians: a band-pass that answers to blobs about
        // `sigma` across and flattens everything larger, so nuclei of one size
        // survive a textured background.
        void dogPlane(const float* src, float* dst, Index y, Index x, double sigma, double ratio, std::vector<float>& a,
                      std::vector<float>& b, std::vector<float>& tmp) {
            const Index n = y * x;
            a.assign(src, src + n);
            b.assign(src, src + n);
            gaussianPlane(a.data(), y, x, sigma, tmp);
            gaussianPlane(b.data(), y, x, sigma * std::max(1.1, ratio), tmp);
            for (Index i = 0; i < n; ++i) dst[i] = std::max(0.0f, a[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)]);
        }

        // Frangi vesselness in the plane: the Hessian's eigenvalues at several
        // scales say how tube-like each pixel is, which finds filaments that a
        // threshold on intensity alone breaks into dashes.
        void frangiPlane(const float* src, float* dst, Index y, Index x, double sigmaMin, double sigmaMax, int steps,
                         std::vector<float>& work, std::vector<float>& tmp) {
            const Index n = y * x;
            std::fill(dst, dst + n, 0.0f);
            steps = std::max(1, steps);
            for (int k = 0; k < steps; ++k) {
                const double sigma = steps == 1 ? sigmaMin
                                                : sigmaMin + (sigmaMax - sigmaMin) * static_cast<double>(k) / (steps - 1);
                work.assign(src, src + n);
                gaussianPlane(work.data(), y, x, sigma, tmp);
                const double norm = sigma * sigma;   // scale normalisation
                // second derivatives, then the eigenvalues of [[xx, xy], [xy, yy]]
                double maxS = 0.0;
                std::vector<float> vesselness(static_cast<std::size_t>(n), 0.0f);
                std::vector<double> sVals(static_cast<std::size_t>(n), 0.0);
                for (Index yy = 0; yy < y; ++yy)
                    for (Index xx = 0; xx < x; ++xx) {
                        const Index i = yy * x + xx;
                        const Index xm = std::max<Index>(0, xx - 1), xp = std::min(x - 1, xx + 1);
                        const Index ym = std::max<Index>(0, yy - 1), yp = std::min(y - 1, yy + 1);
                        const double c = work[static_cast<std::size_t>(i)];
                        const double dxx = work[static_cast<std::size_t>(yy * x + xp)] + work[static_cast<std::size_t>(yy * x + xm)] - 2.0 * c;
                        const double dyy = work[static_cast<std::size_t>(yp * x + xx)] + work[static_cast<std::size_t>(ym * x + xx)] - 2.0 * c;
                        const double dxy = 0.25 * (work[static_cast<std::size_t>(yp * x + xp)] + work[static_cast<std::size_t>(ym * x + xm)] -
                                                   work[static_cast<std::size_t>(yp * x + xm)] - work[static_cast<std::size_t>(ym * x + xp)]);
                        const double a = norm * dxx, b = norm * dyy, cxy = norm * dxy;
                        const double t = std::sqrt((a - b) * (a - b) + 4.0 * cxy * cxy);
                        double l1 = 0.5 * (a + b + t), l2 = 0.5 * (a + b - t);
                        if (std::abs(l1) > std::abs(l2)) std::swap(l1, l2);   // |l1| <= |l2|
                        if (l2 >= 0.0) continue;                              // dark ridge: not a bright tube
                        const double rb = std::abs(l1) / std::max(std::abs(l2), 1e-12);
                        const double sMag = std::sqrt(l1 * l1 + l2 * l2);
                        sVals[static_cast<std::size_t>(i)] = sMag;
                        maxS = std::max(maxS, sMag);
                        vesselness[static_cast<std::size_t>(i)] = static_cast<float>(std::exp(-rb * rb / 0.5));
                    }
                const double c2 = 2.0 * std::max(1e-12, 0.5 * maxS) * std::max(1e-12, 0.5 * maxS);
                for (Index i = 0; i < n; ++i) {
                    if (vesselness[static_cast<std::size_t>(i)] <= 0.0f) continue;
                    const double sMag = sVals[static_cast<std::size_t>(i)];
                    const float v = static_cast<float>(vesselness[static_cast<std::size_t>(i)] * (1.0 - std::exp(-sMag * sMag / c2)));
                    dst[i] = std::max(dst[i], v);
                }
            }
        }

        class ClassicalSegmentationOperation final : public Operation {
        public:
            ClassicalSegmentationOperation() {
                info_.kind = "classic";
                info_.name = "Classical segmentation";
                info_.group = "Segment";
                info_.kindLabel = "SEGMENT";
                info_.diagnostics = DiagnosticsKind::Segment;
                info_.defaultCache = CachePolicy::Memory;
                info_.separableOverT = true;
                info_.producesLabels = true;
                info_.helpPage = "classic";
                info_.params = {
                    channelParam("channel", "Channel", 0),
                    choiceParam("enhance", "Enhance", {"None", "Blobs (DoG)", "Tubes (Frangi)"}, "None")
                        .withHelp("What to bring out before the threshold: round objects of one size, or filaments"),
                    doubleParam("enhance_sigma", "Feature σ", 2.0).range(0.3, 100.0, 0.5, 1).withUnit("px")
                        .withHelp("Blobs: the radius they respond to. Tubes: the smallest tube width"),
                    doubleParam("enhance_sigma_max", "Feature σ max", 6.0).range(0.3, 100.0, 0.5, 1).withUnit("px")
                        .withHelp("Tubes: the largest width; the response is the best over the range").asAdvanced(),
                    intParam("enhance_scales", "Scales", 4).range(1, 16)
                        .withHelp("Tubes: how many widths between the two σ").asAdvanced(),
                    intParam("tophat", "Background radius", 0).range(0, 2000).withUnit("px")
                        .withHelp("White top-hat: removes background structures larger than this radius (0 = off)"),
                    doubleParam("sigma", "Smoothing σ", 1.0).range(0.0, 50.0, 0.5, 1).withUnit("px")
                        .withHelp("Gaussian blur before the threshold (0 = none)"),
                    choiceParam("method", "Threshold", {"Otsu", "Multi-Otsu", "Manual", "Percentile", "Local mean", "Local contrast"}, "Otsu"),
                    doubleParam("value", "Value", 0.5).range(-1e9, 1e9, 0.01, 4).withHelp("Manual threshold"),
                    doubleParam("percentile", "Percentile", 90.0).range(0.0, 100.0, 0.5, 1).withUnit("%"),
                    intParam("window", "Local window", 51).range(3, 4001).withUnit("px")
                        .withHelp("Local mean: side of the neighbourhood the mean is taken over"),
                    doubleParam("local_ratio", "Local ratio", 1.1).range(0.0, 10.0, 0.05, 2)
                        .withHelp("Local mean: foreground where value > ratio × local mean + offset"),
                    doubleParam("local_offset", "Local offset", 0.0).range(-1e9, 1e9, 0.01, 4).asAdvanced(),
                    doubleParam("contrast_k", "Contrast k", 1.5).range(0.0, 10.0, 0.1, 2)
                        .withHelp("Local contrast: the cut sits k local standard deviations above the local mean, so it "
                                  "follows both the background level and the local noise"),
                    intParam("opening", "Opening radius", 1).range(0, 100).withUnit("px")
                        .withHelp("Binary opening drops specks and necks thinner than this (0 = off)"),
                    boolParam("fill_holes", "Fill holes", true).withHelp("Enclosed background inside an object becomes object, per plane"),
                    choiceParam("post", "Instances", {"Watershed (distance)", "Connected components"}, "Watershed (distance)"),
                    choiceParam("seeds", "Seeds", {"Distance maxima", "H-maxima"}, "H-maxima")
                        .withHelp("What splits touching objects: the peaks of the distance map, or only the peaks that "
                                  "stand clear of their surroundings (fewer false splits)"),
                    doubleParam("seed_distance", "Seed distance", 8.0).range(1.0, 200.0, 0.5, 1).withUnit("px")
                        .withHelp("Distance maxima: minimum distance between seeds").asAdvanced(),
                    doubleParam("seed_depth", "Seed depth", 2.0).range(0.1, 100.0, 0.5, 1).withUnit("px")
                        .withHelp("H-maxima: how far a peak must stand above its surroundings to be its own object; "
                                  "raise it when one object is split, lower it when two are merged"),
                    intParam("min_voxels", "Min. voxels", 20).range(0, 1000000000),
                    stringParam("class_name", "Class", "object").asAdvanced(),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& p, const DatasetMeta& meta) const override {
                const std::string method = p.getString("method", "Otsu");
                std::string cut = method;
                if (method == "Manual") cut = "> " + formatNumber(p.getDouble("value", 0.5), 3);
                else if (method == "Percentile") cut = "> p" + formatNumber(p.getDouble("percentile", 90.0), 0);
                else if (method == "Local mean") cut = "local " + std::to_string(p.getInt("window", 51)) + " px";
                else if (method == "Local contrast") cut = "local ± " + std::to_string(p.getInt("window", 51)) + " px";
                const std::string enhance = p.getString("enhance", "None");
                const std::string enhanceText =
                    enhance == "Blobs (DoG)"      ? "blobs σ " + formatNumber(p.getDouble("enhance_sigma", 2.0), 1)
                    : enhance == "Tubes (Frangi)" ? "tubes σ " + formatNumber(p.getDouble("enhance_sigma", 2.0), 1) + "-" +
                                                        formatNumber(p.getDouble("enhance_sigma_max", 6.0), 1)
                                                  : std::string();
                const double sigma = p.getDouble("sigma", 1.0);
                const Index tophat = p.getInt("tophat", 0);
                return joinSummary({channelName(meta, p.getInt("channel", 0)), enhanceText,
                                    tophat > 0 ? "top-hat " + std::to_string(tophat) : "",
                                    sigma > 0 ? "σ " + formatNumber(sigma, 1) : "",
                                    cut,
                                    p.getString("post", "Watershed (distance)").rfind("Watershed", 0) != 0 ? "components"
                                    : p.getString("seeds", "H-maxima") == "H-maxima"                       ? "watershed h"
                                                                                                          : "watershed d"});
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                const Validation v = validate(p, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                const DatasetMeta& meta = input.meta;
                const Dims5& d = meta.dims;
                const Index channel = p.getInt("channel", 0);
                const Index n = d.z * d.y * d.x, plane = d.y * d.x;
                StepOutput out;
                out.meta = meta;
                out.array = input.materialize([&](double f, const std::string& m) { ctx.report(0.1 * f, m); });
                auto labels = std::make_shared<LabelVolume>(d.t, d.z, d.y, d.x);

                const Index tophat = p.getInt("tophat", 0);
                const double sigma = p.getDouble("sigma", 1.0);
                const std::string enhance = p.getString("enhance", "None");
                const double enhanceSigma = p.getDouble("enhance_sigma", 2.0);
                const double enhanceSigmaMax = p.getDouble("enhance_sigma_max", 6.0);
                const int enhanceScales = static_cast<int>(p.getInt("enhance_scales", 4));
                const double contrastK = p.getDouble("contrast_k", 1.5);
                const std::string method = p.getString("method", "Otsu");
                const Index window = std::max<Index>(1, p.getInt("window", 51) / 2);
                const double ratio = p.getDouble("local_ratio", 1.1), offset = p.getDouble("local_offset", 0.0);
                const Index opening = p.getInt("opening", 1);
                const bool fillHoles = p.getBool("fill_holes", true);

                LabelPostOptions post;
                post.post = p.getString("post", "Watershed (distance)").rfind("Watershed", 0) == 0 ? "Watershed (distance)"
                                                                                                     : "Connected components";
                post.threshold = 0.5;
                post.minVoxels = p.getInt("min_voxels", 20);
                post.seedMinDistance = p.getDouble("seed_distance", 8.0);
                post.seeds = p.getString("seeds", "H-maxima");
                post.seedDepth = p.getDouble("seed_depth", 2.0);
                post.className = p.getString("class_name", "object");

                std::vector<float> work(static_cast<std::size_t>(n)), tmp, scratch, localMean, localStd, enhA, enhB;
                std::vector<double> integral, integralSq;
                std::vector<std::uint8_t> mask(static_cast<std::size_t>(n)), maskTmp, seen;
                std::vector<Index> stack;
                std::vector<float> fg(static_cast<std::size_t>(n));
                std::uint32_t total = 0;
                std::string cutText;
                double foregroundFraction = 0.0;
                for (Index t = 0; t < d.t; ++t) {
                    ctx.throwIfCancelled();
                    const double base = 0.1 + 0.9 * static_cast<double>(t) / d.t, span = 0.9 / d.t;
                    const BufferView<const float> vol = out.array->volume(channel, t);
                    std::copy_n(vol.data(), n, work.data());
                    // 1. enhance the feature, flatten and smooth, per plane
                    for (Index z = 0; z < d.z; ++z) {
                        ctx.throwIfCancelled();
                        float* pl = work.data() + z * plane;
                        if (enhance != "None") {
                            enhA.resize(static_cast<std::size_t>(plane));
                            if (enhance == "Blobs (DoG)")
                                dogPlane(pl, enhA.data(), d.y, d.x, enhanceSigma, 1.6, scratch, enhB, tmp);
                            else
                                frangiPlane(pl, enhA.data(), d.y, d.x, enhanceSigma, std::max(enhanceSigma, enhanceSigmaMax),
                                            enhanceScales, scratch, tmp);
                            std::copy_n(enhA.data(), plane, pl);
                        }
                        topHatPlane(pl, d.y, d.x, tophat, scratch, tmp);
                        gaussianPlane(pl, d.y, d.x, sigma, tmp);
                        if (z % 8 == 0) ctx.report(base + span * 0.3 * z / std::max<Index>(1, d.z), "filtering");
                    }
                    ctx.throwIfCancelled();
                    // 2. threshold
                    float cut = static_cast<float>(p.getDouble("value", 0.5));
                    if (method == "Otsu") cut = otsuThreshold(work.data(), n);
                    else if (method == "Multi-Otsu") cut = multiOtsuThresholds(work.data(), n).second;
                    else if (method == "Percentile") cut = percentiles(work.data(), n, 0.0, p.getDouble("percentile", 90.0)).second;
                    if (method == "Local mean" || method == "Local contrast") {
                        const bool byContrast = method == "Local contrast";
                        localMean.resize(static_cast<std::size_t>(plane));
                        if (byContrast) localStd.resize(static_cast<std::size_t>(plane));
                        for (Index z = 0; z < d.z; ++z) {
                            ctx.throwIfCancelled();
                            const float* pl = work.data() + z * plane;
                            std::uint8_t* m = mask.data() + z * plane;
                            if (byContrast) {
                                // k standard deviations above the local mean: the cut
                                // rises with the background and with the local noise,
                                // so a flat region does not turn into speckle
                                localStatsPlane(pl, d.y, d.x, window, localMean.data(), localStd.data(), integral, integralSq);
                                for (Index i = 0; i < plane; ++i) {
                                    const double mu = localMean[static_cast<std::size_t>(i)];
                                    const double sd = localStd[static_cast<std::size_t>(i)];
                                    m[i] = pl[i] > mu + contrastK * sd + offset ? 1 : 0;
                                }
                            } else {
                                localMeanPlane(pl, d.y, d.x, window, localMean.data(), integral);
                                for (Index i = 0; i < plane; ++i)
                                    m[i] = pl[i] > ratio * localMean[static_cast<std::size_t>(i)] + offset ? 1 : 0;
                            }
                        }
                        if (t == 0) cutText = byContrast ? "local mean + " + formatNumber(contrastK, 2) + " SD"
                                                         : "local mean × " + formatNumber(ratio, 2);
                    } else {
                        for (Index i = 0; i < n; ++i) mask[static_cast<std::size_t>(i)] = work[static_cast<std::size_t>(i)] > cut ? 1 : 0;
                        if (t == 0) cutText = "> " + formatNumber(cut, 4);
                    }
                    ctx.report(base + span * 0.5, "mask");
                    // 3. clean the mask, per plane
                    for (Index z = 0; z < d.z; ++z) {
                        ctx.throwIfCancelled();
                        std::uint8_t* m = mask.data() + z * plane;
                        if (opening > 0) {
                            boxFilterPlane<std::uint8_t, false>(m, d.y, d.x, opening, maskTmp);
                            boxFilterPlane<std::uint8_t, true>(m, d.y, d.x, opening, maskTmp);
                        }
                        if (fillHoles) fillHolesPlane(m, d.y, d.x, seen, stack);
                    }
                    ctx.throwIfCancelled();
                    Index on = 0;
                    for (Index i = 0; i < n; ++i) {
                        fg[static_cast<std::size_t>(i)] = mask[static_cast<std::size_t>(i)] ? 1.0f : 0.0f;
                        on += mask[static_cast<std::size_t>(i)];
                    }
                    foregroundFraction += static_cast<double>(on) / static_cast<double>(std::max<Index>(1, n)) / d.t;
                    ctx.report(base + span * 0.6, "instances");
                    // 4. instances
                    total += labelsFromProbabilities(fg.data(), nullptr, d.z, d.y, d.x, post, *labels, t);
                }
                for (LabelStats& s : labels->stats()) s.confidence = 1.0;   // intensities, not probabilities
                labels->applyFlags(post.flags);
                out.labels = labels;
                out.ranOn = Backend::Cpu;
                out.note = cutText + " · " + std::to_string(total) + " labels · CPU";
                out.diagnostics = labelDiagnostics(*labels, summary(p, meta) + " · " + std::to_string(total) + " labels");
                out.diagnostics.facts.push_back({"Threshold", cutText});
                out.diagnostics.facts.push_back({"Foreground", formatNumber(100.0 * foregroundFraction, 1) + " %"});
                ctx.report(1.0, "");
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeClassicalSegmentationOperation() { return std::make_unique<ClassicalSegmentationOperation>(); }

} // namespace sirius::app
