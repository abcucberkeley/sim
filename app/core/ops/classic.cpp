// Classical segmentation: the conventional recipe for nuclei and blobs
// without a model -- flatten the background (white top-hat), smooth
// (Gaussian), cut with a global (Otsu, percentile, manual) or local-mean
// threshold, clean the mask (binary opening, hole filling) and split it into
// instances by connected components or a distance watershed. Everything is
// per plane in (y, x) except the instance step, which is 3D.
#include "core/ops/builtin.hpp"

#include <algorithm>
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
                    intParam("tophat", "Background radius", 0).range(0, 2000).withUnit("px")
                        .withHelp("White top-hat: removes background structures larger than this radius (0 = off)"),
                    doubleParam("sigma", "Smoothing σ", 1.0).range(0.0, 50.0, 0.5, 1).withUnit("px")
                        .withHelp("Gaussian blur before the threshold (0 = none)"),
                    choiceParam("method", "Threshold", {"Otsu", "Manual", "Percentile", "Local mean"}, "Otsu"),
                    doubleParam("value", "Value", 0.5).range(-1e9, 1e9, 0.01, 4).withHelp("Manual threshold"),
                    doubleParam("percentile", "Percentile", 90.0).range(0.0, 100.0, 0.5, 1).withUnit("%"),
                    intParam("window", "Local window", 51).range(3, 4001).withUnit("px")
                        .withHelp("Local mean: side of the neighbourhood the mean is taken over"),
                    doubleParam("local_ratio", "Local ratio", 1.1).range(0.0, 10.0, 0.05, 2)
                        .withHelp("Local mean: foreground where value > ratio × local mean + offset"),
                    doubleParam("local_offset", "Local offset", 0.0).range(-1e9, 1e9, 0.01, 4).asAdvanced(),
                    intParam("opening", "Opening radius", 1).range(0, 100).withUnit("px")
                        .withHelp("Binary opening drops specks and necks thinner than this (0 = off)"),
                    boolParam("fill_holes", "Fill holes", true).withHelp("Enclosed background inside an object becomes object, per plane"),
                    choiceParam("post", "Instances", {"Watershed (distance)", "Connected components"}, "Watershed (distance)"),
                    doubleParam("seed_distance", "Seed distance", 8.0).range(1.0, 200.0, 0.5, 1).withUnit("px")
                        .withHelp("Watershed: minimum distance between the seeds that split touching objects").asAdvanced(),
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
                const double sigma = p.getDouble("sigma", 1.0);
                const Index tophat = p.getInt("tophat", 0);
                return joinSummary({channelName(meta, p.getInt("channel", 0)),
                                    tophat > 0 ? "top-hat " + std::to_string(tophat) : "",
                                    sigma > 0 ? "σ " + formatNumber(sigma, 1) : "",
                                    cut,
                                    p.getString("post", "Watershed (distance)").rfind("Watershed", 0) == 0 ? "watershed" : "components"});
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
                post.className = p.getString("class_name", "object");

                std::vector<float> work(static_cast<std::size_t>(n)), tmp, scratch, localMean;
                std::vector<double> integral;
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
                    // 1. flatten and smooth, per plane
                    for (Index z = 0; z < d.z; ++z) {
                        float* pl = work.data() + z * plane;
                        topHatPlane(pl, d.y, d.x, tophat, scratch, tmp);
                        gaussianPlane(pl, d.y, d.x, sigma, tmp);
                        if (z % 8 == 0) ctx.report(base + span * 0.3 * z / std::max<Index>(1, d.z), "filtering");
                    }
                    ctx.throwIfCancelled();
                    // 2. threshold
                    float cut = static_cast<float>(p.getDouble("value", 0.5));
                    if (method == "Otsu") cut = otsuThreshold(work.data(), n);
                    else if (method == "Percentile") cut = percentiles(work.data(), n, 0.0, p.getDouble("percentile", 90.0)).second;
                    if (method == "Local mean") {
                        localMean.resize(static_cast<std::size_t>(plane));
                        for (Index z = 0; z < d.z; ++z) {
                            const float* pl = work.data() + z * plane;
                            localMeanPlane(pl, d.y, d.x, window, localMean.data(), integral);
                            std::uint8_t* m = mask.data() + z * plane;
                            for (Index i = 0; i < plane; ++i) m[i] = pl[i] > ratio * localMean[static_cast<std::size_t>(i)] + offset ? 1 : 0;
                        }
                        if (t == 0) cutText = "local mean × " + formatNumber(ratio, 2);
                    } else {
                        for (Index i = 0; i < n; ++i) mask[static_cast<std::size_t>(i)] = work[static_cast<std::size_t>(i)] > cut ? 1 : 0;
                        if (t == 0) cutText = "> " + formatNumber(cut, 4);
                    }
                    ctx.report(base + span * 0.5, "mask");
                    // 3. clean the mask, per plane
                    for (Index z = 0; z < d.z; ++z) {
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
