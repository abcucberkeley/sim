// Contrast: linear rescale between two percentiles of the intensity
// histogram, then a gamma, per channel. The histograms it shows are also
// available as a live preview before the step runs.
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <sirius/image_ops.hpp>

#include "core/array_source.hpp"

namespace sirius::app {

    namespace {

        struct ChannelWindow {
            float lo = 0.0f, hi = 1.0f;
            DiagnosticHistogram histogram;
        };

        // Histogram + percentile window of channel `c` from at most
        // `maxPlanes` planes spread over t and z (all of them when 0).
        ChannelWindow windowOf(const StepInput& input, Index c, double loPct, double hiPct, double gamma,
                               Index maxPlanes) {
            const Dims5& d = input.meta.dims;
            const Index planes = d.t * d.z;
            const Index step = maxPlanes > 0 ? std::max<Index>(1, (planes + maxPlanes - 1) / maxPlanes) : 1;
            std::vector<float> samples;
            samples.reserve(static_cast<std::size_t>(((planes + step - 1) / step) * d.planeSize()));
            std::vector<float> plane(static_cast<std::size_t>(d.planeSize()));
            for (Index k = 0; k < planes; k += step) {
                const Index t = k / d.z, z = k % d.z;
                const float* src = nullptr;
                if (input.hasArray()) src = input.array->plane(c, t, z);
                else if (input.source) {
                    input.source->readPlane(c, t, z, plane.data());
                    src = plane.data();
                }
                if (!src) break;
                samples.insert(samples.end(), src, src + d.planeSize());
            }
            ChannelWindow w;
            const Index n = static_cast<Index>(samples.size());
            if (n == 0) return w;
            const auto pct = percentiles(samples.data(), n, loPct, hiPct);
            w.lo = pct.first;
            w.hi = pct.second;
            float mn = std::numeric_limits<float>::infinity(), mx = -mn;
            for (float v : samples) {
                if (std::isnan(v)) continue;
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            if (!(mn <= mx)) { mn = 0.0f; mx = 1.0f; }
            if (mx <= mn) mx = mn + 1.0f;
            w.histogram.bins = histogram(samples.data(), n, 30, mn, mx);
            w.histogram.binLo = mn;
            w.histogram.binHi = mx;
            w.histogram.lo = w.lo;
            w.histogram.hi = w.hi;
            w.histogram.gamma = gamma;
            if (c >= 0 && static_cast<std::size_t>(c) < input.meta.channels.size()) {
                w.histogram.channel = input.meta.channels[static_cast<std::size_t>(c)].label;
                w.histogram.color = input.meta.channels[static_cast<std::size_t>(c)].color;
            } else {
                w.histogram.channel = "ch " + std::to_string(c);
            }
            return w;
        }

        Diagnostics contrastDiagnostics(const StepInput& input, const ParamSet& params, Index maxPlanes,
                                        const std::string& summary) {
            Diagnostics d;
            d.kind = DiagnosticsKind::Contrast;
            d.summary = summary;
            const double lo = params.getDouble("lo_percentile", 0.2), hi = params.getDouble("hi_percentile", 99.8);
            const double gamma = params.getDouble("gamma", 1.0);
            for (Index c = 0; c < input.meta.dims.c; ++c)
                d.histograms.push_back(windowOf(input, c, lo, hi, gamma, maxPlanes).histogram);
            return d;
        }

        class ContrastOperation final : public Operation {
        public:
            ContrastOperation() {
                info_.kind = "contrast";
                info_.name = "Contrast";
                info_.group = "Intensity";
                info_.kindLabel = "INTENSITY";
                info_.diagnostics = DiagnosticsKind::Contrast;
                info_.defaultCache = CachePolicy::Recompute;
                info_.separableOverT = false;   // the window spans every time point
                info_.helpPage = "contrast";
                info_.params = {
                    doubleParam("lo_percentile", "Low percentile", 0.2).range(0.0, 50.0, 0.1, 2).withUnit("%"),
                    doubleParam("hi_percentile", "High percentile", 99.8).range(50.0, 100.0, 0.1, 2).withUnit("%"),
                    doubleParam("gamma", "Gamma", 1.0).range(0.1, 5.0, 0.05, 2),
                    boolParam("per_channel", "Per channel", true).withHelp("Separate window for every channel"),
                    boolParam("bake", "Bake into data", true)
                        .withHelp("The step rewrites intensities into 0..1; kept for future display-only use")
                        .asAdvanced(),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            // Histograms update live while the percentiles are dragged.
            std::optional<Diagnostics> preview(const StepInput& input, const ParamSet& params) const override {
                if (!input.hasArray() && !input.source) return std::nullopt;
                return contrastPreview(input, params);
            }

            std::string summary(const ParamSet& params, const DatasetMeta&) const override {
                char buf[64];
                std::snprintf(buf, sizeof buf, "percentile %.1f – %.1f", params.getDouble("lo_percentile", 0.2),
                              params.getDouble("hi_percentile", 99.8));
                const double gamma = params.getDouble("gamma", 1.0);
                char g[32];
                std::snprintf(g, sizeof g, "γ %.2g", gamma);
                return joinSummary({buf, params.getBool("per_channel", true) ? "per channel" : "global",
                                    gamma != 1.0 ? g : ""});
            }

            Validation validate(const ParamSet& params, const DatasetMeta& input) const override {
                Validation v = Operation::validate(params, input);
                if (params.getDouble("lo_percentile") >= params.getDouble("hi_percentile"))
                    v.errors.push_back("The low percentile must be below the high one.");
                return v;
            }

            DatasetMeta outputMeta(const ParamSet&, const DatasetMeta& input) const override {
                DatasetMeta out = input;
                out.sourceType = PixelType::Float32;
                return out;
            }

            StepOutput run(const StepInput& input, const ParamSet& params, const StepContext& ctx) const override {
                const Validation v = validate(params, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                const DatasetMeta& meta = input.meta;
                StepOutput out;
                out.meta = outputMeta(params, meta);
                ctx.report(0.0, "reading");
                ArrayPtr in = input.materialize([&](double f, const std::string& m) { ctx.report(0.4 * f, m); });
                auto result = std::make_shared<Array5>(in->clone());
                const double lo = params.getDouble("lo_percentile", 0.2), hi = params.getDouble("hi_percentile", 99.8);
                const float gamma = static_cast<float>(params.getDouble("gamma", 1.0));
                const bool perChannel = params.getBool("per_channel", true);
                const Dims5& d = meta.dims;
                const Index channelSize = d.t * d.z * d.planeSize();
                StepInput full{meta, result, nullptr, input.labels};
                std::string note;
                if (perChannel) {
                    for (Index c = 0; c < d.c; ++c) {
                        ctx.throwIfCancelled();
                        ctx.report(0.4 + 0.6 * static_cast<double>(c) / d.c, "channel " + std::to_string(c));
                        const ChannelWindow w = windowOf(full, c, lo, hi, gamma, 0);
                        float* p = result->plane(c, 0, 0);
                        rescaleGamma(p, channelSize, w.lo, w.hi, gamma);
                        char buf[64];
                        std::snprintf(buf, sizeof buf, "%s%g – %g", c ? " · " : "", w.lo, w.hi);
                        note += buf;
                    }
                } else {
                    const auto pct = percentiles(result->data(), result->numel(), lo, hi);
                    rescaleGamma(result->data(), result->numel(), pct.first, pct.second, gamma);
                    char buf[64];
                    std::snprintf(buf, sizeof buf, "%g – %g", pct.first, pct.second);
                    note = buf;
                }
                out.array = result;
                out.labels = input.labels ? input.labels->clone() : nullptr;
                out.ranOn = Backend::Cpu;
                out.note = note + " · CPU";
                out.diagnostics = contrastDiagnostics(StepInput{meta, in, nullptr, nullptr}, params, 0,
                                                      summary(params, meta));
                ctx.report(1.0, "");
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    Diagnostics contrastPreview(const StepInput& input, const ParamSet& params) {
        // 8 planes per channel keep this under ~100 ms on 2048² planes
        ParamSet p = params;
        if (const Operation* op = findOperation("contrast")) p.applyDefaults(op->info().params);
        return contrastDiagnostics(input, p, 8, findOperation("contrast") ? findOperation("contrast")->summary(p, input.meta) : "");
    }

    std::unique_ptr<Operation> makeContrastOperation() { return std::make_unique<ContrastOperation>(); }

} // namespace sirius::app
