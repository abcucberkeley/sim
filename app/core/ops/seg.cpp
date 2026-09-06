// Torch segmentation: a TorchScript model run tile-wise by the Python worker
// (the same worker serves the HPC backend), probabilities turned into
// instance labels natively.
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>

namespace sirius::app {

    namespace {

        constexpr const char* kWatershed = "Watershed on boundary channel";
        constexpr const char* kComponents = "Connected components";
        constexpr const char* kNone = "None (raw probabilities)";

        std::string shapeText(const nlohmann::json& shape) {
            if (!shape.is_array()) return "?";
            std::string out = "(";
            for (std::size_t i = 0; i < shape.size(); ++i) {
                if (i) out += ", ";
                const nlohmann::json& e = shape[i];
                if (e.is_number()) out += std::to_string(e.get<long long>());
                else if (e.is_string()) out += e.get<std::string>();
                else out += "?";
            }
            return out + ")";
        }

        class TorchSegmentationOperation final : public Operation {
        public:
            TorchSegmentationOperation() {
                info_.kind = "seg";
                info_.name = "Torch segmentation";
                info_.group = "Segment";
                info_.kindLabel = "SEGMENT";
                info_.diagnostics = DiagnosticsKind::Segment;
                info_.defaultCache = CachePolicy::Disk;
                info_.separableOverT = true;
                info_.hasGpuPath = true;
                info_.remoteCapable = true;
                info_.producesLabels = true;
                info_.helpPage = "seg";
                info_.params = {
                    pathParam("model", "Torch model").withFilter("Models (*.pt *.pts *.pth *.onnx);;All files (*)")
                        .withHelp("TorchScript (or ONNX) model taking (1, 1, Z, Y, X) float32"),
                    channelParam("input_channel", "Input channel", 0),
                    doubleListParam("tile", "Tile", {32.0, 256.0, 256.0}).withUnit("px")
                        .withHelp("Tile extent (z, y, x); must fit GPU memory"),
                    intParam("overlap", "Overlap", 32).range(0, 512).withUnit("px")
                        .withHelp("Tile halo; should exceed the model's receptive-field radius"),
                    doubleParam("threshold", "Threshold", 0.5).range(0.0, 1.0, 0.01, 2)
                        .withHelp("Foreground probability cut"),
                    choiceParam("post", "Post-processing", {kWatershed, kComponents, kNone}, kWatershed),
                    intParam("min_voxels", "Min. voxels", 0).range(0, 1000000000).withHelp("Drop smaller objects (0 = keep all)"),
                    doubleParam("label_opacity", "Label opacity", 0.45).range(0.0, 1.0, 0.05, 2),
                    stringParam("class_name", "Class", "nucleus").asAdvanced(),
                    doubleParam("seed_distance", "Seed distance", 5.0).range(1.0, 200.0, 0.5, 1).withUnit("px")
                        .withHelp("Minimum distance between watershed seeds").asAdvanced(),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& p, const DatasetMeta&) const override {
                const std::string model = p.getString("model");
                std::string post = p.getString("post", kWatershed);
                post = post.rfind("Watershed", 0) == 0 ? "watershed" : post == kComponents ? "components" : "probabilities";
                return joinSummary({model.empty() ? "no model" : std::filesystem::path(model).filename().string(), post});
            }

            Validation validate(const ParamSet& p, const DatasetMeta& in) const override {
                Validation v = Operation::validate(p, in);
                const std::string model = p.getString("model");
                if (model.empty()) v.errors.push_back("Choose a Torch model.");
                else if (!std::filesystem::exists(model)) v.errors.push_back("Model not found: " + model);
                if (in.rgb) v.errors.push_back("Segmentation needs an intensity channel, not an RGB merge.");
                const std::vector<double> tile = p.getDoubleList("tile");
                if (tile.size() != 3 || std::any_of(tile.begin(), tile.end(), [](double d) { return d < 1; }))
                    v.errors.push_back("Tile must be three positive extents (z, y, x).");
                return v;
            }

            DatasetMeta outputMeta(const ParamSet&, const DatasetMeta& in) const override { return in; }

            std::size_t estimatedOutputBytes(const ParamSet&, const DatasetMeta& in) const override {
                return in.dims.bytes() + static_cast<std::size_t>(in.dims.t * in.dims.z * in.dims.planeSize()) * sizeof(std::uint32_t);
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                const Validation v = validate(p, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                if (!ctx.remote)
                    throw std::runtime_error("Torch segmentation needs the Python worker: start it from Preferences ▸ Worker "
                                             "or choose the HPC backend");
                if (!ctx.remote->supports("torch_segment"))
                    throw std::runtime_error("The connected worker does not implement torch_segment (" +
                                             ctx.remote->capabilities().hostname + ")");
                const DatasetMeta& meta = input.meta;
                const Dims5& d = meta.dims;
                const Index channel = p.getInt("input_channel", 0);
                const std::vector<double> tile = p.getDoubleList("tile");

                StepOutput out;
                out.meta = meta;
                out.array = input.materialize([&](double f, const std::string& m) { ctx.report(0.05 * f, m); });
                auto labels = std::make_shared<LabelVolume>(d.t, d.z, d.y, d.x);

                LabelPostOptions post;
                post.post = p.getString("post", kWatershed);
                post.threshold = p.getDouble("threshold", 0.5);
                post.minVoxels = p.getInt("min_voxels", 0);
                post.seedMinDistance = p.getDouble("seed_distance", 5.0);
                post.className = p.getString("class_name", "nucleus");

                nlohmann::json params = {
                    {"model", p.getString("model")},
                    {"tile", {static_cast<Index>(tile[0]), static_cast<Index>(tile[1]), static_cast<Index>(tile[2])}},
                    {"overlap", p.getInt("overlap", 32)},
                    {"device", ctx.backend == Backend::Cpu ? "cpu" : "auto"},
                };
                double seconds = 0.0;
                std::uint32_t total = 0;
                std::string classes;
                for (Index t = 0; t < d.t; ++t) {
                    ctx.throwIfCancelled();
                    const double base = 0.05 + 0.9 * static_cast<double>(t) / d.t, span = 0.9 / d.t;
                    const BufferView<const float> vol = out.array->volume(channel, t);
                    rpc::TensorRef in;
                    in.name = "input";
                    in.dtype = "float32";
                    in.shape = {d.z, d.y, d.x};
                    in.data = vol.data();
                    in.nbytes = vol.bytes();
                    const auto t0 = std::chrono::steady_clock::now();
                    WorkerResult r = ctx.remote->call(
                        "run", {{"kind", "torch_segment"}, {"params", params}}, {in},
                        [&](double f, const std::string& m) { ctx.report(base + span * 0.8 * f, m); },
                        [&] { return ctx.isCancelled(); });
                    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    ctx.throwIfCancelled();
                    const rpc::Tensor* prob = nullptr;
                    for (const rpc::Tensor& tensor : r.tensors)
                        if (tensor.name == "prob") prob = &tensor;
                    if (!prob) throw std::runtime_error("the worker returned no 'prob' tensor");
                    if (prob->shape.size() != 4 || prob->shape[1] != d.z || prob->shape[2] != d.y || prob->shape[3] != d.x)
                        throw std::runtime_error("the worker's probabilities do not match the volume");
                    const float* fg = prob->asFloat32();
                    const Index volSize = d.z * d.planeSize();
                    const float* boundary = prob->shape[0] > 1 ? fg + volSize : nullptr;
                    if (r.result.contains("class_names") && r.result["class_names"].is_array() && !r.result["class_names"].empty())
                        post.className = r.result["class_names"][0].get<std::string>();
                    ctx.report(base + span * 0.85, "labelling");
                    total += labelsFromProbabilities(fg, boundary, d.z, d.y, d.x, post, *labels, t);
                    if (classes.empty() && r.result.contains("model")) classes = r.result["model"].dump();
                }
                out.labels = labels;
                out.ranOn = ctx.backend;
                out.seconds = seconds;
                char note[160];
                std::snprintf(note, sizeof note, "%.1f s · %u labels · %s", seconds, total,
                              ctx.remote->capabilities().device.empty() ? "worker" : ctx.remote->capabilities().device.c_str());
                out.note = note;
                out.diagnostics = labelDiagnostics(*labels, summary(p, meta));
                out.diagnostics.summary = summary(p, meta) + " · " + std::to_string(total) + " labels";
                ctx.report(1.0, "");
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    nlohmann::json torchModelInfo(RemoteWorker& worker, const std::string& modelPath) {
        WorkerResult r = worker.call("model_info", {{"model", modelPath}});
        return r.result;
    }

    std::string torchModelSummary(const nlohmann::json& info) {
        if (!info.is_object()) return "no model";
        std::string out = info.value("format", "TorchScript");
        if (info.contains("input_shape")) {
            out += " · in " + shapeText(info["input_shape"]);
            if (info.contains("input_dtype")) out += " " + info["input_dtype"].get<std::string>();
        }
        if (info.contains("output_shape")) out += " · out " + shapeText(info["output_shape"]);
        if (info.contains("size_bytes") && info["size_bytes"].is_number())
            out += " · " + formatBytes(info["size_bytes"].get<std::uint64_t>());
        return out;
    }

    std::unique_ptr<Operation> makeTorchSegmentationOperation() { return std::make_unique<TorchSegmentationOperation>(); }

} // namespace sirius::app
