#include "core/workbench.hpp"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <stdexcept>

#include "core/ops/builtin.hpp"
#include "core/ops/plugin.hpp"

namespace sirius::app {

    using json = nlohmann::json;

    // --- enums ------------------------------------------------------------------

    const char* toString(ViewMode m) noexcept {
        switch (m) {
            case ViewMode::Ortho: return "ortho";
            case ViewMode::Volume: return "3d";
            case ViewMode::Compare: return "compare";
        }
        return "?";
    }
    const char* toString(ViewerTool t) noexcept {
        switch (t) {
            case ViewerTool::Navigate: return "nav";
            case ViewerTool::Probe: return "probe";
            case ViewerTool::Measure: return "measure";
            case ViewerTool::Roi: return "roi";
            case ViewerTool::Paint: return "paint";
        }
        return "?";
    }
    const char* toString(PaintTool t) noexcept {
        switch (t) {
            case PaintTool::Brush: return "brush";
            case PaintTool::Erase: return "erase";
            case PaintTool::Fill: return "fill";
            case PaintTool::Pick: return "pick";
            case PaintTool::Merge: return "merge";
            case PaintTool::Split: return "split";
            case PaintTool::Delete: return "delete";
            case PaintTool::Lasso: return "lasso";
        }
        return "?";
    }

    namespace {
        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        }
    } // namespace

    std::optional<ViewMode> viewModeFromString(const std::string& s) noexcept {
        const std::string l = lower(s);
        if (l == "ortho" || l == "2d") return ViewMode::Ortho;
        if (l == "3d" || l == "volume") return ViewMode::Volume;
        if (l == "compare") return ViewMode::Compare;
        return std::nullopt;
    }
    std::optional<ViewerTool> viewerToolFromString(const std::string& s) noexcept {
        const std::string l = lower(s);
        if (l == "nav" || l == "navigate") return ViewerTool::Navigate;
        if (l == "probe") return ViewerTool::Probe;
        if (l == "measure") return ViewerTool::Measure;
        if (l == "roi") return ViewerTool::Roi;
        if (l == "paint") return ViewerTool::Paint;
        return std::nullopt;
    }
    std::optional<PaintTool> paintToolFromString(const std::string& s) noexcept {
        static const PaintTool all[] = {PaintTool::Brush, PaintTool::Erase, PaintTool::Fill, PaintTool::Pick,
                                        PaintTool::Merge, PaintTool::Split, PaintTool::Delete, PaintTool::Lasso};
        const std::string l = lower(s);
        for (PaintTool t : all)
            if (l == toString(t)) return t;
        return std::nullopt;
    }

    // --- ViewState ---------------------------------------------------------------

    json ViewState::toJson() const {
        return {{"mode", toString(mode)},
                {"tool", toString(tool)},
                {"paint_tool", toString(paintTool)},
                {"brush_px", brushPx},
                {"paint_3d", paint3d},
                {"z", z},
                {"t", t},
                {"crosshair_x", cx},
                {"crosshair_y", cy},
                {"crosshair", crosshair},
                {"labels", labels},
                {"bounding_box", boundingBox},
                {"scale_bar", scaleBar},
                {"sync_zt", syncZT},
                {"channels", channelVisible},
                {"zoom", zoom},
                {"pan", {panX, panY}},
                {"yaw", yaw},
                {"pitch", pitch},
                {"clip_z", {clipZ[0], clipZ[1]}},
                {"label_opacity", labelOpacity},
                {"selected_label", selectedLabel}};
    }

    ViewState ViewState::fromJson(const json& j) { return fromJson(j, ViewState{}); }

    ViewState ViewState::fromJson(const json& j, const ViewState& base) {
        ViewState s = base;
        if (!j.is_object()) return s;
        auto str = [&](const char* k) { return j.contains(k) && j[k].is_string() ? j[k].get<std::string>() : std::string(); };
        if (auto m = viewModeFromString(str("mode"))) s.mode = *m;
        if (auto t = viewerToolFromString(str("tool"))) s.tool = *t;
        if (auto t = paintToolFromString(str("paint_tool"))) s.paintTool = *t;
        auto num = [&](const char* k, auto& out) {
            if (j.contains(k) && j[k].is_number()) out = static_cast<std::decay_t<decltype(out)>>(j[k].get<double>());
        };
        auto boolean = [&](const char* k, bool& out) {
            if (j.contains(k) && j[k].is_boolean()) out = j[k].get<bool>();
        };
        num("brush_px", s.brushPx);
        boolean("paint_3d", s.paint3d);
        num("z", s.z);
        num("t", s.t);
        num("crosshair_x", s.cx);
        num("crosshair_y", s.cy);
        boolean("crosshair", s.crosshair);
        boolean("labels", s.labels);
        boolean("bounding_box", s.boundingBox);
        boolean("scale_bar", s.scaleBar);
        boolean("sync_zt", s.syncZT);
        if (j.contains("channels") && j["channels"].is_array()) {
            s.channelVisible.clear();
            for (const json& e : j["channels"]) s.channelVisible.push_back(e.is_boolean() ? e.get<bool>() : true);
        }
        num("zoom", s.zoom);
        if (j.contains("pan") && j["pan"].is_array() && j["pan"].size() == 2) {
            s.panX = j["pan"][0].get<double>();
            s.panY = j["pan"][1].get<double>();
        }
        num("yaw", s.yaw);
        num("pitch", s.pitch);
        if (j.contains("clip_z") && j["clip_z"].is_array() && j["clip_z"].size() == 2)
            s.clipZ = {j["clip_z"][0].get<double>(), j["clip_z"][1].get<double>()};
        num("label_opacity", s.labelOpacity);
        num("selected_label", s.selectedLabel);
        return s;
    }

    // --- RunJob ------------------------------------------------------------------

    void RunJob::execute() {
        const auto t0 = std::chrono::steady_clock::now();
        finished_ = false;
        error_.clear();
        reports_.clear();
        // Steps that will actually run, for an overall progress fraction.
        int toRun = 0;
        for (int i = 0; i <= target_; ++i) {
            const Step& s = pipeline_.at(i);
            if ((i == 0 || s.enabled) && !executor_->isFresh(pipeline_, i)) ++toRun;
        }
        int done = 0;
        int currentStep = -1;
        StepContext ctx = ctx_;
        ctx.progress = [&](double f, const std::string& m) {
            const double overall = toRun > 0 ? (done + std::clamp(f, 0.0, 1.0)) / toRun : 1.0;
            progress_.set(overall, currentStep, m);
        };
        ctx.cancelled = [this] { return cancelled_.load(); };
        auto onStep = [&](const StepReport& r) {
            if (r.note == "running" && !r.ran && r.error.empty()) {
                currentStep = r.index;
                progress_.set(toRun > 0 ? static_cast<double>(done) / toRun : 0.0, r.index, "");
            } else if (r.ran) {
                ++done;
                progress_.set(toRun > 0 ? static_cast<double>(done) / toRun : 1.0, r.index, "");
            }
        };
        try {
            output_ = executor_->run(pipeline_, target_, ctx, &reports_, onStep);
        } catch (const std::exception& e) {
            error_ = e.what();
        } catch (...) {
            error_ = "unknown error";
        }
        if (cancelled_.load()) error_ = "cancelled";
        seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        progress_.set(1.0, -1, "");
        finished_ = true;
    }

    // --- Workbench ---------------------------------------------------------------

    Workbench::Workbench(std::filesystem::path scratchDir) : executor_(std::move(scratchDir)) {
        registerBuiltinOperations();
        pipeline_ = Pipeline();
        // Default pipeline of the design: Load + Contrast, Contrast selected and viewed.
        if (findOperation("contrast")) pipeline_.add("contrast");
        selected_ = viewed_ = std::min(1, pipeline_.size() - 1);
        if (!cudaAvailable()) backend_ = Backend::Cpu;
    }

    Workbench::~Workbench() = default;

    void Workbench::addObserver(Observer* o) {
        if (o && std::find(observers_.begin(), observers_.end(), o) == observers_.end()) observers_.push_back(o);
    }

    void Workbench::removeObserver(Observer* o) {
        observers_.erase(std::remove(observers_.begin(), observers_.end(), o), observers_.end());
    }

    void Workbench::notify(void (Observer::*fn)()) {
        // copy: an observer may remove itself while being notified
        const std::vector<Observer*> obs = observers_;
        for (Observer* o : obs) (o->*fn)();
    }

    void Workbench::notifyStep(int index) {
        const std::vector<Observer*> obs = observers_;
        for (Observer* o : obs) o->stepChanged(index);
    }

    void Workbench::logLine(const std::string& line) {
        char stamp[16];
        const std::time_t now = std::time(nullptr);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &now);
#else
        localtime_r(&now, &tm);
#endif
        std::strftime(stamp, sizeof stamp, "%H:%M:%S ", &tm);
        log_.push_back(stamp + line);
        if (log_.size() > 5000) log_.erase(log_.begin(), log_.begin() + 1000);
        const std::vector<Observer*> obs = observers_;
        for (Observer* o : obs) o->logged(log_.back());
    }

    // --- snapshots / undo ----------------------------------------------------------

    Workbench::Snapshot Workbench::snapshot() const {
        Snapshot s;
        s.pipeline = pipeline_.toJson();
        s.selected = selected_;
        s.viewed = viewed_;
        s.view = view_;
        return s;
    }

    void Workbench::restore(const Snapshot& s) {
        pipeline_ = Pipeline::fromJson(s.pipeline);
        selected_ = s.selected;
        viewed_ = s.viewed;
        view_ = s.view;
        clampSelection();
        notify(&Observer::pipelineChanged);
        notify(&Observer::selectionChanged);
        notify(&Observer::viewedStepChanged);
        notify(&Observer::viewStateChanged);
        notify(&Observer::outputsChanged);
    }

    void Workbench::pushEdit(const std::string& label, const Snapshot& before, const std::string& mergeKey) {
        Snapshot first = before;
        if (!mergeKey.empty()) {
            if (lastMergeKey_ == mergeKey && mergeBefore_.count(mergeKey)) first = mergeBefore_[mergeKey];
            else mergeBefore_[mergeKey] = before;
        } else {
            mergeBefore_.clear();
        }
        lastMergeKey_ = mergeKey;
        const Snapshot after = snapshot();
        Command c;
        c.label = label;
        c.mergeKey = mergeKey;
        c.undo = [this, first] { restore(first); };
        c.redo = [this, after] { restore(after); };
        history_.push(std::move(c));
        notify(&Observer::historyChanged);
    }

    void Workbench::undo() {
        if (!history_.canUndo()) return;
        const std::string label = history_.undoLabel();
        history_.undo();
        lastMergeKey_.clear();
        mergeBefore_.clear();
        logLine("Undo: " + label);
        notify(&Observer::historyChanged);
    }

    void Workbench::redo() {
        if (!history_.canRedo()) return;
        const std::string label = history_.redoLabel();
        history_.redo();
        lastMergeKey_.clear();
        logLine("Redo: " + label);
        notify(&Observer::historyChanged);
    }

    void Workbench::clampSelection() {
        const int last = std::max(0, pipeline_.size() - 1);
        selected_ = std::clamp(selected_, 0, last);
        viewed_ = std::clamp(viewed_, 0, last);
    }

    // --- dataset -------------------------------------------------------------------

    namespace {
        void applyOpenOptions(ParamSet& params, const std::string& path, const OpenOptions& o, const DatasetMeta& meta) {
            params.set("path", path);
            params.set("read_as", std::string(o.readAll ? "Full load to RAM" : "Lazy (chunk on demand)"));
            if (o.pageOrder) {
                params.set("page_order", o.pageOrder->order);
                params.set("c", static_cast<std::int64_t>(o.pageOrder->c));
                params.set("t", static_cast<std::int64_t>(o.pageOrder->t));
                params.set("z", static_cast<std::int64_t>(o.pageOrder->z));
            }
            if (o.voxelUm) {
                params.set("voxel_x", (*o.voxelUm)[0]);
                params.set("voxel_y", (*o.voxelUm)[1]);
                params.set("voxel_z", (*o.voxelUm)[2]);
            }
            if (o.sim) {
                params.set("sim_ndirs", static_cast<std::int64_t>(o.sim->present ? o.sim->ndirs : 0));
                params.set("sim_nphases", static_cast<std::int64_t>(o.sim->present ? o.sim->nphases : 0));
            } else if (meta.sim.present) {
                params.set("sim_ndirs", static_cast<std::int64_t>(meta.sim.ndirs));
                params.set("sim_nphases", static_cast<std::int64_t>(meta.sim.nphases));
            }
            if (meta.lightSheet) params.set("sheet_angle", meta.sheetAngleDeg);
        }
    } // namespace

    OpenOptions Workbench::openOptionsFromLoadParams(const ParamSet& p) {
        OpenOptions o;
        o.readAll = p.getString("read_as").rfind("Full", 0) == 0;
        const std::string order = p.getString("page_order");
        const Index c = p.getInt("c", 0), t = p.getInt("t", 0), z = p.getInt("z", 0);
        if (c > 0 || t > 0 || z > 0 || (!order.empty() && order != "czt")) {
            PageOrder po;
            po.order = order.empty() ? "czt" : order;
            po.c = std::max<Index>(c, 1);
            po.t = std::max<Index>(t, 1);
            po.z = z;
            o.pageOrder = po;
        }
        const double vx = p.getDouble("voxel_x", 0.0), vy = p.getDouble("voxel_y", 0.0), vz = p.getDouble("voxel_z", 0.0);
        if (vx > 0.0 && vy > 0.0 && vz > 0.0) o.voxelUm = std::array<double, 3>{vx, vy, vz};
        const int ndirs = static_cast<int>(p.getInt("sim_ndirs", 0)), nphases = static_cast<int>(p.getInt("sim_nphases", 0));
        if (ndirs > 0 && nphases > 0) {
            SimLayout sim;
            sim.present = true;
            sim.ndirs = ndirs;
            sim.nphases = nphases;
            sim.fastSi = p.getBool("sim_fast", false);
            o.sim = sim;
        }
        return o;
    }

    void Workbench::openDataset(const std::string& path, const OpenOptions& options) {
        OpenResult opened = sirius::app::openDataset(path, options);   // throws with a message
        const Snapshot before = snapshot();
        source_ = opened.source;
        datasetMeta_ = opened.meta;
        Step& load = pipeline_.at(0);
        applyOpenOptions(load.params, path, options, datasetMeta_);
        if (const Operation* op = findOperation("load")) {
            load.params.applyDefaults(op->info().params);
            load.params.coerce(op->info().params);
        }
        executor_.clear();
        auto out = std::make_shared<StepOutput>();
        out->meta = datasetMeta_;
        out->source = source_;
        if (source_->inMemory()) out->array = source_->readAll();
        out->note = "opened " + datasetMeta_.format;
        loadOutput_ = out;
        executor_.seed(pipeline_, 0, out);
        view_.channelVisible.assign(static_cast<std::size_t>(std::max<Index>(datasetMeta_.dims.c, 1)), true);
        view_.z = std::min<Index>(view_.z, std::max<Index>(datasetMeta_.dims.z - 1, 0));
        view_.t = std::min<Index>(view_.t, std::max<Index>(datasetMeta_.dims.t - 1, 0));
        view_.cx = datasetMeta_.dims.x / 2;
        view_.cy = datasetMeta_.dims.y / 2;
        view_.z = datasetMeta_.dims.z / 2;
        pushEdit("Open " + datasetMeta_.name, before);
        logLine("Opened " + path + " · " + datasetMeta_.shapeString() + " · " + opened.metadataSummary);
        notify(&Observer::datasetChanged);
        notify(&Observer::pipelineChanged);
        notifyStep(0);
        notify(&Observer::viewStateChanged);
        notify(&Observer::outputsChanged);
    }

    void Workbench::setDataset(std::shared_ptr<ArraySource> source) {
        if (!source) {
            closeDataset();
            return;
        }
        const Snapshot before = snapshot();
        source_ = std::move(source);
        datasetMeta_ = source_->meta();
        Step& load = pipeline_.at(0);
        load.params.set("path", datasetMeta_.sourcePath);
        executor_.clear();
        auto out = std::make_shared<StepOutput>();
        out->meta = datasetMeta_;
        out->source = source_;
        if (source_->inMemory()) out->array = source_->readAll();
        loadOutput_ = out;
        executor_.seed(pipeline_, 0, out);
        view_.channelVisible.assign(static_cast<std::size_t>(std::max<Index>(datasetMeta_.dims.c, 1)), true);
        view_.cx = datasetMeta_.dims.x / 2;
        view_.cy = datasetMeta_.dims.y / 2;
        view_.z = datasetMeta_.dims.z / 2;
        view_.t = 0;
        pushEdit("Set dataset", before);
        notify(&Observer::datasetChanged);
        notify(&Observer::pipelineChanged);
        notify(&Observer::viewStateChanged);
        notify(&Observer::outputsChanged);
    }

    void Workbench::closeDataset() {
        if (!source_) return;
        source_.reset();
        loadOutput_.reset();
        datasetMeta_ = DatasetMeta{};
        pipeline_.at(0).params.set("path", std::string());
        executor_.clear();
        history_.clear();
        logLine("Closed dataset");
        notify(&Observer::datasetChanged);
        notify(&Observer::pipelineChanged);
        notify(&Observer::outputsChanged);
        notify(&Observer::historyChanged);
    }

    // --- pipeline edits -------------------------------------------------------------

    StepId Workbench::addStep(const std::string& kind, int at) {
        const Snapshot before = snapshot();
        const StepId id = pipeline_.add(kind, at);
        const int index = pipeline_.indexOf(id);
        // let the operation seed its parameters from the data it will see
        if (auto upstream = upstreamOutput(index)) {
            try {
                Step& s = pipeline_.at(index);
                s.params = s.op().initialParams(s.params, upstream->asInput());
            } catch (const std::exception& e) {
                logLine(std::string("Initial parameters: ") + e.what());
            }
        }
        selected_ = viewed_ = index;
        onStepSelected(index);
        pushEdit("Add " + pipeline_.at(index).name, before);
        logLine("Added step " + Step::number(index) + " " + pipeline_.at(index).name);
        notify(&Observer::pipelineChanged);
        notify(&Observer::selectionChanged);
        notify(&Observer::viewedStepChanged);
        notify(&Observer::viewStateChanged);
        return id;
    }

    void Workbench::removeStep(int index) {
        if (index < 1 || index >= pipeline_.size()) return;
        const Snapshot before = snapshot();
        const std::string name = pipeline_.at(index).name;
        const StepId id = pipeline_.at(index).id;
        pipeline_.remove(index);
        executor_.invalidate(id);
        if (selected_ >= index) selected_ = std::max(0, selected_ - 1);
        if (viewed_ >= index) viewed_ = std::max(0, viewed_ - 1);
        clampSelection();
        pushEdit("Remove " + name, before);
        logLine("Removed step " + name);
        notify(&Observer::pipelineChanged);
        notify(&Observer::selectionChanged);
        notify(&Observer::viewedStepChanged);
        notify(&Observer::outputsChanged);
    }

    bool Workbench::moveStep(int index, int delta) {
        const Snapshot before = snapshot();
        if (!pipeline_.move(index, delta)) return false;
        const int j = index + delta;
        auto remap = [&](int k) { return k == index ? j : (k == j ? index : k); };
        selected_ = remap(selected_);
        viewed_ = remap(viewed_);
        pushEdit(std::string("Move ") + pipeline_.at(j).name + (delta < 0 ? " up" : " down"), before);
        notify(&Observer::pipelineChanged);
        notify(&Observer::selectionChanged);
        notify(&Observer::viewedStepChanged);
        notify(&Observer::outputsChanged);
        return true;
    }

    StepId Workbench::duplicateStep(int index) {
        if (index < 1 || index >= pipeline_.size()) return 0;
        const Snapshot before = snapshot();
        const StepId id = pipeline_.duplicate(index);
        selected_ = pipeline_.indexOf(id);
        pushEdit("Duplicate " + pipeline_.at(index).name, before);
        notify(&Observer::pipelineChanged);
        notify(&Observer::selectionChanged);
        return id;
    }

    void Workbench::setStepEnabled(int index, bool on) {
        if (index < 1 || index >= pipeline_.size() || pipeline_.at(index).enabled == on) return;
        const Snapshot before = snapshot();
        pipeline_.setEnabled(index, on);
        pushEdit(std::string(on ? "Enable " : "Skip ") + pipeline_.at(index).name, before);
        notifyStep(index);
        notify(&Observer::pipelineChanged);
        notify(&Observer::outputsChanged);
    }

    void Workbench::setStepParams(int index, const ParamSet& params, const std::string& label,
                                  const std::string& mergeKey) {
        if (index < 0 || index >= pipeline_.size()) return;
        const Snapshot before = snapshot();
        pipeline_.setParams(index, params);
        if (pipeline_.at(index).params == ParamSet::fromJson(before.pipeline["steps"][static_cast<std::size_t>(index)]["params"]))
            return;   // nothing changed after coercion
        pushEdit(label.empty() ? "Edit " + pipeline_.at(index).name : label, before, mergeKey);
        notifyStep(index);
        notify(&Observer::outputsChanged);
    }

    void Workbench::setStepParam(int index, const std::string& key, const ParamValue& value, const std::string& mergeKey) {
        if (index < 0 || index >= pipeline_.size()) return;
        const Step& s = pipeline_.at(index);
        ParamSet p = s.params;
        const ParamValue* old = p.find(key);
        const std::string oldText = old ? toDisplayString(*old) : std::string("—");
        p.set(key, value);
        std::string label = "Step " + Step::number(index) + " · " + key + " " + oldText + " → " + toDisplayString(value);
        for (const ParamSpec& spec : s.op().info().params)
            if (spec.key == key) label = "Step " + Step::number(index) + " · " + spec.label + " " + oldText + " → " + toDisplayString(value);
        setStepParams(index, p, label, mergeKey.empty() ? std::string() : mergeKey + "#" + std::to_string(s.id) + "#" + key);
    }

    void Workbench::setStepCache(int index, CachePolicy policy) {
        if (index < 0 || index >= pipeline_.size() || pipeline_.at(index).cache == policy) return;
        const Snapshot before = snapshot();
        pipeline_.setCache(index, policy);
        pushEdit(std::string("Cache ") + pipeline_.at(index).name + " · " + toString(policy), before);
        notifyStep(index);
    }

    void Workbench::renameStep(int index, const std::string& name) {
        if (index < 0 || index >= pipeline_.size()) return;
        const Snapshot before = snapshot();
        pipeline_.rename(index, name);
        pushEdit("Rename step " + Step::number(index), before);
        notifyStep(index);
        notify(&Observer::pipelineChanged);
    }

    void Workbench::replacePipeline(const Pipeline& p, const std::string& label) {
        const Snapshot before = snapshot();
        std::vector<Step> steps = p.steps();
        // Keep our Load step (its params describe the open dataset) unless the
        // incoming pipeline names a dataset path of its own.
        const bool keepLoad = !steps.empty() && steps.front().params.getString("path").empty();
        pipeline_.replaceSteps(steps, keepLoad);
        if (source_) executor_.seed(pipeline_, 0, loadOutput_);
        selected_ = std::min(1, pipeline_.size() - 1);
        viewed_ = pipeline_.size() - 1;
        clampSelection();
        onStepSelected(selected_);
        pushEdit(label, before);
        notify(&Observer::pipelineChanged);
        notify(&Observer::selectionChanged);
        notify(&Observer::viewedStepChanged);
        notify(&Observer::viewStateChanged);
        notify(&Observer::outputsChanged);
    }

    void Workbench::loadPipeline(const std::string& path) {
        Pipeline p = Pipeline::load(path);
        // Relative Path parameters (OTF, parameter file, model, flat field)
        // are relative to the pipeline file, so pipelines travel with their data.
        const std::filesystem::path base = std::filesystem::absolute(std::filesystem::path(path)).parent_path();
        for (int i = 0; i < p.size(); ++i) {
            Step& s = p.at(i);
            const Operation* op = findOperation(s.kind);
            if (!op) continue;
            for (const ParamSpec& spec : op->info().params) {
                if (spec.type != ParamType::Path) continue;
                const std::string v = s.params.getString(spec.key);
                if (v.empty() || std::filesystem::path(v).is_absolute()) continue;
                std::error_code ec;
                const std::filesystem::path beside = base / v;
                if (std::filesystem::exists(beside, ec)) s.params.set(spec.key, beside.lexically_normal().string());
            }
        }
        replacePipeline(p, "Load pipeline " + std::filesystem::path(path).filename().string());
        pipelinePath_ = path;
        logLine("Loaded pipeline " + path);
        // A pipeline that names its dataset opens it (relative paths resolve
        // against the pipeline file, then the working directory).
        const std::string dataset = pipeline_.at(0).params.getString("path");
        if (dataset.empty() || (source_ && datasetMeta_.sourcePath == dataset)) return;
        std::filesystem::path resolved = dataset;
        if (resolved.is_relative()) {
            const std::filesystem::path beside = std::filesystem::path(path).parent_path() / resolved;
            std::error_code ec;
            if (std::filesystem::exists(beside, ec)) resolved = beside;
        }
        try {
            openDataset(resolved.string(), openOptionsFromLoadParams(pipeline_.at(0).params));
        } catch (const std::exception& e) {
            logLine("The pipeline's dataset could not be opened: " + std::string(e.what()));
        }
    }

    void Workbench::savePipeline(const std::string& path) const {
        pipeline_.save(path);
        const_cast<Workbench*>(this)->pipelinePath_ = path;
        const_cast<Workbench*>(this)->logLine("Saved pipeline " + path);
    }

    void Workbench::loadExamplePipeline() {
        replacePipeline(Pipeline::example(), "Load example pipeline");
        logLine("Loaded the example pipeline");
    }

    void Workbench::copyParameters(int index) {
        if (index < 0 || index >= pipeline_.size()) return;
        clipboard_ = std::make_pair(pipeline_.at(index).kind, pipeline_.at(index).params);
    }

    bool Workbench::pasteParameters(int index) {
        if (!clipboard_ || index < 0 || index >= pipeline_.size()) return false;
        if (pipeline_.at(index).kind != clipboard_->first) return false;
        setStepParams(index, clipboard_->second, "Paste parameters into " + pipeline_.at(index).name);
        return true;
    }

    // --- descriptions -------------------------------------------------------------

    DatasetMeta Workbench::inputMetaOf(int index) const {
        DatasetMeta meta = datasetMeta_;
        for (int i = 0; i < std::min(index, pipeline_.size()); ++i) {
            const Step& s = pipeline_.at(i);
            if (i > 0 && !s.enabled) continue;
            if (const Operation* op = findOperation(s.kind)) {
                try {
                    meta = op->outputMeta(s.params, meta);
                } catch (const std::exception&) {
                }
            }
        }
        return meta;
    }

    DatasetMeta Workbench::outputMetaOf(int index) const {
        if (index < 0 || index >= pipeline_.size()) return datasetMeta_;
        const DatasetMeta in = inputMetaOf(index);
        const Step& s = pipeline_.at(index);
        if (index > 0 && !s.enabled) return in;
        if (const Operation* op = findOperation(s.kind)) {
            try {
                return op->outputMeta(s.params, in);
            } catch (const std::exception&) {
            }
        }
        return in;
    }

    std::string Workbench::stepSummary(int index) const {
        if (index < 0 || index >= pipeline_.size()) return {};
        const Step& s = pipeline_.at(index);
        if (const Operation* op = findOperation(s.kind)) {
            try {
                return op->summary(s.params, inputMetaOf(index));
            } catch (const std::exception& e) {
                return e.what();
            }
        }
        return "unknown operation";
    }

    Validation Workbench::stepValidation(int index) const {
        Validation v;
        if (index < 0 || index >= pipeline_.size()) {
            v.errors.push_back("no such step");
            return v;
        }
        const Step& s = pipeline_.at(index);
        if (index > 0 && !source_) {
            v.errors.push_back("No dataset loaded.");
            return v;
        }
        if (const Operation* op = findOperation(s.kind)) {
            try {
                return op->validate(s.params, inputMetaOf(index));
            } catch (const std::exception& e) {
                v.errors.push_back(e.what());
            }
        } else {
            v.errors.push_back("unknown operation " + s.kind);
        }
        return v;
    }

    std::size_t Workbench::estimatedBytesOf(int index) const {
        if (index < 0 || index >= pipeline_.size()) return 0;
        const Step& s = pipeline_.at(index);
        if (const Operation* op = findOperation(s.kind)) {
            try {
                return op->estimatedOutputBytes(s.params, inputMetaOf(index));
            } catch (const std::exception&) {
            }
        }
        return 0;
    }

    // --- selection & view -------------------------------------------------------------

    void Workbench::onStepSelected(int index) {
        if (index < 0 || index >= pipeline_.size()) return;
        const std::string& kind = pipeline_.at(index).kind;
        const OpInfo& info = pipeline_.at(index).op().info();
        if (kind == "volrec") view_.mode = ViewMode::Volume;
        if (info.producesLabels || info.needsLabels) {
            view_.labels = true;
            view_.tool = ViewerTool::Paint;
        } else if (view_.tool == ViewerTool::Paint) {
            view_.tool = ViewerTool::Probe;
        }
    }

    void Workbench::select(int index) {
        if (index < 0 || index >= pipeline_.size() || index == selected_) return;
        selected_ = index;
        const ViewState before = view_;
        onStepSelected(index);
        notify(&Observer::selectionChanged);
        if (!(before.mode == view_.mode && before.tool == view_.tool && before.labels == view_.labels))
            notify(&Observer::viewStateChanged);
    }

    void Workbench::view(int index) {
        if (index < 0 || index >= pipeline_.size() || index == viewed_) return;
        viewed_ = index;
        const DatasetMeta meta = outputMetaOf(index);
        view_.channelVisible.resize(static_cast<std::size_t>(std::max<Index>(meta.dims.c, 1)), true);
        notify(&Observer::viewedStepChanged);
    }

    void Workbench::setViewState(const ViewState& s) {
        view_ = s;
        notify(&Observer::viewStateChanged);
    }

    void Workbench::setViewMode(ViewMode m) {
        if (view_.mode == m) return;
        view_.mode = m;
        notify(&Observer::viewStateChanged);
    }

    void Workbench::setTool(ViewerTool t) {
        if (view_.tool == t) return;
        view_.tool = t;
        if (t == ViewerTool::Paint) view_.labels = true;
        notify(&Observer::viewStateChanged);
    }

    void Workbench::setPaintTool(PaintTool t) {
        view_.paintTool = t;
        view_.tool = ViewerTool::Paint;
        view_.labels = true;
        notify(&Observer::viewStateChanged);
    }

    void Workbench::setZ(Index z) {
        const DatasetMeta meta = outputMetaOf(viewed_);
        z = std::clamp<Index>(z, 0, std::max<Index>(meta.dims.z - 1, 0));
        if (view_.z == z) return;
        view_.z = z;
        notify(&Observer::viewStateChanged);
    }

    void Workbench::setT(Index t) {
        const DatasetMeta meta = outputMetaOf(viewed_);
        t = std::clamp<Index>(t, 0, std::max<Index>(meta.dims.t - 1, 0));
        if (view_.t == t) return;
        view_.t = t;
        notify(&Observer::viewStateChanged);
    }

    void Workbench::setCrosshair(Index x, Index y, Index z) {
        const DatasetMeta meta = outputMetaOf(viewed_);
        view_.cx = std::clamp<Index>(x, 0, std::max<Index>(meta.dims.x - 1, 0));
        view_.cy = std::clamp<Index>(y, 0, std::max<Index>(meta.dims.y - 1, 0));
        view_.z = std::clamp<Index>(z, 0, std::max<Index>(meta.dims.z - 1, 0));
        notify(&Observer::viewStateChanged);
    }

    void Workbench::setChannelVisible(Index c, bool on) {
        if (c < 0) return;
        if (static_cast<std::size_t>(c) >= view_.channelVisible.size()) view_.channelVisible.resize(static_cast<std::size_t>(c) + 1, true);
        view_.channelVisible[static_cast<std::size_t>(c)] = on;
        notify(&Observer::viewStateChanged);
    }

    void Workbench::toggleCrosshair() {
        view_.crosshair = !view_.crosshair;
        notify(&Observer::viewStateChanged);
    }

    void Workbench::toggleLabels() {
        view_.labels = !view_.labels;
        notify(&Observer::viewStateChanged);
    }

    // --- outputs ---------------------------------------------------------------------

    std::shared_ptr<const StepOutput> Workbench::output(int index) const {
        if (index < 0 || index >= pipeline_.size()) return nullptr;
        if (index == 0) return loadOutput_;
        return executor_.lastOutput(pipeline_.at(index).id);
    }

    bool Workbench::outputFresh(int index) const {
        if (index < 0 || index >= pipeline_.size()) return false;
        return executor_.isFresh(pipeline_, index);
    }

    std::shared_ptr<const StepOutput> Workbench::displayOutput(int* actualIndex) const {
        // A live-preview step that has not run (or is stale) is shown on its
        // input: the viewer applies the step's parameters itself.
        if (viewedIsLivePreview()) return upstreamOutput(viewed_, actualIndex);
        for (int i = viewed_; i >= 0; --i) {
            if (i > 0 && !pipeline_.at(i).enabled) continue;
            auto out = output(i);
            if (out && (out->array || out->source)) {
                if (actualIndex) *actualIndex = i;
                return out;
            }
        }
        if (actualIndex) *actualIndex = -1;
        return nullptr;
    }

    std::shared_ptr<const StepOutput> Workbench::upstreamOutput(int index, int* actualIndex) const {
        for (int i = std::min(index, pipeline_.size()) - 1; i >= 0; --i) {
            if (i > 0 && !pipeline_.at(i).enabled) continue;
            auto out = output(i);
            if (out && (out->array || out->source)) {
                if (actualIndex) *actualIndex = i;
                return out;
            }
        }
        if (actualIndex) *actualIndex = -1;
        return nullptr;
    }

    bool Workbench::viewedIsLivePreview() const {
        if (!source_ || viewed_ <= 0 || viewed_ >= pipeline_.size()) return false;
        const Step& s = pipeline_.at(viewed_);
        if (!s.enabled) return false;
        const Operation* op = findOperation(s.kind);
        return op && op->info().livePreview && !outputFresh(viewed_);
    }

    Diagnostics Workbench::previewDiagnostics(int index) const {
        Diagnostics d;
        if (index < 0 || index >= pipeline_.size()) return d;
        const Step& s = pipeline_.at(index);
        const Operation* op = findOperation(s.kind);
        if (!op) return d;
        d.kind = op->info().diagnostics;
        d.summary = stepSummary(index);
        const DatasetMeta in = inputMetaOf(index), out = outputMetaOf(index);
        d.facts.push_back({"Input", index > 0 ? in.shapeString() : std::string("—")});
        d.facts.push_back({"Output", out.shapeString()});
        const std::size_t bytes = estimatedBytesOf(index);
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.1f GB", static_cast<double>(bytes) / 1e9);
        d.facts.push_back({"Est. output", bytes ? buf : "—"});
        const Validation v = stepValidation(index);
        for (const std::string& w : v.warnings) d.warnings.push_back(w);
        for (const std::string& e : v.errors) d.warnings.push_back(e);
        // The operation's own live preview, when it has one and an input exists.
        if (source_ && index > 0) {
            std::shared_ptr<const StepOutput> upstream = upstreamOutput(index);
            if (upstream) {
                try {
                    if (auto p = op->preview(upstream->asInput(), s.params)) {
                        p->warnings.insert(p->warnings.end(), d.warnings.begin(), d.warnings.end());
                        if (p->summary.empty()) p->summary = d.summary;
                        return *p;
                    }
                } catch (const std::exception& e) {
                    d.warnings.push_back(e.what());
                }
            }
        }
        return d;
    }

    Diagnostics Workbench::selectedDiagnostics() const {
        if (auto out = output(selected_); out && !out->diagnostics.empty() && selected_ > 0) {
            Diagnostics d = out->diagnostics;
            if (!outputFresh(selected_)) d.warnings.insert(d.warnings.begin(), "Parameters changed since this result: run the step again.");
            return d;
        }
        return previewDiagnostics(selected_);
    }

    void Workbench::clearCache(int index) {
        if (index < 1 || index >= pipeline_.size()) return;
        executor_.invalidate(pipeline_.at(index).id);
        logLine("Cleared cache of " + pipeline_.at(index).name);
        notify(&Observer::outputsChanged);
    }

    void Workbench::clearAllCaches() {
        executor_.clear();
        if (source_) executor_.seed(pipeline_, 0, loadOutput_);
        logLine("Cleared all caches");
        notify(&Observer::outputsChanged);
    }

    std::size_t Workbench::cachedBytes() const { return executor_.cachedBytes(); }

    // --- running ------------------------------------------------------------------

    void Workbench::setBackend(Backend b) {
        if (backend_ == b) return;
        backend_ = b;
        logLine(std::string("Backend: ") + toString(b));
        notify(&Observer::backendChanged);
    }

    void Workbench::setCudaDevice(int index) {
        cudaDevice_ = std::max(0, index);
        notify(&Observer::backendChanged);
    }

    void Workbench::setRemoteConfig(RemoteConfig c) {
        remote_ = std::move(c);
        notify(&Observer::backendChanged);
    }

    int Workbench::loadPlugins(bool reload) {
        if (!launcher_) {
            logLine("Plugins: no Python worker launcher configured.");
            return 0;
        }
        try {
            std::unique_ptr<RemoteWorker> worker = launcher_();
            if (!worker) throw std::runtime_error("the Python worker did not start");
            const PluginLoadResult r = registerPluginOperations(*worker, reload);
            plugins_.clear();
            for (const PluginLoadResult::Entry& e : r.entries) plugins_.push_back({e.kind, e.name, e.file, e.error});
            pluginDirs_ = r.dirs;
            for (const std::string& e : r.errors) logLine("Plugin error: " + e);
            std::string kinds;
            for (const std::string& k : r.kinds) kinds += (kinds.empty() ? "" : ", ") + k;
            logLine(r.kinds.empty() ? "Plugins: none found" + (r.dirs.empty() ? std::string() : " in " + r.dirs.back())
                                    : "Plugins: " + kinds);
            notify(&Observer::operationsChanged);
            return static_cast<int>(r.kinds.size());
        } catch (const std::exception& e) {
            logLine(std::string("Plugins unavailable: ") + e.what());
            return 0;
        }
    }

    std::shared_ptr<RunJob> Workbench::createRun(int target) {
        if (activeRun_) {
            logLine("A run is already in progress.");
            return nullptr;
        }
        if (!source_) {
            logLine("Open a dataset before running.");
            return nullptr;
        }
        if (target < 0 || target >= pipeline_.size()) target = pipeline_.size() - 1;
        bool needsWorker = false;
        for (int i = 1; i <= target; ++i) {
            const Step& s = pipeline_.at(i);
            if (!s.enabled) continue;
            const Validation v = stepValidation(i);
            if (!v.ok()) {
                logLine("Step " + Step::number(i) + " " + s.name + " cannot run: " + v.firstError());
                return nullptr;
            }
            if (s.op().info().remoteCapable && !executor_.isFresh(pipeline_, i)) needsWorker = true;
        }
        auto job = std::make_shared<RunJob>();
        job->pipeline_ = pipeline_;
        job->target_ = target;
        job->executor_ = &executor_;
        job->ctx_.backend = backend_;
        job->ctx_.device = (backend_ == Backend::Cuda && cudaAvailable()) ? Device::cuda(cudaDevice_) : Device::cpu();
        job->ctx_.scratchDir = executor_.scratchDir();
        try {
            if (backend_ == Backend::Hpc) {
                job->ownedRemote_ = RemoteWorker::connect(remote_.host, remote_.port, remote_.token);
                logLine("HPC worker: " + job->ownedRemote_->capabilities().device + " on " +
                        job->ownedRemote_->capabilities().hostname);
            } else if (needsWorker) {
                if (!launcher_) throw std::runtime_error("no Python worker launcher configured");
                job->ownedRemote_ = launcher_();
                if (!job->ownedRemote_) throw std::runtime_error("the Python worker did not start");
                logLine("Local worker: " + job->ownedRemote_->capabilities().device);
            }
        } catch (const std::exception& e) {
            logLine(std::string("Worker unavailable: ") + e.what() +
                    " (Preferences ▸ Worker sets the Python interpreter; Preferences ▸ HPC the remote host).");
            return nullptr;
        }
        job->ctx_.remote = job->ownedRemote_.get();
        activeRun_ = job;
        logLine("Run to step " + Step::number(target) + " on " + toString(backend_));
        notify(&Observer::runStateChanged);
        return job;
    }

    void Workbench::finishRun(const std::shared_ptr<RunJob>& job) {
        if (!job) return;
        if (activeRun_ == job) activeRun_.reset();
        for (const StepReport& r : job->reports()) {
            if (r.index < 0 || r.index >= pipeline_.size()) continue;
            const std::string name = Step::number(r.index) + " " + pipeline_.at(r.index).name;
            if (!r.error.empty()) logLine("Step " + name + " failed: " + r.error);
            else if (r.ran) {
                char buf[32];
                std::snprintf(buf, sizeof buf, "%.1f s", r.seconds);
                logLine("Step " + name + " · " + buf + (r.note.empty() ? "" : " · " + r.note));
            } else if (r.skipped) logLine("Step " + name + " skipped");
        }
        if (job->succeeded()) {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%.1f s", job->seconds());
            logLine(std::string("Run finished in ") + buf);
        } else {
            logLine("Run " + (job->error() == "cancelled" ? std::string("cancelled") : "failed: " + job->error()));
        }
        job->ownedRemote_.reset();
        const DatasetMeta meta = outputMetaOf(viewed_);
        view_.channelVisible.resize(static_cast<std::size_t>(std::max<Index>(meta.dims.c, 1)), true);
        view_.z = std::clamp<Index>(view_.z, 0, std::max<Index>(meta.dims.z - 1, 0));
        view_.t = std::clamp<Index>(view_.t, 0, std::max<Index>(meta.dims.t - 1, 0));
        notify(&Observer::outputsChanged);
        notify(&Observer::viewStateChanged);
        notify(&Observer::runStateChanged);
    }

    void Workbench::cancelRun() {
        if (activeRun_) {
            activeRun_->cancel();
            logLine("Cancelling…");
        }
    }

    // --- labels ------------------------------------------------------------------

    std::shared_ptr<LabelVolume> Workbench::viewedLabels() const {
        auto out = displayOutput();
        return out ? out->labels : nullptr;
    }

    void Workbench::recordLabelDiff(const std::string& label, StepId id, LabelDiff diff) {
        if (diff.empty()) return;
        auto labels = viewedLabels();
        if (!labels) return;
        std::shared_ptr<LabelDiff> d = std::make_shared<LabelDiff>(std::move(diff));
        Command c;
        c.label = label;
        c.undo = [this, labels, d, id] {
            labels->apply(*d, false);
            const std::vector<Observer*> obs = observers_;
            for (Observer* o : obs) o->labelsChanged(id);
        };
        c.redo = [this, labels, d, id] {
            labels->apply(*d, true);
            const std::vector<Observer*> obs = observers_;
            for (Observer* o : obs) o->labelsChanged(id);
        };
        history_.push(std::move(c));
        notify(&Observer::historyChanged);
        const std::vector<Observer*> obs = observers_;
        for (Observer* o : obs) o->labelsChanged(id);
    }

    void Workbench::beginPaintStroke() {
        ++strokeCounter_;
        strokeDiff_ = LabelDiff{};
        int actual = -1;
        auto out = displayOutput(&actual);
        strokeStep_ = (out && actual >= 0) ? pipeline_.at(actual).id : 0;
        if (view_.selectedLabel == 0 && out && out->labels && view_.paintTool == PaintTool::Brush)
            view_.selectedLabel = out->labels->maxLabel() + 1;
    }

    void Workbench::paintLabels(Index z, Index y, Index x, bool erase) {
        static const bool trace = std::getenv("SIRIUS_TRACE_VIEW") != nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        auto labels = viewedLabels();
        if (!labels) return;
        const auto t1 = std::chrono::steady_clock::now();
        struct Report {
            bool on;
            std::chrono::steady_clock::time_point t0, t1;
            ~Report() {
                if (!on) return;
                const auto t2 = std::chrono::steady_clock::now();
                std::fprintf(stderr, "paintLabels: lookup %lld us · edit+notify %lld us\n",
                             static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()),
                             static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()));
            }
        } report{trace, t0, t1};
        if (strokeCounter_ == 0) beginPaintStroke();
        const double radius = std::max(1.0, view_.brushPx / 2.0);
        const Index zRadius = view_.paint3d ? std::max<Index>(1, view_.brushPx / 6) : 0;
        const std::uint32_t label = erase ? 0u : view_.selectedLabel;
        LabelDiff diff = labels->paint(view_.t, z, y, x, radius, zRadius, label, erase ? view_.selectedLabel : 0u);
        if (diff.empty()) return;
        // One stroke = one undo entry: the accumulated diff replaces the entry
        // pushed by the previous mouse move (History merges by key).
        strokeDiff_.t = diff.t;
        strokeDiff_.indices.insert(strokeDiff_.indices.end(), diff.indices.begin(), diff.indices.end());
        strokeDiff_.before.insert(strokeDiff_.before.end(), diff.before.begin(), diff.before.end());
        strokeDiff_.after.insert(strokeDiff_.after.end(), diff.after.begin(), diff.after.end());
        auto d = std::make_shared<LabelDiff>(strokeDiff_);
        const StepId id = strokeStep_;
        Command c;
        c.label = erase ? "Erase labels" : "Paint label " + std::to_string(label);
        c.mergeKey = "stroke#" + std::to_string(strokeCounter_);
        c.undo = [this, labels, d, id] {
            labels->apply(*d, false);
            const std::vector<Observer*> obs = observers_;
            for (Observer* o : obs) o->labelsChanged(id);
        };
        c.redo = [this, labels, d, id] {
            labels->apply(*d, true);
            const std::vector<Observer*> obs = observers_;
            for (Observer* o : obs) o->labelsChanged(id);
        };
        history_.push(std::move(c));
        notify(&Observer::historyChanged);
        const std::vector<Observer*> obs = observers_;
        for (Observer* o : obs) o->labelsChanged(id);
    }

    void Workbench::fillLabel(Index z, Index y, Index x) {
        auto labels = viewedLabels();
        if (!labels) return;
        int actual = -1;
        displayOutput(&actual);
        const std::uint32_t label = view_.selectedLabel ? view_.selectedLabel : labels->maxLabel() + 1;
        recordLabelDiff("Fill label " + std::to_string(label), actual >= 0 ? pipeline_.at(actual).id : 0,
                        labels->fill(view_.t, z, y, x, label));
    }

    void Workbench::mergeLabels(const std::vector<std::uint32_t>& ids) {
        auto labels = viewedLabels();
        if (!labels || ids.size() < 2) return;
        int actual = -1;
        displayOutput(&actual);
        LabelDiff diff = labels->merge(view_.t, ids);
        labels->recomputeStats(view_.t);
        recordLabelDiff("Merge labels", actual >= 0 ? pipeline_.at(actual).id : 0, std::move(diff));
    }

    void Workbench::splitLabel(std::uint32_t id, std::array<Index, 3> a, std::array<Index, 3> b) {
        auto labels = viewedLabels();
        if (!labels || id == 0) return;
        int actual = -1;
        displayOutput(&actual);
        LabelDiff diff = labels->split(view_.t, id, a, b);
        labels->recomputeStats(view_.t);
        recordLabelDiff("Split label " + std::to_string(id), actual >= 0 ? pipeline_.at(actual).id : 0, std::move(diff));
    }

    void Workbench::deleteLabel(std::uint32_t id) {
        auto labels = viewedLabels();
        if (!labels || id == 0) return;
        int actual = -1;
        displayOutput(&actual);
        LabelDiff diff = labels->remove(view_.t, id);
        labels->recomputeStats(view_.t);
        if (view_.selectedLabel == id) view_.selectedLabel = 0;
        recordLabelDiff("Delete label " + std::to_string(id), actual >= 0 ? pipeline_.at(actual).id : 0, std::move(diff));
    }

    void Workbench::setLabelReviewed(std::uint32_t id, bool reviewed) {
        auto labels = viewedLabels();
        if (!labels) return;
        for (LabelStats& s : labels->stats())
            if (s.id == id) s.reviewed = reviewed;
        int actual = -1;
        displayOutput(&actual);
        const std::vector<Observer*> obs = observers_;
        for (Observer* o : obs) o->labelsChanged(actual >= 0 ? pipeline_.at(actual).id : 0);
    }

    void Workbench::acceptAllReviewed() {
        auto labels = viewedLabels();
        if (!labels) return;
        for (LabelStats& s : labels->stats()) s.reviewed = true;
        int actual = -1;
        displayOutput(&actual);
        logLine("Accepted all reviewed labels");
        const std::vector<Observer*> obs = observers_;
        for (Observer* o : obs) o->labelsChanged(actual >= 0 ? pipeline_.at(actual).id : 0);
    }

    std::uint32_t Workbench::nextFlaggedLabel(bool forward) {
        auto labels = viewedLabels();
        if (!labels) return 0;
        const auto& stats = labels->stats();
        if (stats.empty()) return 0;
        const int n = static_cast<int>(stats.size());
        int start = 0;
        for (int i = 0; i < n; ++i)
            if (stats[static_cast<std::size_t>(i)].id == view_.selectedLabel) start = i;
        for (int k = 1; k <= n; ++k) {
            const int i = ((start + (forward ? k : -k)) % n + n) % n;
            const LabelStats& s = stats[static_cast<std::size_t>(i)];
            if (s.flags.empty() || s.reviewed) continue;
            view_.selectedLabel = s.id;
            view_.cx = (s.bbox[4] + s.bbox[5]) / 2;
            view_.cy = (s.bbox[2] + s.bbox[3]) / 2;
            view_.z = (s.bbox[0] + s.bbox[1]) / 2;
            view_.labels = true;
            notify(&Observer::viewStateChanged);
            return s.id;
        }
        return 0;
    }

} // namespace sirius::app
