#ifndef SIRIUS_APP_WORKBENCH_HPP
#define SIRIUS_APP_WORKBENCH_HPP

// The session: one dataset, one pipeline, the executor's cached outputs,
// the undo history and the viewer state, behind a single Qt-free facade that
// the widgets, the assistant's tool API and the tests all drive the same
// way. Every edit goes through here so it is undoable and observed.
//
// Threading: the workbench is single-threaded (the GUI thread). Runs are
// prepared here as RunJob objects, executed by the caller on a worker
// thread (RunJob::execute is self-contained) and folded back in with
// finishRun() on the GUI thread; progress is read from the job's atomics.

#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "core/array_source.hpp"
#include "core/dataset.hpp"
#include "core/executor.hpp"
#include "core/history.hpp"
#include "core/labels.hpp"
#include "core/operation.hpp"
#include "core/pipeline.hpp"
#include "core/rpc.hpp"

namespace sirius::app {

    enum class ViewMode { Ortho, Volume, Compare };
    enum class ViewerTool { Navigate, Probe, Measure, Roi, Paint };
    enum class PaintTool { Brush, Erase, Fill, Pick, Merge, Split, Delete, Lasso };

    const char* toString(ViewMode m) noexcept;       // "ortho" "3d" "compare"
    const char* toString(ViewerTool t) noexcept;     // "nav" "probe" "measure" "roi" "paint"
    const char* toString(PaintTool t) noexcept;
    std::optional<ViewMode> viewModeFromString(const std::string& s) noexcept;
    std::optional<ViewerTool> viewerToolFromString(const std::string& s) noexcept;
    std::optional<PaintTool> paintToolFromString(const std::string& s) noexcept;

    struct ViewState {
        ViewMode mode = ViewMode::Ortho;
        ViewerTool tool = ViewerTool::Probe;
        PaintTool paintTool = PaintTool::Brush;
        int brushPx = 18;
        bool paint3d = true;
        Index z = 0, t = 0;
        Index cx = 0, cy = 0;                   // crosshair voxel (x, y)
        bool crosshair = true;
        bool labels = false;
        bool boundingBox = true;
        bool scaleBar = true;
        bool syncZT = true;
        std::vector<bool> channelVisible;       // resized to the viewed output's channels
        double zoom = 1.0;                      // 1 = fit
        double panX = 0.0, panY = 0.0;          // screen pixels
        double yaw = 35.0, pitch = 22.0;        // 3D
        std::array<double, 2> clipZ{0.0, 1.0};
        double labelOpacity = 0.45;
        std::uint32_t selectedLabel = 0;

        bool channelOn(Index c) const noexcept {
            return c < 0 || static_cast<std::size_t>(c) >= channelVisible.size() || channelVisible[static_cast<std::size_t>(c)];
        }
        nlohmann::json toJson() const;
        static ViewState fromJson(const nlohmann::json& j);
        static ViewState fromJson(const nlohmann::json& j, const ViewState& base);
    };

    struct RunProgress {
        std::atomic<double> fraction{0.0};
        std::atomic<int> stepIndex{-1};
        std::mutex mutex;
        std::string message;                    // guarded by mutex
        std::string messageCopy() { std::lock_guard<std::mutex> g(mutex); return message; }
        void set(double f, int step, const std::string& m) {
            fraction.store(f);
            stepIndex.store(step);
            std::lock_guard<std::mutex> g(mutex);
            message = m;
        }
    };

    class Workbench;

    // One run, prepared on the GUI thread, executed anywhere.
    class RunJob {
    public:
        // Pipeline snapshot, target step index and context are fixed at creation.
        int target() const noexcept { return target_; }
        const Pipeline& pipeline() const noexcept { return pipeline_; }
        RunProgress& progress() noexcept { return progress_; }
        void cancel() noexcept { cancelled_.store(true); }
        bool cancelled() const noexcept { return cancelled_.load(); }

        // Blocking; never throws (errors land in error()).
        void execute();
        bool succeeded() const noexcept { return finished_ && error_.empty(); }
        bool finished() const noexcept { return finished_; }
        const std::string& error() const noexcept { return error_; }
        const std::vector<StepReport>& reports() const noexcept { return reports_; }
        std::shared_ptr<const StepOutput> output() const noexcept { return output_; }
        double seconds() const noexcept { return seconds_; }

    private:
        friend class Workbench;
        Pipeline pipeline_;
        int target_ = 0;
        Executor* executor_ = nullptr;
        StepContext ctx_;
        std::unique_ptr<RemoteWorker> ownedRemote_;   // HPC connection for the job
        RunProgress progress_;
        std::atomic<bool> cancelled_{false};
        bool finished_ = false;
        std::string error_;
        std::vector<StepReport> reports_;
        std::shared_ptr<const StepOutput> output_;
        double seconds_ = 0.0;
    };

    struct RemoteConfig {
        std::string host = "localhost";
        int port = 7645;
        std::string token;
    };

    class Workbench {
    public:
        // What changed; the Qt layer forwards these as signals. Called on the
        // GUI thread, never during a run's worker execution.
        class Observer {
        public:
            virtual ~Observer() = default;
            virtual void datasetChanged() {}
            virtual void pipelineChanged() {}                 // steps added / removed / moved / edited
            virtual void stepChanged(int /*index*/) {}        // one step's params / name / cache / enabled
            virtual void selectionChanged() {}
            virtual void viewedStepChanged() {}
            virtual void viewStateChanged() {}
            virtual void outputsChanged() {}                  // cached outputs / freshness
            virtual void labelsChanged(StepId /*id*/) {}      // label voxels edited
            virtual void runStateChanged() {}                 // started / finished
            virtual void historyChanged() {}
            virtual void backendChanged() {}
            virtual void operationsChanged() {}               // plugins (re)loaded
            virtual void logged(const std::string& /*line*/) {}
        };

        explicit Workbench(std::filesystem::path scratchDir);
        ~Workbench();
        Workbench(const Workbench&) = delete;
        Workbench& operator=(const Workbench&) = delete;

        void addObserver(Observer* o);
        void removeObserver(Observer* o);

        // --- dataset ---------------------------------------------------------
        void openDataset(const std::string& path, const OpenOptions& options = {});
        void setDataset(std::shared_ptr<ArraySource> source);   // tests, scripted data
        void closeDataset();
        bool hasDataset() const noexcept { return static_cast<bool>(source_); }
        const DatasetMeta& dataset() const noexcept { return datasetMeta_; }
        std::shared_ptr<ArraySource> source() const noexcept { return source_; }

        // --- pipeline (every mutation is one undo entry) ---------------------
        const Pipeline& pipeline() const noexcept { return pipeline_; }
        StepId addStep(const std::string& kind, int at = -1);
        void removeStep(int index);
        bool moveStep(int index, int delta);
        StepId duplicateStep(int index);
        void setStepEnabled(int index, bool on);
        void setStepParams(int index, const ParamSet& params, const std::string& label = {},
                           const std::string& mergeKey = {});
        void setStepParam(int index, const std::string& key, const ParamValue& value, const std::string& mergeKey = {});
        void setStepCache(int index, CachePolicy policy);
        void renameStep(int index, const std::string& name);
        void replacePipeline(const Pipeline& p, const std::string& label);
        void loadPipeline(const std::string& path);   // also opens the dataset the Load step names
        // OpenOptions equivalent to a Load step's parameters (page order, voxel size, SIM layout).
        static OpenOptions openOptionsFromLoadParams(const ParamSet& loadParams);
        void savePipeline(const std::string& path) const;
        void loadExamplePipeline();
        std::string pipelinePath() const noexcept { return pipelinePath_; }
        // Copy / paste parameters between steps of the same kind.
        void copyParameters(int index);
        bool pasteParameters(int index);
        bool hasCopiedParameters() const noexcept { return static_cast<bool>(clipboard_); }

        // Per-step description for the UI without running anything.
        std::string stepSummary(int index) const;
        Validation stepValidation(int index) const;
        DatasetMeta inputMetaOf(int index) const;      // meta arriving at the step
        DatasetMeta outputMetaOf(int index) const;     // meta leaving it (predicted)
        std::size_t estimatedBytesOf(int index) const;

        // --- selection & viewing --------------------------------------------
        int selectedIndex() const noexcept { return selected_; }
        int viewedIndex() const noexcept { return viewed_; }
        void select(int index);
        void view(int index);
        const ViewState& viewState() const noexcept { return view_; }
        void setViewState(const ViewState& s);          // notifies when changed
        void setViewMode(ViewMode m);
        void setTool(ViewerTool t);
        void setPaintTool(PaintTool t);
        void setZ(Index z);
        void setT(Index t);
        void setCrosshair(Index x, Index y, Index z);
        void setChannelVisible(Index c, bool on);
        void toggleCrosshair();
        void toggleLabels();

        // --- outputs ---------------------------------------------------------
        // Last computed output of step `index` (fresh or stale), or null.
        std::shared_ptr<const StepOutput> output(int index) const;
        bool outputFresh(int index) const;
        // What the viewer should draw for the viewed step: its output if it
        // has one, else the nearest computed upstream output (Load's lazy
        // source at worst). `actualIndex` reports which one it is.
        std::shared_ptr<const StepOutput> displayOutput(int* actualIndex = nullptr) const;
        // Nearest computed output upstream of step `index` (the step's input).
        std::shared_ptr<const StepOutput> upstreamOutput(int index, int* actualIndex = nullptr) const;
        // True while the viewed step is shown as a live preview on its input
        // (OpInfo::livePreview and not run or stale).
        bool viewedIsLivePreview() const;
        // Diagnostics of the selected step: the last run's, or the
        // operation's live preview when it offers one.
        Diagnostics selectedDiagnostics() const;
        void clearCache(int index);
        void clearAllCaches();
        std::size_t cachedBytes() const;

        // --- running ---------------------------------------------------------
        Backend backend() const noexcept { return backend_; }
        void setBackend(Backend b);
        int cudaDevice() const noexcept { return cudaDevice_; }
        void setCudaDevice(int index);
        const RemoteConfig& remoteConfig() const noexcept { return remote_; }
        void setRemoteConfig(RemoteConfig c);
        // Steps that need the Python worker (Operation::remoteCapable) get a
        // local worker from this launcher when the backend is not HPC; the
        // Qt layer installs one that spawns app/python/sirius_worker.
        using WorkerLauncher = std::function<std::unique_ptr<RemoteWorker>()>;
        void setLocalWorkerLauncher(WorkerLauncher launcher) { launcher_ = std::move(launcher); }
        // Starts the local worker (through the launcher), registers the user
        // operations it finds (app/python/sirius_worker/plugins.py) and logs the
        // outcome; returns the number registered. `reload` re-imports the files.
        int loadPlugins(bool reload);
        // A job for step `target` (or the last step when -1); null with a log
        // line when nothing can run. The caller executes it and calls finishRun.
        std::shared_ptr<RunJob> createRun(int target = -1);
        void finishRun(const std::shared_ptr<RunJob>& job);
        bool running() const noexcept { return static_cast<bool>(activeRun_); }
        std::shared_ptr<RunJob> activeRun() const noexcept { return activeRun_; }
        void cancelRun();

        // --- labels (undoable, on the viewed output) -------------------------
        std::shared_ptr<LabelVolume> viewedLabels() const;
        // One brush stroke = one undo entry: call beginPaintStroke() on press,
        // paintLabels() on every move.
        void beginPaintStroke();
        void paintLabels(Index z, Index y, Index x, bool erase);          // uses brush size / label
        void fillLabel(Index z, Index y, Index x);
        void mergeLabels(const std::vector<std::uint32_t>& ids);
        void splitLabel(std::uint32_t id, std::array<Index, 3> a, std::array<Index, 3> b);
        void deleteLabel(std::uint32_t id);
        void setLabelReviewed(std::uint32_t id, bool reviewed);
        void acceptAllReviewed();
        std::uint32_t nextFlaggedLabel(bool forward);

        // --- history & log ---------------------------------------------------
        History& history() noexcept { return history_; }
        void undo();
        void redo();
        const std::vector<std::string>& log() const noexcept { return log_; }
        void logLine(const std::string& line);

        const std::filesystem::path& scratchDir() const noexcept { return executor_.scratchDir(); }
        Executor& executor() noexcept { return executor_; }

    private:
        struct Snapshot {
            nlohmann::json pipeline;
            int selected = 1, viewed = 1;
            ViewState view;
        };
        Snapshot snapshot() const;
        void restore(const Snapshot& s);
        void pushEdit(const std::string& label, const Snapshot& before, const std::string& mergeKey = {});
        void notify(void (Observer::*fn)());
        void notifyStep(int index);
        void clampSelection();
        void onStepSelected(int index);
        Diagnostics previewDiagnostics(int index) const;
        void recordLabelDiff(const std::string& label, StepId id, LabelDiff diff);

        std::vector<Observer*> observers_;
        std::shared_ptr<ArraySource> source_;
        DatasetMeta datasetMeta_;
        Pipeline pipeline_;
        std::string pipelinePath_;
        Executor executor_;
        History history_;
        ViewState view_;
        int selected_ = 1;
        int viewed_ = 1;
        Backend backend_ = Backend::Cuda;
        int cudaDevice_ = 0;
        RemoteConfig remote_;
        std::shared_ptr<RunJob> activeRun_;
        std::vector<std::string> log_;
        std::optional<std::pair<std::string, ParamSet>> clipboard_;
        std::shared_ptr<const StepOutput> loadOutput_;   // the Load step's lazy output
        WorkerLauncher launcher_;
        std::map<std::string, Snapshot> mergeBefore_;    // first "before" of an open merge group
        std::string lastMergeKey_;
        int strokeCounter_ = 0;
        LabelDiff strokeDiff_;
        StepId strokeStep_ = 0;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_WORKBENCH_HPP
