#ifndef SIRIUS_APP_EXECUTOR_HPP
#define SIRIUS_APP_EXECUTOR_HPP

// Runs a pipeline top to bottom and caches step outputs per the step's cache
// policy. A step's cache entry is keyed by a fingerprint of its own
// definition and of the fingerprint of the output it consumed, so editing a
// parameter invalidates exactly the steps downstream of it, and reordering
// steps recomputes only what actually changed. Disk-cached arrays are
// spilled to the scratch directory and reloaded on demand.
//
// Threading: run() executes on the calling (worker) thread; every query is
// safe to call from another thread while a run is in progress.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/operation.hpp"
#include "core/pipeline.hpp"

namespace sirius::app {

    struct StepReport {
        StepId id = 0;
        int index = -1;
        bool ran = false;                // false when served from the cache or skipped
        bool skipped = false;            // disabled: input passed through
        double seconds = 0.0;
        std::string note;
        std::string error;               // non-empty when the step failed
    };

    class Executor {
    public:
        explicit Executor(std::filesystem::path scratchDir);
        ~Executor();

        // Fingerprint of step `index` given the pipeline (definition + upstream).
        std::string fingerprint(const Pipeline& p, int index) const;
        // Cached output of step `index` if it is fresh for the pipeline as it is now.
        std::shared_ptr<const StepOutput> cached(const Pipeline& p, int index) const;
        bool isFresh(const Pipeline& p, int index) const { return cached(p, index) != nullptr; }
        // Last output computed for a step, fresh or not (what the viewer shows
        // while the user edits parameters).
        std::shared_ptr<const StepOutput> lastOutput(StepId id) const;

        // Output of the pipeline at step `index`, running every stale enabled
        // step from the top down to it (skipped steps pass their input on).
        // Throws on failure; the failing step's report is the last one.
        std::shared_ptr<const StepOutput> run(const Pipeline& p, int index, const StepContext& ctx,
                                              std::vector<StepReport>* reports = nullptr,
                                              const std::function<void(const StepReport&)>& onStep = {});
        // Run every step (to the last one).
        std::shared_ptr<const StepOutput> runAll(const Pipeline& p, const StepContext& ctx,
                                                 std::vector<StepReport>* reports = nullptr,
                                                 const std::function<void(const StepReport&)>& onStep = {});

        // Install an output for step `index` as if it had just run (the Load
        // step's lazy source when a dataset is opened).
        void seed(const Pipeline& p, int index, std::shared_ptr<const StepOutput> out);
        void invalidate(StepId id);
        void clear();
        std::size_t cachedBytes() const;
        std::size_t cachedBytesOf(StepId id) const;
        const std::filesystem::path& scratchDir() const noexcept { return scratch_; }

        // Spill / restore for CachePolicy::Disk (also used by tests).
        static void writeArrayFile(const std::filesystem::path& path, const Array5& a);
        static std::shared_ptr<Array5> readArrayFile(const std::filesystem::path& path);

    private:
        struct Entry;
        std::shared_ptr<const StepOutput> load(Entry& e) const;
        void store(const Step& step, const std::string& fp, std::shared_ptr<const StepOutput> out);
        void refreshPolicies(const Pipeline& p);   // caller holds mutex_

        std::filesystem::path scratch_;
        mutable std::mutex mutex_;
        std::unordered_map<StepId, std::unique_ptr<Entry>> entries_;
    };

    // Hash helper shared with the tool API (stable across runs).
    std::string stableHash(const std::string& s);

} // namespace sirius::app

#endif // SIRIUS_APP_EXECUTOR_HPP
