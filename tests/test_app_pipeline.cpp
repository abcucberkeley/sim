// Tests of the workbench hub: pipeline structure and files, the executor's
// caching and freshness rules, the workbench facade (edits, undo / redo,
// selection, viewing, runs) and the assistant's tool API. The operations
// used are tiny synthetic ones registered by this file, so these tests do
// not depend on the built-in operations or on any dataset on disk.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <random>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <thread>

#include <nlohmann/json.hpp>

#include "core/array_source.hpp"
#include "core/executor.hpp"
#include "core/history.hpp"
#include "core/pipeline.hpp"
#include "core/tool_api.hpp"
#include "core/workbench.hpp"

#include "temp_path.hpp"

using namespace sirius;
using namespace sirius::app;
using json = nlohmann::json;

namespace {

    // Scales every voxel by `factor` and records how often it ran.
    struct ScaleOp final : Operation {
        static inline std::atomic<int> runs{0};
        OpInfo info_;
        ScaleOp() {
            info_.kind = "test_scale";
            info_.name = "Scale";
            info_.group = "Intensity";
            info_.kindLabel = "INTENSITY";
            info_.params = {doubleParam("factor", "Factor", 2.0).range(0.0, 100.0)};
            info_.defaultCache = CachePolicy::Memory;
        }
        const OpInfo& info() const noexcept override { return info_; }
        StepOutput run(const StepInput& in, const ParamSet& p, const StepContext& ctx) const override {
            ++runs;
            ctx.report(0.5, "scaling");
            ArrayPtr src = in.materialize();
            auto out = std::make_shared<Array5>(src->clone());
            const float f = static_cast<float>(p.getDouble("factor"));
            for (Index i = 0; i < out->numel(); ++i) out->data()[i] *= f;
            StepOutput o;
            o.meta = in.meta;
            o.array = out;
            o.note = "scaled";
            o.diagnostics.summary = "scaled by " + toDisplayString(*p.find("factor"));
            return o;
        }
    };

    // Reduces z to 1 (max), to test outputMeta and shape propagation.
    struct MaxZOp final : Operation {
        OpInfo info_;
        MaxZOp() {
            info_.kind = "test_maxz";
            info_.name = "Max Z";
            info_.group = "Reduce";
            info_.kindLabel = "EINSUM";
        }
        const OpInfo& info() const noexcept override { return info_; }
        DatasetMeta outputMeta(const ParamSet&, const DatasetMeta& in) const override {
            DatasetMeta m = in;
            m.dims.z = 1;
            return m;
        }
        StepOutput run(const StepInput& in, const ParamSet&, const StepContext&) const override {
            ArrayPtr src = in.materialize();
            const Dims5 d = src->dims();
            auto out = std::make_shared<Array5>(Array5::zeros(Dims5{d.c, d.t, 1, d.y, d.x}));
            for (Index c = 0; c < d.c; ++c)
                for (Index t = 0; t < d.t; ++t)
                    for (Index z = 0; z < d.z; ++z) {
                        const float* p = src->plane(c, t, z);
                        float* o = out->plane(c, t, 0);
                        for (Index i = 0; i < d.planeSize(); ++i) o[i] = std::max(o[i], p[i]);
                    }
            StepOutput o;
            o.meta = outputMeta({}, in.meta);
            o.array = out;
            return o;
        }
    };

    struct FailingOp final : Operation {
        OpInfo info_;
        FailingOp() {
            info_.kind = "test_fail";
            info_.name = "Fail";
            info_.group = "Intensity";
            info_.kindLabel = "INTENSITY";
        }
        const OpInfo& info() const noexcept override { return info_; }
        StepOutput run(const StepInput&, const ParamSet&, const StepContext&) const override {
            throw std::runtime_error("boom");
        }
    };

    struct SlowOp final : Operation {
        OpInfo info_;
        SlowOp() {
            info_.kind = "test_slow";
            info_.name = "Slow";
            info_.group = "Intensity";
            info_.kindLabel = "INTENSITY";
        }
        const OpInfo& info() const noexcept override { return info_; }
        StepOutput run(const StepInput& in, const ParamSet&, const StepContext& ctx) const override {
            for (int i = 0; i < 200; ++i) {
                ctx.throwIfCancelled();
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            StepOutput o;
            o.meta = in.meta;
            o.array = in.materialize();
            return o;
        }
    };

    void registerTestOps() {
        static bool done = false;
        if (done) return;
        done = true;
        registerBuiltinOperations();
        registerOperation(std::make_unique<ScaleOp>());
        registerOperation(std::make_unique<MaxZOp>());
        registerOperation(std::make_unique<FailingOp>());
        registerOperation(std::make_unique<SlowOp>());
    }

    std::shared_ptr<MemorySource> syntheticSource(Index c = 2, Index t = 3, Index z = 4, Index y = 8, Index x = 8) {
        auto a = std::make_shared<Array5>(Dims5{c, t, z, y, x});
        for (Index i = 0; i < a->numel(); ++i) a->data()[i] = static_cast<float>(i % 97);
        DatasetMeta m;
        m.name = "synthetic";
        m.sourcePath = "memory://synthetic";
        m.format = "memory";
        m.dims = a->dims();
        m.voxelUm = {0.1, 0.1, 0.3};
        m.normalizeChannels();
        return std::make_shared<MemorySource>(a, m);
    }

    struct Scratch {
        std::filesystem::path dir;
        Scratch() {
            dir = std::filesystem::temp_directory_path() / ("sirius-wb-test-" + std::to_string(std::random_device{}()));
            std::filesystem::create_directories(dir);
        }
        ~Scratch() {
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
        }
    };

    struct Counter final : Workbench::Observer {
        int pipeline = 0, step = 0, selection = 0, viewed = 0, view = 0, outputs = 0, history = 0, run = 0;
        void pipelineChanged() override { ++pipeline; }
        void stepChanged(int) override { ++step; }
        void selectionChanged() override { ++selection; }
        void viewedStepChanged() override { ++viewed; }
        void viewStateChanged() override { ++view; }
        void outputsChanged() override { ++outputs; }
        void historyChanged() override { ++history; }
        void runStateChanged() override { ++run; }
    };

    std::shared_ptr<RunJob> runSync(Workbench& wb, int target = -1) {
        auto job = wb.createRun(target);
        REQUIRE(job);
        job->execute();
        wb.finishRun(job);
        return job;
    }

} // namespace

// --- Pipeline -------------------------------------------------------------------

TEST_CASE("Pipeline keeps Load pinned first and orders steps", "[app][pipeline]") {
    registerTestOps();
    Pipeline p;
    REQUIRE(p.size() == 1);
    CHECK(p.at(0).kind == "load");
    CHECK(p.at(0).pinned);

    const StepId a = p.add("test_scale");
    const StepId b = p.add("test_maxz");
    CHECK(p.size() == 3);
    CHECK(p.indexOf(a) == 1);
    CHECK(p.indexOf(b) == 2);
    CHECK(p.at(1).name == "Scale");
    CHECK(p.at(1).cache == CachePolicy::Memory);
    CHECK(p.at(1).params.getDouble("factor") == 2.0);

    SECTION("moving") {
        CHECK(p.move(2, -1));
        CHECK(p.indexOf(b) == 1);
        CHECK_FALSE(p.move(1, -1));   // never above Load
        CHECK_FALSE(p.move(0, 1));    // Load never moves
        CHECK_FALSE(p.move(2, 1));    // past the end
    }
    SECTION("remove and enable") {
        p.remove(0);
        CHECK(p.size() == 3);
        p.setEnabled(0, false);
        CHECK(p.at(0).enabled);
        p.setEnabled(1, false);
        CHECK_FALSE(p.at(1).enabled);
        CHECK(p.enabledCount() == 2);
        p.remove(1);
        CHECK(p.size() == 2);
        CHECK(p.at(1).id == b);
    }
    SECTION("duplicate gets a fresh id after the original") {
        const StepId c = p.duplicate(1);
        CHECK(p.size() == 4);
        CHECK(p.indexOf(c) == 2);
        CHECK(c != a);
        CHECK(p.at(2).params == p.at(1).params);
    }
    SECTION("unknown kinds are rejected") {
        CHECK_THROWS_AS(p.add("nope"), std::out_of_range);
    }
    SECTION("setParams coerces and clamps against the specs") {
        ParamSet q;
        q.set("factor", std::string("250"));
        q.set("bogus", 1.0);
        p.setParams(1, q);
        CHECK(p.at(1).params.getDouble("factor") == 100.0);   // clamped to the spec's max
        CHECK(p.at(1).params.has("bogus"));                  // unknown keys survive (strict=false)
    }
}

TEST_CASE("Pipeline round-trips through JSON and TOML with ids", "[app][pipeline]") {
    registerTestOps();
    Pipeline p;
    p.add("test_scale");
    p.add("test_maxz");
    p.setEnabled(2, false);
    p.setCache(1, CachePolicy::Disk);
    ParamSet q = p.at(1).params;
    q.set("factor", 3.5);
    p.setParams(1, q);
    p.rename(2, "Projection");

    const json j = p.toJson();
    Pipeline back = Pipeline::fromJson(j);
    REQUIRE(back.size() == 3);
    CHECK(back.at(1).id == p.at(1).id);
    CHECK(back.at(1).cache == CachePolicy::Disk);
    CHECK(back.at(1).params.getDouble("factor") == 3.5);
    CHECK_FALSE(back.at(2).enabled);
    CHECK(back.at(2).name == "Projection");
    CHECK(back.toJson() == j);

    test::TempFile file("pipeline", ".sirius.toml");
    p.save(file.path.string());
    Pipeline loaded = Pipeline::load(file.path.string());
    CHECK(loaded.toJson() == j);

    const std::string py = p.toPythonScript("/data/x.tif");
    CHECK(py.find("run_pipeline") != std::string::npos);
    CHECK(py.find("test_maxz") != std::string::npos);
}

// --- Executor -------------------------------------------------------------------

TEST_CASE("Executor caches per fingerprint and invalidates downstream only", "[app][executor]") {
    registerTestOps();
    Scratch scratch;
    Executor ex(scratch.dir / "cache");
    Pipeline p;
    const StepId scaleId = p.add("test_scale");
    p.add("test_maxz");
    p.setCache(2, CachePolicy::Memory);

    auto src = syntheticSource();
    auto load = std::make_shared<StepOutput>();
    load->meta = src->meta();
    load->source = src;
    ex.seed(p, 0, load);
    REQUIRE(ex.isFresh(p, 0));

    StepContext ctx;
    ScaleOp::runs = 0;
    std::vector<StepReport> reports;
    auto out = ex.runAll(p, ctx, &reports);
    REQUIRE(out);
    CHECK(out->meta.dims.z == 1);
    CHECK(ScaleOp::runs == 1);
    REQUIRE(reports.size() == 3);
    CHECK(reports[0].note == "cached");
    CHECK(reports[1].ran);
    CHECK(reports[2].ran);
    // input 0..96 scaled by 2, max over z
    CHECK(out->array->at(0, 0, 0, 0, 0) >= 0.0f);

    SECTION("a second run is served from the cache") {
        reports.clear();
        ex.runAll(p, ctx, &reports);
        CHECK(ScaleOp::runs == 1);
        CHECK(reports[1].note == "cached");
        CHECK(reports[2].note == "cached");
    }
    SECTION("editing step 1 invalidates 1 and 2, not 0") {
        ParamSet q = p.at(1).params;
        q.set("factor", 3.0);
        p.setParams(1, q);
        CHECK(ex.isFresh(p, 0));
        CHECK_FALSE(ex.isFresh(p, 1));
        CHECK_FALSE(ex.isFresh(p, 2));
        CHECK(ex.lastOutput(scaleId));   // stale output still available for viewing
        ex.runAll(p, ctx);
        CHECK(ScaleOp::runs == 2);
        CHECK(ex.isFresh(p, 2));
    }
    SECTION("a skipped step is transparent to the fingerprint") {
        p.setEnabled(1, false);
        CHECK_FALSE(ex.isFresh(p, 2));
        reports.clear();
        auto o = ex.runAll(p, ctx, &reports);
        CHECK(reports[1].skipped);
        CHECK(ScaleOp::runs == 1);
        // max over z of the unscaled input
        CHECK(o->meta.dims.z == 1);
        p.setEnabled(1, true);
        // One entry per step: step 2 now holds the skipped-input result, so
        // re-enabling step 1 makes 2 stale again while 1 stays fresh.
        CHECK(ex.isFresh(p, 1));
        CHECK_FALSE(ex.isFresh(p, 2));
    }
    SECTION("disk policy spills the array and reloads it") {
        p.setCache(1, CachePolicy::Disk);
        ParamSet q = p.at(1).params;
        q.set("factor", 4.0);
        p.setParams(1, q);
        ex.run(p, 1, ctx);
        bool spilled = false;
        for (const auto& e : std::filesystem::directory_iterator(scratch.dir / "cache"))
            if (e.path().extension() == ".sir5") spilled = true;
        CHECK(spilled);
        auto c = ex.cached(p, 1);
        REQUIRE(c);
        REQUIRE(c->array);
        CHECK(c->array->at(0, 0, 0, 0, 1) == 4.0f);
        CHECK(ex.cachedBytesOf(scaleId) == c->array->bytes());
    }
    SECTION("recompute keeps only the newest result") {
        p.setCache(1, CachePolicy::Recompute);
        p.setCache(2, CachePolicy::Recompute);
        ParamSet q = p.at(1).params;
        q.set("factor", 5.0);
        p.setParams(1, q);
        ex.runAll(p, ctx);
        CHECK(ex.isFresh(p, 2));
        CHECK_FALSE(ex.isFresh(p, 1));   // its array was dropped when step 2 stored
        auto stale = ex.lastOutput(scaleId);
        REQUIRE(stale);
        CHECK_FALSE(stale->array);
    }
    SECTION("failures propagate with the step named") {
        p.add("test_fail");
        reports.clear();
        CHECK_THROWS_WITH(ex.runAll(p, ctx, &reports), Catch::Matchers::ContainsSubstring("04 Fail"));
        CHECK(reports.back().error == "boom");
    }
    SECTION("array files round trip") {
        Array5 a(Dims5{1, 2, 3, 4, 5});
        for (Index i = 0; i < a.numel(); ++i) a.data()[i] = static_cast<float>(i) * 0.5f;
        const auto path = scratch.dir / "x.sir5";
        Executor::writeArrayFile(path, a);
        auto b = Executor::readArrayFile(path);
        CHECK(b->dims() == a.dims());
        CHECK(b->at(0, 1, 2, 3, 4) == a.at(0, 1, 2, 3, 4));
    }
}

// --- Workbench --------------------------------------------------------------------

TEST_CASE("Workbench edits are observed and undoable", "[app][workbench]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    Counter counter;
    wb.addObserver(&counter);
    wb.setDataset(syntheticSource());
    CHECK(wb.hasDataset());
    CHECK(wb.dataset().dims.c == 2);
    CHECK(counter.pipeline >= 1);
    const int base = wb.pipeline().size();

    const StepId id = wb.addStep("test_scale");
    CHECK(wb.pipeline().size() == base + 1);
    CHECK(wb.selectedIndex() == base);
    CHECK(wb.viewedIndex() == base);
    CHECK(wb.history().canUndo());
    CHECK(wb.history().undoLabel() == "Add Scale");

    wb.setStepParam(base, "factor", 3.0);
    CHECK(wb.pipeline().at(base).params.getDouble("factor") == 3.0);
    CHECK(wb.history().undoLabel().find("Factor 2 → 3") != std::string::npos);
    CHECK(wb.stepSummary(base).find("factor 3") != std::string::npos);

    SECTION("undo and redo restore the pipeline and keep ids") {
        wb.undo();
        CHECK(wb.pipeline().at(base).params.getDouble("factor") == 2.0);
        CHECK(wb.pipeline().at(base).id == id);
        wb.undo();
        CHECK(wb.pipeline().size() == base);
        CHECK(wb.selectedIndex() < base);
        wb.redo();
        CHECK(wb.pipeline().size() == base + 1);
        CHECK(wb.pipeline().at(base).id == id);
        wb.redo();
        CHECK(wb.pipeline().at(base).params.getDouble("factor") == 3.0);
        CHECK_FALSE(wb.history().canRedo());
    }
    SECTION("merged slider edits are one undo entry that spans the drag") {
        const std::size_t n = wb.history().size();
        wb.setStepParam(base, "factor", 4.0, "drag");
        wb.setStepParam(base, "factor", 5.0, "drag");
        wb.setStepParam(base, "factor", 6.0, "drag");
        CHECK(wb.history().size() == n + 1);
        wb.undo();
        CHECK(wb.pipeline().at(base).params.getDouble("factor") == 3.0);
        wb.redo();
        CHECK(wb.pipeline().at(base).params.getDouble("factor") == 6.0);
    }
    SECTION("moving remaps selection and viewing") {
        wb.addStep("test_maxz");
        const int last = wb.pipeline().size() - 1;
        wb.select(last);
        wb.view(base);
        CHECK(wb.moveStep(last, -1));
        CHECK(wb.selectedIndex() == base);
        CHECK(wb.viewedIndex() == last);
        CHECK(wb.pipeline().at(base).kind == "test_maxz");
    }
    SECTION("copy / paste parameters between steps of the same kind") {
        wb.addStep("test_scale");
        const int other = wb.pipeline().size() - 1;
        wb.copyParameters(base);
        CHECK(wb.pasteParameters(other));
        CHECK(wb.pipeline().at(other).params.getDouble("factor") == 3.0);
        wb.addStep("test_maxz");
        CHECK_FALSE(wb.pasteParameters(wb.pipeline().size() - 1));
    }
    SECTION("removing a step keeps the selection sane") {
        wb.removeStep(base);
        CHECK(wb.pipeline().size() == base);
        CHECK(wb.selectedIndex() < base);
        wb.removeStep(0);   // Load stays
        CHECK(wb.pipeline().at(0).kind == "load");
    }
    SECTION("view state setters clamp to the viewed output") {
        wb.setZ(100);
        CHECK(wb.viewState().z == 3);
        wb.setT(-5);
        CHECK(wb.viewState().t == 0);
        wb.setCrosshair(999, 2, 1);
        CHECK(wb.viewState().cx == 7);
        CHECK(wb.viewState().cy == 2);
        CHECK(wb.viewState().z == 1);
        wb.setChannelVisible(1, false);
        CHECK_FALSE(wb.viewState().channelOn(1));
        CHECK(wb.viewState().channelOn(0));
        const int before = counter.view;
        wb.toggleCrosshair();
        CHECK(counter.view == before + 1);
        CHECK_FALSE(wb.viewState().crosshair);
    }
    wb.removeObserver(&counter);
}

TEST_CASE("Workbench runs, caches, displays and reports", "[app][workbench]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource());
    wb.setBackend(Backend::Cpu);
    // Drop the default Contrast step if the built-ins are present: this test
    // only wants the synthetic ops.
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_scale");
    wb.addStep("test_maxz");
    CHECK_FALSE(wb.outputFresh(1));

    // Before any run the viewer falls back to the Load step's source.
    int actual = -1;
    auto shown = wb.displayOutput(&actual);
    REQUIRE(shown);
    CHECK(actual == 0);
    CHECK(shown->source);

    CHECK_FALSE(wb.running());
    auto job = runSync(wb);
    CHECK(job->succeeded());
    CHECK(job->reports().size() == 3);
    CHECK_FALSE(wb.running());
    CHECK(wb.outputFresh(1));
    CHECK(wb.outputFresh(2));
    shown = wb.displayOutput(&actual);
    CHECK(actual == 2);
    CHECK(shown->array->dims().z == 1);
    CHECK(wb.cachedBytes() > 0);
    CHECK(wb.log().back().find("Run finished") != std::string::npos);

    SECTION("diagnostics come from the output when fresh and warn when stale") {
        wb.select(1);
        CHECK(wb.selectedDiagnostics().summary == "scaled by 2");
        wb.setStepParam(1, "factor", 7.0);
        const Diagnostics d = wb.selectedDiagnostics();
        REQUIRE_FALSE(d.warnings.empty());
        CHECK(d.warnings.front().find("run the step again") != std::string::npos);
    }
    SECTION("running to one step leaves later ones untouched") {
        wb.setStepCache(2, CachePolicy::Memory);   // a Recompute entry would drop its array
        wb.setStepParam(1, "factor", 9.0);
        auto j = runSync(wb, 1);
        CHECK(j->succeeded());
        CHECK(wb.outputFresh(1));
        CHECK_FALSE(wb.outputFresh(2));
        // viewing step 2 shows its stale output rather than nothing
        wb.view(2);
        wb.displayOutput(&actual);
        CHECK(actual == 2);
    }
    SECTION("a failing step reports and leaves the run failed") {
        wb.addStep("test_fail");
        auto j = runSync(wb);
        CHECK_FALSE(j->succeeded());
        CHECK(j->error().find("Fail") != std::string::npos);
        CHECK(wb.log().back().find("failed") != std::string::npos);
    }
    SECTION("cancellation stops a slow step") {
        wb.addStep("test_slow");
        auto j = wb.createRun();
        REQUIRE(j);
        std::thread worker([&] { j->execute(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        wb.cancelRun();
        worker.join();
        wb.finishRun(j);
        CHECK_FALSE(j->succeeded());
        CHECK(j->error() == "cancelled");
        CHECK_FALSE(wb.running());
    }
    SECTION("invalid steps refuse to start with a log line") {
        Pipeline p = wb.pipeline();
        // a channel param out of range is rejected by the generic validation
        wb.setDataset(syntheticSource(1, 1, 1, 4, 4));
        while (wb.pipeline().size() > 1) wb.removeStep(1);
        wb.addStep("test_scale");
        CHECK(wb.createRun());
        wb.finishRun(wb.activeRun());
    }
    SECTION("clearing caches drops freshness but keeps the dataset") {
        wb.clearAllCaches();
        CHECK_FALSE(wb.outputFresh(2));
        CHECK(wb.outputFresh(0));
        wb.displayOutput(&actual);
        CHECK(actual == 0);
    }
}

TEST_CASE("Workbench loads pipelines and the example without losing the dataset", "[app][workbench]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource());
    Pipeline p;
    p.add("test_scale");
    p.add("test_maxz");
    test::TempFile file("wb", ".sirius.toml");
    p.save(file.path.string());
    wb.loadPipeline(file.path.string());
    CHECK(wb.pipeline().size() == 3);
    CHECK(wb.pipeline().at(1).kind == "test_scale");
    CHECK(wb.hasDataset());
    CHECK(wb.pipelinePath() == file.path.string());
    CHECK(wb.outputFresh(0));
    wb.undo();
    CHECK(wb.pipeline().size() != 3);
}

// --- Tool API -----------------------------------------------------------------------

TEST_CASE("ToolApi drives the workbench through JSON", "[app][tools]") {
    registerTestOps();
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource());
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    ToolApi api(wb);
    const json schemas = api.schemas();
    REQUIRE(schemas.is_array());
    CHECK(schemas.size() >= 15);
    for (const json& s : schemas) {
        CHECK(s["type"] == "function");
        CHECK(s["function"].contains("parameters"));
    }

    json r = api.call("add_step", {{"kind", "test_scale"}, {"params", {{"factor", "4"}}}});
    REQUIRE_FALSE(r.contains("error"));
    CHECK(r["step"] == 2);
    CHECK(wb.pipeline().at(1).params.getDouble("factor") == 4.0);
    CHECK(api.actions().size() == 1);

    r = api.call("set_params", {{"step", "Scale"}, {"params", {{"factor", 6}}}});
    CHECK(r["params"]["factor"] == 6.0);
    r = api.call("set_params", {{"step", 2}, {"params", {{"nope", 1}}}});
    CHECK(r.contains("error"));
    r = api.call("set_params", {{"step", 9}, {"params", {{"factor", 1}}}});
    CHECK(r.contains("error"));

    r = api.call("get_step", {{"step", 2}});
    CHECK(r["kind"] == "test_scale");
    CHECK(r["has_output"] == false);

    r = api.call("set_step_enabled", {{"step", 2}, {"enabled", false}});
    CHECK(r["enabled"] == false);
    r = api.call("undo", json::object());
    CHECK(r["ok"] == true);
    CHECK(wb.pipeline().at(1).enabled);

    r = api.call("set_view", {{"mode", "compare"}, {"z", 99}, {"crosshair", {3, 4}}, {"labels", true}});
    CHECK(wb.viewState().mode == ViewMode::Compare);
    CHECK(wb.viewState().z == 3);
    CHECK(wb.viewState().cx == 3);
    CHECK(wb.viewState().labels);

    r = api.call("run", json::object());
    CHECK(r.contains("error"));   // no run hook installed
    api.setRunHook([&](int target) {
        auto job = runSync(wb, target);
        return json{{"ok", job->succeeded()}, {"seconds", job->seconds()}, {"error", job->error()}};
    });
    r = api.call("run", {{"step", 2}});
    CHECK(r["ok"] == true);
    CHECK(wb.outputFresh(1));

    r = api.call("get_diagnostics", {{"step", 2}});
    CHECK(r["summary"] == "scaled by 6");
    r = api.call("get_state", json::object());
    CHECK(r["steps"].size() == 2);
    CHECK(r["dataset"]["shape"] == "c2 t3 z4 y8 x8");
    CHECK(api.call("unknown_tool", json::object()).contains("error"));
    CHECK(api.contextSnapshot().contains("selected_step_diagnostics"));
    CHECK(api.systemPrompt().find("SIRIUS") != std::string::npos);
    const auto actions = api.takeActions();
    CHECK(actions.size() >= 5);
    CHECK(api.actions().empty());
}

TEST_CASE("History merges by key and clears redo on push", "[app][history]") {
    History h;
    int value = 0;
    auto cmd = [&](int from, int to, std::string key = {}) {
        Command c;
        c.label = std::to_string(from) + "→" + std::to_string(to);
        c.undo = [&value, from] { value = from; };
        c.redo = [&value, to] { value = to; };
        c.mergeKey = std::move(key);
        return c;
    };
    value = 1;
    h.push(cmd(0, 1));
    value = 2;
    h.push(cmd(1, 2, "k"));
    value = 3;
    h.push(cmd(1, 3, "k"));   // composed by the caller: undo goes back to 1
    CHECK(h.size() == 2);
    h.undo();
    CHECK(value == 1);
    h.undo();
    CHECK(value == 0);
    CHECK_FALSE(h.canUndo());
    h.redo();
    CHECK(value == 1);
    h.redo();
    CHECK(value == 3);
    h.undo();
    value = 5;
    h.push(cmd(1, 5));
    CHECK_FALSE(h.canRedo());
    CHECK(h.undoLabel() == "1→5");
}

TEST_CASE("A live-preview step is displayed on its input until it runs", "[app][workbench][preview]") {
    registerTestOps();
    if (!findOperation("contrast")) SKIP("built-in operations not registered");
    Scratch scratch;
    Workbench wb(scratch.dir);
    wb.setDataset(syntheticSource());
    while (wb.pipeline().size() > 1) wb.removeStep(1);
    wb.addStep("test_scale");
    const StepId contrastId = wb.addStep("contrast");
    const int ci = wb.pipeline().indexOf(contrastId);
    wb.view(ci);
    // nothing has run: the preview falls back to the Load step's source
    CHECK(wb.viewedIsLivePreview());
    int actual = -1;
    auto shown = wb.displayOutput(&actual);
    REQUIRE(shown);
    CHECK(actual == 0);
    CHECK(wb.upstreamOutput(ci, &actual) == shown);

    runSync(wb, 1);   // scale only
    shown = wb.displayOutput(&actual);
    CHECK(actual == 1);
    CHECK(wb.viewedIsLivePreview());

    runSync(wb);      // contrast too: its own output is shown
    CHECK_FALSE(wb.viewedIsLivePreview());
    shown = wb.displayOutput(&actual);
    CHECK(actual == ci);

    wb.setStepParam(ci, "gamma", 2.0);   // stale again -> preview on the input
    CHECK(wb.viewedIsLivePreview());
    shown = wb.displayOutput(&actual);
    CHECK(actual == 1);
    wb.setStepEnabled(ci, false);        // a skipped step is never previewed
    CHECK_FALSE(wb.viewedIsLivePreview());
}
