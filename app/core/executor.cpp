#include "core/executor.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "core/array_source.hpp"

namespace sirius::app {

    namespace {
        // FNV-1a over the bytes: stable across runs (std::hash is not), cheap,
        // and collisions between the handful of steps of one session are not
        // a realistic concern.
        std::uint64_t fnv1a(const std::string& s) noexcept {
            std::uint64_t h = 1469598103934665603ull;
            for (unsigned char c : s) {
                h ^= c;
                h *= 1099511628211ull;
            }
            return h;
        }
    } // namespace

    std::string stableHash(const std::string& s) {
        char buf[24];
        std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(fnv1a(s)));
        return buf;
    }

    struct Executor::Entry {
        StepId id = 0;
        std::string fingerprint;
        CachePolicy policy = CachePolicy::Recompute;
        // Everything but the array stays in memory; the array is here for
        // Memory (and, until the next run, Recompute) and on disk for Disk.
        std::shared_ptr<const StepOutput> output;
        std::filesystem::path diskPath;
        std::size_t bytes = 0;
        bool arrayOnDisk = false;
        std::weak_ptr<const StepOutput> restored;   // the reloaded copy while someone holds it
    };

    Executor::Executor(std::filesystem::path scratchDir) : scratch_(std::move(scratchDir)) {
        std::error_code ec;
        std::filesystem::create_directories(scratch_, ec);
    }

    Executor::~Executor() {
        std::lock_guard<std::mutex> g(mutex_);
        for (auto& [id, e] : entries_)
            if (e && !e->diskPath.empty()) {
                std::error_code ec;
                std::filesystem::remove(e->diskPath, ec);
            }
    }

    std::string Executor::fingerprint(const Pipeline& p, int index) const {
        std::string upstream;
        for (int i = 0; i <= index && i < p.size(); ++i) {
            const Step& s = p.at(i);
            if (i > 0 && !s.enabled) continue;   // a skipped step is transparent
            const std::string own = s.kind + "|" + s.params.toJson().dump() + "|" + upstream;
            upstream = stableHash(own);
        }
        return upstream;
    }

    std::shared_ptr<const StepOutput> Executor::load(Entry& e) const {
        if (!e.arrayOnDisk || !e.output) return e.output;
        // Reload the spilled array into a fresh StepOutput; the entry keeps
        // the on-disk copy so memory can be released again later. While a
        // caller (the viewer, a paint stroke) still holds the restored output
        // it is handed out again: one object, one read, instead of the whole
        // array coming off the disk for every refresh.
        if (auto held = e.restored.lock()) return held;
        auto restored = std::make_shared<StepOutput>(*e.output);
        restored->array = readArrayFile(e.diskPath);
        e.restored = restored;
        return restored;
    }

    std::shared_ptr<const StepOutput> Executor::cached(const Pipeline& p, int index) const {
        if (index < 0 || index >= p.size()) return nullptr;
        const Step& s = p.at(index);
        const std::string fp = fingerprint(p, index);
        std::lock_guard<std::mutex> g(mutex_);
        auto it = entries_.find(s.id);
        if (it == entries_.end() || !it->second || it->second->fingerprint != fp) return nullptr;
        Entry& e = *it->second;
        if (!e.output) return nullptr;
        if (e.policy == CachePolicy::Recompute && !e.output->array && !e.output->source) return nullptr;
        return load(e);
    }

    std::shared_ptr<const StepOutput> Executor::lastOutput(StepId id) const {
        std::lock_guard<std::mutex> g(mutex_);
        auto it = entries_.find(id);
        if (it == entries_.end() || !it->second || !it->second->output) return nullptr;
        return load(*it->second);
    }

    void Executor::refreshPolicies(const Pipeline& p) {
        // Policies live in the pipeline and may change after an entry was
        // stored; the eviction pass below must see the current ones.
        for (auto& [id, e] : entries_)
            if (e)
                if (const Step* s = p.find(id)) e->policy = s->cache;
    }

    void Executor::store(const Step& step, const std::string& fp, std::shared_ptr<const StepOutput> out) {
        std::lock_guard<std::mutex> g(mutex_);
        auto& slot = entries_[step.id];
        if (!slot) slot = std::make_unique<Entry>();
        Entry& e = *slot;
        if (!e.diskPath.empty()) {
            std::error_code ec;
            std::filesystem::remove(e.diskPath, ec);
            e.diskPath.clear();
        }
        e.id = step.id;
        e.fingerprint = fp;
        e.policy = step.cache;
        e.arrayOnDisk = false;
        e.bytes = out && out->array ? out->array->bytes() : 0;
        if (step.cache == CachePolicy::Disk && out && out->array && !out->array->empty()) {
            e.diskPath = scratch_ / ("step-" + std::to_string(step.id) + "-" + fp + ".sir5");
            writeArrayFile(e.diskPath, *out->array);
            auto shell = std::make_shared<StepOutput>(*out);
            shell->array = nullptr;
            e.output = shell;
            e.arrayOnDisk = true;
        } else {
            e.output = std::move(out);
        }
        // "Recompute" keeps nothing beyond the most recent result: drop the
        // arrays of every other recompute entry now that a newer one exists.
        for (auto& [id, other] : entries_) {
            if (!other || id == step.id || other->policy != CachePolicy::Recompute || !other->output) continue;
            if (other->output->array) {
                auto shell = std::make_shared<StepOutput>(*other->output);
                shell->array = nullptr;
                other->output = shell;
                other->bytes = 0;
            }
        }
    }

    std::shared_ptr<const StepOutput> Executor::run(const Pipeline& p, int index, const StepContext& ctx,
                                                    std::vector<StepReport>* reports,
                                                    const std::function<void(const StepReport&)>& onStep) {
        if (index < 0 || index >= p.size()) throw std::out_of_range("Executor::run: no step " + std::to_string(index));
        std::shared_ptr<const StepOutput> current;
        for (int i = 0; i <= index; ++i) {
            const Step& step = p.at(i);
            StepReport report;
            report.id = step.id;
            report.index = i;
            if (i > 0 && !step.enabled) {
                report.skipped = true;
                report.note = "skipped";
                if (reports) reports->push_back(report);
                if (onStep) onStep(report);
                continue;
            }
            const std::string fp = fingerprint(p, i);
            if (auto c = cached(p, i)) {
                current = std::move(c);
                report.note = "cached";
                if (reports) reports->push_back(report);
                if (onStep) onStep(report);
                continue;
            }
            report.note = "running";
            if (onStep) onStep(report);
            ctx.throwIfCancelled();

            const Operation& op = step.op();
            StepInput input;
            if (i > 0) {
                if (!current) throw std::runtime_error("step " + Step::number(i) + " has no input");
                input = current->asInput();
            }
            const auto t0 = std::chrono::steady_clock::now();
            std::shared_ptr<StepOutput> out;
            try {
                out = std::make_shared<StepOutput>(op.run(input, step.params, ctx));
            } catch (const std::exception& e) {
                report.error = e.what();
                if (reports) reports->push_back(report);
                if (onStep) onStep(report);
                if (std::string(e.what()) == "cancelled") throw;   // not the step's fault
                throw std::runtime_error("Step " + Step::number(i) + " " + step.name + ": " + e.what());
            }
            const auto t1 = std::chrono::steady_clock::now();
            out->seconds = std::chrono::duration<double>(t1 - t0).count();
            if (out->meta.dims.numel() <= 0) out->meta.dims = input.meta.dims;
            // labels carried through unless the step produced its own
            if (!out->labels && input.labels) out->labels = std::const_pointer_cast<LabelVolume>(input.labels);
            {
                std::lock_guard<std::mutex> g(mutex_);
                refreshPolicies(p);
            }
            store(step, fp, out);
            current = out;
            report.ran = true;
            report.seconds = out->seconds;
            report.note = out->note;
            if (reports) reports->push_back(report);
            if (onStep) onStep(report);
        }
        return current;
    }

    std::shared_ptr<const StepOutput> Executor::runAll(const Pipeline& p, const StepContext& ctx,
                                                       std::vector<StepReport>* reports,
                                                       const std::function<void(const StepReport&)>& onStep) {
        return run(p, p.size() - 1, ctx, reports, onStep);
    }

    void Executor::seed(const Pipeline& p, int index, std::shared_ptr<const StepOutput> out) {
        if (index < 0 || index >= p.size()) return;
        {
            std::lock_guard<std::mutex> g(mutex_);
            refreshPolicies(p);
        }
        store(p.at(index), fingerprint(p, index), std::move(out));
    }

    void Executor::invalidate(StepId id) {
        std::lock_guard<std::mutex> g(mutex_);
        auto it = entries_.find(id);
        if (it == entries_.end()) return;
        if (it->second && !it->second->diskPath.empty()) {
            std::error_code ec;
            std::filesystem::remove(it->second->diskPath, ec);
        }
        entries_.erase(it);
    }

    void Executor::clear() {
        std::lock_guard<std::mutex> g(mutex_);
        for (auto& [id, e] : entries_)
            if (e && !e->diskPath.empty()) {
                std::error_code ec;
                std::filesystem::remove(e->diskPath, ec);
            }
        entries_.clear();
    }

    std::size_t Executor::cachedBytes() const {
        std::lock_guard<std::mutex> g(mutex_);
        std::size_t total = 0;
        for (const auto& [id, e] : entries_)
            if (e && e->output) {
                if (e->output->array) total += e->output->array->bytes();
                if (e->output->labels) total += static_cast<std::size_t>(e->output->labels->t() * e->output->labels->volumeSize()) * sizeof(std::uint32_t);
            }
        return total;
    }

    std::size_t Executor::cachedBytesOf(StepId id) const {
        std::lock_guard<std::mutex> g(mutex_);
        auto it = entries_.find(id);
        if (it == entries_.end() || !it->second) return 0;
        return it->second->bytes;
    }

    // --- spill files --------------------------------------------------------------

    namespace {
        constexpr char kMagic[8] = {'S', 'I', 'R', '5', 'A', 'R', 'R', '1'};
    }

    void Executor::writeArrayFile(const std::filesystem::path& path, const Array5& a) {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("cannot write cache file " + path.string());
        out.write(kMagic, sizeof kMagic);
        const Index dims[5] = {a.dims().c, a.dims().t, a.dims().z, a.dims().y, a.dims().x};
        out.write(reinterpret_cast<const char*>(dims), sizeof dims);
        out.write(reinterpret_cast<const char*>(a.data()), static_cast<std::streamsize>(a.bytes()));
        if (!out) throw std::runtime_error("short write to cache file " + path.string());
    }

    std::shared_ptr<Array5> Executor::readArrayFile(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot read cache file " + path.string());
        char magic[8];
        in.read(magic, sizeof magic);
        if (std::memcmp(magic, kMagic, sizeof magic) != 0) throw std::runtime_error("not a cache file: " + path.string());
        Index dims[5];
        in.read(reinterpret_cast<char*>(dims), sizeof dims);
        auto a = std::make_shared<Array5>(Dims5{dims[0], dims[1], dims[2], dims[3], dims[4]});
        in.read(reinterpret_cast<char*>(a->data()), static_cast<std::streamsize>(a->bytes()));
        if (!in) throw std::runtime_error("short read from cache file " + path.string());
        return a;
    }

} // namespace sirius::app
