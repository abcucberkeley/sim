#include "core/session.hpp"

#include <sirius/constants.hpp>
#include <sirius/legacy_config.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace sirius::app {

    // --- parameter files -------------------------------------------------

    namespace {
        bool hasExtension(const std::string& path, const char* ext) {
            const auto dot = path.find_last_of('.');
            if (dot == std::string::npos) return false;
            std::string got = path.substr(dot);
            std::transform(got.begin(), got.end(), got.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return got == ext;
        }
    } // namespace

    ParameterFormat detectParameterFormat(const std::string& path) {
        if (hasExtension(path, ".toml")) return ParameterFormat::Toml;

        std::ifstream file(path);
        if (!file) throw std::runtime_error("Failed to open parameter file: " + path);
        std::string line;
        while (std::getline(file, line)) {
            const auto first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) continue;
            const char c = line[first];
            if (c == '#' || c == ';') continue;          // comment in either format
            return c == '[' ? ParameterFormat::Toml : ParameterFormat::Legacy;
        }
        return ParameterFormat::Legacy;   // empty file: the legacy loader yields defaults
    }

    SIMParameters loadParametersAuto(const std::string& path, ParameterFormat* format) {
        const ParameterFormat f = detectParameterFormat(path);
        if (format) *format = f;
        return f == ParameterFormat::Toml ? loadParameters(path)
                                          : fromLegacy(loadLegacyConfig(path));
    }

    // --- fit summary -----------------------------------------------------

    std::vector<FitRow> summarizeFit(const SimFit& fit) {
        std::vector<FitRow> rows;
        rows.reserve(fit.k0.size());
        for (std::size_t d = 0; d < fit.k0.size(); ++d) {
            FitRow r;
            r.direction = static_cast<int>(d);
            r.kx = fit.k0[d][0];
            r.ky = fit.k0[d][1];
            const double mag = std::hypot(r.kx, r.ky);
            r.spacingUm = mag > 0.0 ? 1.0 / mag : 0.0;
            r.angleDeg = std::atan2(r.ky, r.kx) * 180.0 / kPi;
            if (d < fit.amps.size()) {
                r.ampMagnitude.reserve(fit.amps[d].size());
                for (const auto& a : fit.amps[d]) r.ampMagnitude.push_back(std::abs(a));
            }
            rows.push_back(std::move(r));
        }
        return rows;
    }

    // --- session ---------------------------------------------------------

    struct ReconSession::Impl {
        Buffer<double> raw;
        std::string rawPath;
        std::string otfPath;
        SIMParameters params;

        // Anything that invalidates the cached reconstructor bumps a
        // generation counter; comparing counters is cheaper and less
        // error-prone than comparing every parameter field.
        std::uint64_t setupGeneration = 0;   // parameters or OTF
        std::uint64_t rawGeneration = 0;

        struct Cache {
            std::uint64_t setupGeneration = 0;
            std::uint64_t rawGeneration = 0;
            Device device;
            PlanRigor rigor = PlanRigor::Measure;
            std::unique_ptr<SimReconstructor> recon;
            Buffer<double> rawOnDevice;   // CUDA only
        };
        std::optional<Cache> cache;

        Index sectionsPerZ() const noexcept {
            return static_cast<Index>(params.ndirs) * params.nphases;
        }
    };

    ReconSession::ReconSession() : impl_(std::make_unique<Impl>()) {}
    ReconSession::~ReconSession() = default;
    ReconSession::ReconSession(ReconSession&&) noexcept = default;
    ReconSession& ReconSession::operator=(ReconSession&&) noexcept = default;

    void ReconSession::loadRaw(const std::string& path) {
        TiffFile file(path);
        TiffReadOptions opts;
        opts.device = Device::cpu();
        setRaw(file.readStack<double>(opts), path);
    }

    void ReconSession::setRaw(Buffer<double> stack, std::string label) {
        if (stack.rank() != 3)
            throw std::invalid_argument("raw stack must be (sections, ny, nx), got " +
                                        stack.shape().toString());
        if (!stack.device().isCpu())
            throw std::invalid_argument("raw stack must live in host memory");
        impl_->raw = std::move(stack);
        impl_->rawPath = std::move(label);
        ++impl_->rawGeneration;
    }

    bool ReconSession::hasRaw() const noexcept { return !impl_->raw.empty(); }
    const Buffer<double>& ReconSession::raw() const noexcept { return impl_->raw; }
    const std::string& ReconSession::rawPath() const noexcept { return impl_->rawPath; }

    void ReconSession::setOtfPath(std::string path) {
        if (path == impl_->otfPath) return;
        impl_->otfPath = std::move(path);
        ++impl_->setupGeneration;
    }
    const std::string& ReconSession::otfPath() const noexcept { return impl_->otfPath; }

    void ReconSession::setParameters(const SIMParameters& p) {
        impl_->params = p;
        ++impl_->setupGeneration;
    }
    const SIMParameters& ReconSession::parameters() const noexcept { return impl_->params; }

    Index ReconSession::inferredNz() const noexcept {
        if (!hasRaw()) return 0;
        const Index perZ = impl_->sectionsPerZ();
        const Index nsec = impl_->raw.dim(0);
        return (perZ > 0 && nsec % perZ == 0) ? nsec / perZ : 0;
    }

    std::string ReconSession::validate() const {
        if (!hasRaw()) return "No raw stack loaded.";
        if (impl_->otfPath.empty()) return "No OTF file selected.";
        try {
            impl_->params.validate();
        } catch (const std::exception& e) {
            return std::string("Invalid parameters: ") + e.what();
        }
        const Buffer<double>& raw = impl_->raw;
        const Index nx = raw.dim(2), ny = raw.dim(1);
        if (nx < 4 || ny < 4 || nx % 2 != 0 || ny % 2 != 0)
            return "Image size must be even and at least 4 x 4, got " + std::to_string(nx) + " x " +
                   std::to_string(ny) + ".";
        if (inferredNz() == 0)
            return std::to_string(raw.dim(0)) + " sections is not a multiple of ndirs * nphases = " +
                   std::to_string(impl_->sectionsPerZ()) + ".";
        return {};
    }

    ReconResult ReconSession::reconstruct(Device device, PlanRigor rigor) {
        if (const std::string why = validate(); !why.empty()) throw std::runtime_error(why);
        Impl& s = *impl_;

        // Reuse the reconstructor (FFT plans, work buffers) whenever the setup
        // it was built from is unchanged.
        const bool reuse = s.cache && s.cache->setupGeneration == s.setupGeneration &&
                           s.cache->device == device && s.cache->rigor == rigor;
        if (!reuse) {
            Impl::Cache c;
            c.setupGeneration = s.setupGeneration;
            c.device = device;
            c.rigor = rigor;
            c.recon = std::make_unique<SimReconstructor>(s.params, loadOTF(s.otfPath, s.params),
                                                         device, rigor);
            s.cache.emplace(std::move(c));
        }
        Impl::Cache& c = *s.cache;

        BufferView<const double> input = s.raw.view();
        if (device.isCuda()) {
            if (c.rawOnDevice.empty() || c.rawGeneration != s.rawGeneration) {
                c.rawOnDevice = toDevice(s.raw, device);
                synchronizeDevice(device);
            }
            input = c.rawOnDevice.view();
        }
        c.rawGeneration = s.rawGeneration;

        const auto t0 = std::chrono::steady_clock::now();
        Buffer<double> out = c.recon->reconstruct(input);
        const auto t1 = std::chrono::steady_clock::now();

        ReconResult r;
        r.volume = device.isCuda() ? out.to(Device::cpu()) : std::move(out);
        if (device.isCuda()) synchronizeDevice(device);
        r.fit = c.recon->lastFit();
        r.device = device;
        r.seconds = std::chrono::duration<double>(t1 - t0).count();
        r.plansReused = reuse;
        return r;
    }

} // namespace sirius::app
