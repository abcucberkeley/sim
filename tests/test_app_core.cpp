// Tests of the GUI's Qt-free model (app/core): display mapping, parameter
// format detection, fit summary and the ReconSession, including an
// end-to-end reconstruction of the bundled test data through the session
// and the reconstructor/upload caching it promises.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "core/display_mapping.hpp"
#include "core/session.hpp"

#include "sirius/constants.hpp"
#include "sirius/legacy_config.hpp"
#include "sirius/tiff_io.hpp"

#include "temp_path.hpp"

using namespace sirius;
using namespace sirius::app;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

    const std::filesystem::path kData = SIRIUS_TEST_DATA_DIR;

    // Writes `text` to a unique temp file that is removed on destruction.
    struct TempFile : test::TempFile {
        TempFile(const char* suffix, const std::string& text) : test::TempFile("app", suffix) {
            std::ofstream(path) << text;
        }
    };

    SIMParameters testParameters() {
        return fromLegacy(loadLegacyConfig((kData / "config.txt").string()));
    }

} // namespace

// --- display mapping -----------------------------------------------------

TEST_CASE("minMaxRange ignores NaN and reports empty input as invalid", "[app][display]") {
    const double v[] = {3.0, std::numeric_limits<double>::quiet_NaN(), -1.5, 7.25};
    const DisplayRange r = minMaxRange(v, 4);
    CHECK(r.lo == -1.5);
    CHECK(r.hi == 7.25);
    CHECK(r.valid());

    CHECK_FALSE(minMaxRange(v, 0).valid());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(minMaxRange(&nan, 1).valid());
}

TEST_CASE("percentileRange clips outliers and falls back to min/max on a constant", "[app][display]") {
    std::vector<double> v(1000);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<double>(i);   // 0..999
    v[0] = -1e9;    // one cold and one hot pixel
    v[999] = 1e9;

    const DisplayRange r = percentileRange(v.data(), static_cast<Index>(v.size()), 0.01, 0.99);
    CHECK(r.lo > 0.0);
    CHECK(r.lo < 20.0);
    CHECK(r.hi > 980.0);
    CHECK(r.hi < 999.0);

    SECTION("subsampling keeps the estimate bounded and close") {
        const DisplayRange sub = percentileRange(v.data(), static_cast<Index>(v.size()), 0.01, 0.99, 100);
        CHECK(sub.valid());
        CHECK(sub.lo > -1e8);
        CHECK(sub.hi < 1e8);
    }
    SECTION("degenerate window widens to min/max") {
        std::vector<double> flat(50, 4.0);
        flat[10] = 9.0;
        const DisplayRange f = percentileRange(flat.data(), 50, 0.1, 0.9);
        CHECK(f.lo == 4.0);
        CHECK(f.hi == 9.0);
    }
}

TEST_CASE("mapToGray8 clamps, rounds and zeroes NaN", "[app][display]") {
    const double src[] = {-10.0, 0.0, 0.5, 1.0, 20.0, std::numeric_limits<double>::quiet_NaN()};
    std::uint8_t dst[6];
    mapToGray8(src, 6, DisplayRange{0.0, 1.0}, dst);
    CHECK(dst[0] == 0);
    CHECK(dst[1] == 0);
    CHECK(dst[2] == 128);   // round(127.5)
    CHECK(dst[3] == 255);
    CHECK(dst[4] == 255);
    CHECK(dst[5] == 0);

    SECTION("an invalid range maps everything to black") {
        mapToGray8(src, 6, DisplayRange{1.0, 1.0}, dst);
        for (std::uint8_t g : dst) CHECK(g == 0);
    }
}

// --- parameter files -----------------------------------------------------

TEST_CASE("detectParameterFormat distinguishes TOML from legacy configs", "[app][params]") {
    const TempFile toml(".toml", "ndirs = 3\n");
    const TempFile tomlNoExt(".cfg", "# comment\n\n[geometry]\nndirs = 3\n");
    const TempFile legacy(".txt", "# comment\nndirs=3\nnphases=5\n");
    const TempFile empty(".cfg", "");

    CHECK(detectParameterFormat(toml.path.string()) == ParameterFormat::Toml);
    CHECK(detectParameterFormat(tomlNoExt.path.string()) == ParameterFormat::Toml);
    CHECK(detectParameterFormat(legacy.path.string()) == ParameterFormat::Legacy);
    CHECK(detectParameterFormat(empty.path.string()) == ParameterFormat::Legacy);
    CHECK_THROWS(detectParameterFormat((kData / "does_not_exist.cfg").string()));
}

TEST_CASE("loadParametersAuto reads the bundled legacy config and a TOML round trip", "[app][params]") {
    ParameterFormat format{};
    const SIMParameters legacy = loadParametersAuto((kData / "config.txt").string(), &format);
    CHECK(format == ParameterFormat::Legacy);
    CHECK(legacy.ndirs == 3);
    CHECK(legacy.nphases == 5);

    const test::TempFile toml("app_roundtrip", ".toml");
    saveParameters(toml.str, legacy);
    const SIMParameters again = loadParametersAuto(toml.str, &format);
    CHECK(format == ParameterFormat::Toml);
    CHECK(again.ndirs == legacy.ndirs);
    CHECK(again.nphases == legacy.nphases);
    CHECK_THAT(again.wiener, WithinRel(legacy.wiener, 1e-12));
    CHECK_THAT(again.linespacing_um, WithinRel(legacy.linespacing_um, 1e-12));
}

// --- fit summary ---------------------------------------------------------

TEST_CASE("summarizeFit converts k0 to spacing and angle", "[app][fit]") {
    SimFit fit;
    fit.k0 = {{2.0, 0.0}, {0.0, -4.0}, {0.0, 0.0}};
    fit.amps = {{{1.0, 0.0}, {0.0, 0.5}}, {{1.0, 0.0}, {-0.25, 0.0}}};   // third direction has no amps

    const std::vector<FitRow> rows = summarizeFit(fit);
    REQUIRE(rows.size() == 3);
    CHECK(rows[0].direction == 0);
    CHECK_THAT(rows[0].spacingUm, WithinRel(0.5, 1e-12));
    CHECK_THAT(rows[0].angleDeg, WithinAbs(0.0, 1e-12));
    REQUIRE(rows[0].ampMagnitude.size() == 2);
    CHECK_THAT(rows[0].ampMagnitude[1], WithinRel(0.5, 1e-12));

    CHECK_THAT(rows[1].spacingUm, WithinRel(0.25, 1e-12));
    CHECK_THAT(rows[1].angleDeg, WithinAbs(-90.0, 1e-12));
    CHECK_THAT(rows[1].ampMagnitude[1], WithinRel(0.25, 1e-12));

    CHECK(rows[2].spacingUm == 0.0);   // zero vector: no division by zero
    CHECK(rows[2].ampMagnitude.empty());
}

// --- session -------------------------------------------------------------

TEST_CASE("ReconSession validates its inputs before reconstructing", "[app][session]") {
    ReconSession s;
    CHECK_FALSE(s.hasRaw());
    CHECK(s.inferredNz() == 0);
    CHECK_FALSE(s.validate().empty());
    CHECK_THROWS(s.reconstruct(Device::cpu(), PlanRigor::Estimate));

    SECTION("rejects non-stack and device buffers") {
        CHECK_THROWS_AS(s.setRaw(Buffer<double>(Shape{8, 8})), std::invalid_argument);
    }

    s.setRaw(Buffer<double>(Shape{15, 8, 8}), "synthetic");
    CHECK(s.hasRaw());
    CHECK(s.rawPath() == "synthetic");
    CHECK(s.inferredNz() == 1);   // 15 sections / (3 dirs * 5 phases)
    CHECK(s.validate() == "No OTF file selected.");

    s.setOtfPath((kData / "otf.tif").string());
    CHECK(s.validate().empty());

    SECTION("section count must match ndirs * nphases") {
        s.setRaw(Buffer<double>(Shape{14, 8, 8}));
        CHECK(s.inferredNz() == 0);
        CHECK_FALSE(s.validate().empty());
    }
    SECTION("odd image sizes are rejected") {
        s.setRaw(Buffer<double>(Shape{15, 7, 8}));
        CHECK_FALSE(s.validate().empty());
    }
    SECTION("invalid parameters are reported") {
        SIMParameters p = s.parameters();
        p.nphases = 0;
        s.setParameters(p);
        CHECK(s.validate().rfind("Invalid parameters", 0) == 0);
    }
}

TEST_CASE("ReconSession reconstructs the test data and reuses its plans", "[app][session][reconstruction]") {
    ReconSession s;
    s.setParameters(testParameters());
    s.loadRaw((kData / "raw.tif").string());
    s.setOtfPath((kData / "otf.tif").string());
    REQUIRE(s.raw().shape() == Shape({135, 64, 64}));
    REQUIRE(s.inferredNz() == 9);
    REQUIRE(s.validate().empty());

    ReconResult first = s.reconstruct(Device::cpu(), PlanRigor::Estimate);
    REQUIRE(first.volume.shape() == Shape({9, 128, 128}));
    CHECK(first.device == Device::cpu());
    CHECK_FALSE(first.plansReused);
    CHECK(first.seconds > 0.0);
    REQUIRE(first.fit.k0.size() == 3);

    // same output as the reference, as in test_reconstruction
    const auto expected = readTiffStack<float>((kData / "raw_proc.tif").string());
    double peak = 0.0, diff = 0.0;
    for (Index i = 0; i < first.volume.size(); ++i) {
        peak = std::max(peak, std::abs(static_cast<double>(expected.data()[i])));
        diff = std::max(diff, std::abs(first.volume.data()[i] - static_cast<double>(expected.data()[i])));
    }
    CHECK(diff / peak < 1e-4);

    SECTION("a second run with the same setup reuses the reconstructor") {
        ReconResult second = s.reconstruct(Device::cpu(), PlanRigor::Estimate);
        CHECK(second.plansReused);
        CHECK(second.volume.shape() == first.volume.shape());
        for (Index i = 0; i < first.volume.size(); i += 97)
            CHECK(second.volume.data()[i] == first.volume.data()[i]);
    }
    SECTION("changing the parameters rebuilds it") {
        SIMParameters p = s.parameters();
        p.wiener *= 2.0;
        s.setParameters(p);
        ReconResult second = s.reconstruct(Device::cpu(), PlanRigor::Estimate);
        CHECK_FALSE(second.plansReused);
    }
    SECTION("changing the rigor rebuilds it") {
        ReconResult second = s.reconstruct(Device::cpu(), PlanRigor::Measure);
        CHECK_FALSE(second.plansReused);
    }
    SECTION("re-setting the same OTF path does not invalidate") {
        s.setOtfPath((kData / "otf.tif").string());
        CHECK(s.reconstruct(Device::cpu(), PlanRigor::Estimate).plansReused);
    }
}

TEST_CASE("ReconSession reconstructs on the GPU and returns a host volume", "[app][session][cuda]") {
    if (!cudaAvailable()) SKIP("no CUDA device available");
    ReconSession s;
    s.setParameters(testParameters());
    s.loadRaw((kData / "raw.tif").string());
    s.setOtfPath((kData / "otf.tif").string());

    const ReconResult gpu = s.reconstruct(Device::cuda(0), PlanRigor::Estimate);
    REQUIRE(gpu.volume.device().isCpu());
    REQUIRE(gpu.volume.shape() == Shape({9, 128, 128}));
    CHECK(gpu.device == Device::cuda(0));

    // second run: plans and the uploaded raw stack are reused
    CHECK(s.reconstruct(Device::cuda(0), PlanRigor::Estimate).plansReused);

    const ReconResult cpu = s.reconstruct(Device::cpu(), PlanRigor::Estimate);
    double peak = 0.0, diff = 0.0;
    for (Index i = 0; i < cpu.volume.size(); ++i) {
        peak = std::max(peak, std::abs(cpu.volume.data()[i]));
        diff = std::max(diff, std::abs(cpu.volume.data()[i] - gpu.volume.data()[i]));
    }
    CHECK(diff / peak < 1e-6);
}
