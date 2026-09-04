// End-to-end SIM reconstruction of the bundled test data: raw.tif (3 dirs x
// 5 phases x 9 z of 64x64) + otf.tif must reproduce raw_proc.tif, the output
// of the cudasirecon reference binary (9 x 128 x 128, float32). Runs on the
// CPU always and on the GPU when a CUDA device is available.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>

#include "sirius/buffer.hpp"
#include "sirius/legacy_config.hpp"
#include "sirius/sim_reconstruction.hpp"
#include "sirius/tiff_io.hpp"

using namespace sirius;
using namespace std::filesystem;

namespace {

    struct TestData {
        SIMParameters params;
        OTFRadiallyAveraged otf;
        Eigen::Tensor<double, 3, Eigen::RowMajor> raw;
        Eigen::Tensor<float, 3, Eigen::RowMajor> expected;
    };

    TestData loadTestData() {
        const path dir = SIRIUS_TEST_DATA_DIR;
        SIMParameters params = fromLegacy(loadLegacyConfig((dir / "config.txt").string()));
        OTFRadiallyAveraged otf = loadOTF((dir / "otf.tif").string(), params);
        return TestData{
            std::move(params),
            std::move(otf),
            readTiffStack<double>((dir / "raw.tif").string()),
            readTiffStack<float>((dir / "raw_proc.tif").string()),
        };
    }

    // max |a-b| over the volume, relative to the expected volume's peak
    template <typename TensorA>
    double maxRelDiff(const TensorA& actual, const Eigen::Tensor<float, 3, Eigen::RowMajor>& expected) {
        REQUIRE(actual.dimension(0) == expected.dimension(0));
        REQUIRE(actual.dimension(1) == expected.dimension(1));
        REQUIRE(actual.dimension(2) == expected.dimension(2));
        double peak = 0.0, diff = 0.0;
        for (Eigen::Index i = 0; i < expected.size(); ++i) {
            peak = std::max(peak, std::abs(static_cast<double>(expected.data()[i])));
            diff = std::max(diff, std::abs(static_cast<double>(actual.data()[i]) -
                                           static_cast<double>(expected.data()[i])));
        }
        REQUIRE(peak > 0.0);
        return diff / peak;
    }

    void checkFit(const SimFit& fit, const SIMParameters& params) {
        REQUIRE(fit.k0.size() == 3);
        REQUIRE(params.k0_angles.has_value());
        for (int d = 0; d < 3; ++d) {
            const double mag = std::hypot(fit.k0[d][0], fit.k0[d][1]);
            const double spacing = 1.0 / mag;
            INFO("direction " << d << ": spacing " << spacing << " um");
            // the reference fit lands at ~0.407 um in all three directions
            CHECK(spacing > 0.40);
            CHECK(spacing < 0.415);
            const double angle = std::atan2(fit.k0[d][1], fit.k0[d][0]);
            CHECK(std::abs(angle - (*params.k0_angles)[d]) < 0.05);
            // |amp1| ~ 0.21-0.24, |amp2| ~ 0.72-0.77 for this data
            CHECK(std::abs(fit.amps[d][0]) == 1.0);
            CHECK(std::abs(fit.amps[d][1]) > 0.1);
            CHECK(std::abs(fit.amps[d][1]) < 0.4);
            CHECK(std::abs(fit.amps[d][2]) > 0.6);
            CHECK(std::abs(fit.amps[d][2]) < 0.9);
        }
    }

} // namespace

TEST_CASE("CPU reconstruction reproduces the cudasirecon reference output", "[reconstruction]") {
    TestData t = loadTestData();

    SimReconstructor recon(t.params, t.otf, Device::cpu(), PlanRigor::Estimate);
    Buffer<double> out = recon.reconstruct(t.raw);

    REQUIRE(out.shape() == Shape({9, 128, 128}));
    checkFit(recon.lastFit(), t.params);

    const auto actual = toEigen<3>(out);
    const double rel = maxRelDiff(actual, t.expected);
    INFO("max |actual-expected| / max |expected| = " << rel);
    CHECK(rel < 1e-4);
}

TEST_CASE("Repeated CPU reconstructions of the same input are bit-identical",
          "[reconstruction]") {
    // The k0 bracket search maximizes |modamp|^2, so a reduction whose
    // rounding depends on the OpenMP thread schedule moves the fitted pattern
    // vector and, through it, every output voxel. Reproducibility is required
    // by the Python API contract (recon.reconstruct(x) twice) and is what
    // makes CPU/GPU comparisons meaningful.
    TestData t = loadTestData();

    SimReconstructor recon(t.params, t.otf, Device::cpu(), PlanRigor::Estimate);
    const auto first = toEigen<3>(recon.reconstruct(t.raw));
    const SimFit fit = recon.lastFit();
    const auto second = toEigen<3>(recon.reconstruct(t.raw));

    REQUIRE(first.size() == second.size());
    Eigen::Index differing = 0;
    for (Eigen::Index i = 0; i < first.size(); ++i)
        if (first.data()[i] != second.data()[i]) ++differing;
    INFO(differing << " of " << first.size() << " voxels differ between runs");
    CHECK(differing == 0);

    for (std::size_t d = 0; d < fit.k0.size(); ++d) {
        CHECK(fit.k0[d][0] == recon.lastFit().k0[d][0]);
        CHECK(fit.k0[d][1] == recon.lastFit().k0[d][1]);
    }
}

TEST_CASE("GPU reconstruction reproduces the cudasirecon reference output",
          "[reconstruction][cuda]") {
    if (!cudaAvailable()) SKIP("no CUDA device available");
    const Device gpu = Device::cuda(0);
    TestData t = loadTestData();

    SimReconstructor recon(t.params, t.otf, gpu, PlanRigor::Estimate);
    Stream stream(gpu);
    Buffer<double> dRaw = toDevice(t.raw, gpu, stream);
    stream.synchronize();

    Buffer<double> dOut = recon.reconstruct(dRaw);
    REQUIRE(dOut.device() == gpu);
    REQUIRE(dOut.shape() == Shape({9, 128, 128}));
    checkFit(recon.lastFit(), t.params);

    const auto actual = toEigen<3>(dOut);
    const double rel = maxRelDiff(actual, t.expected);
    INFO("max |actual-expected| / max |expected| = " << rel);
    CHECK(rel < 1e-4);
}

TEST_CASE("CPU and GPU reconstructions agree closely", "[reconstruction][cuda]") {
    if (!cudaAvailable()) SKIP("no CUDA device available");
    const Device gpu = Device::cuda(0);
    TestData t = loadTestData();

    SimReconstructor cpuRecon(t.params, t.otf, Device::cpu(), PlanRigor::Estimate);
    const auto cpuOut = toEigen<3>(cpuRecon.reconstruct(t.raw));

    SimReconstructor gpuRecon(t.params, t.otf, gpu, PlanRigor::Estimate);
    Stream stream(gpu);
    Buffer<double> dRaw = toDevice(t.raw, gpu, stream);
    stream.synchronize();
    const auto gpuOut = toEigen<3>(gpuRecon.reconstruct(dRaw));

    double peak = 0.0, diff = 0.0;
    for (Eigen::Index i = 0; i < cpuOut.size(); ++i) {
        peak = std::max(peak, std::abs(cpuOut.data()[i]));
        diff = std::max(diff, std::abs(cpuOut.data()[i] - gpuOut.data()[i]));
    }
    INFO("max |cpu-gpu| / max |cpu| = " << diff / peak);
    CHECK(diff / peak < 1e-6);
}
