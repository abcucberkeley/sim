// Unit tests of the shared CPU stage kernels (src/sim_cpu_stages.hpp). The
// public preprocess/separation tests cover the same code through the Eigen
// API on tiny inputs; these exercise the raw-pointer contracts directly, on
// sizes that cross the internal blocking/threading boundaries, against
// naive reference formulations.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "sim_cpu_stages.hpp"
#include "sirius/constants.hpp"

using namespace sirius;
using namespace sirius::simdetail;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
    std::vector<double> randomVector(std::size_t n, unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<double> v(n);
        for (double& x : v) x = dist(rng);
        return v;
    }
} // namespace

TEST_CASE("scaleShift applies (x - sub) * mul", "[sim_cpu_stages]") {
    std::vector<double> v = {1.0, 2.0, 3.0, 4.0, 5.0};
    cpu::scaleShift(v.data(), static_cast<IndexT>(v.size()), 2.0, 10.0);
    const double expected[] = {-10.0, 0.0, 10.0, 20.0, 30.0};
    for (std::size_t i = 0; i < v.size(); ++i) CHECK(v[i] == expected[i]);
}

TEST_CASE("planeSums and scalePlanes act per plane", "[sim_cpu_stages]") {
    constexpr IndexT nplanes = 3, planeElems = 7;
    std::vector<double> data(static_cast<std::size_t>(nplanes * planeElems));
    for (IndexT p = 0; p < nplanes; ++p)
        for (IndexT i = 0; i < planeElems; ++i)
            data[static_cast<std::size_t>(p * planeElems + i)] = static_cast<double>(p + 1);

    std::vector<double> sums(static_cast<std::size_t>(nplanes));
    cpu::planeSums(data.data(), nplanes, planeElems, sums.data());
    CHECK(sums[0] == 7.0);
    CHECK(sums[1] == 14.0);
    CHECK(sums[2] == 21.0);

    const double factors[] = {1.0, 0.5, 2.0};
    cpu::scalePlanes(data.data(), nplanes, planeElems, factors);
    cpu::planeSums(data.data(), nplanes, planeElems, sums.data());
    CHECK(sums[0] == 7.0);
    CHECK(sums[1] == 7.0);
    CHECK(sums[2] == 42.0);
}

TEST_CASE("bleachFactors ties sections to the dir0/phase0 reference", "[sim_cpu_stages]") {
    // (d, p, z) -> index (d*nphases + p)*nz + z with ndirs=2, nphases=2, nz=2
    const std::vector<double> sums = {
        40.0, 20.0,   // d0 p0: z0, z1 (reference)
        8.0,  4.0,    // d0 p1
        80.0, 0.0,    // d1 p0 (z1 dark)
        10.0, 5.0,    // d1 p1
    };
    std::vector<double> f(sums.size());

    SECTION("per z plane") {
        cpu::bleachFactors(sums.data(), 2, 2, 2, false, f.data());
        CHECK_THAT(f[0], WithinRel(1.0, 1e-12));
        CHECK_THAT(f[1], WithinRel(1.0, 1e-12));
        CHECK_THAT(f[2], WithinRel(5.0, 1e-12));
        CHECK_THAT(f[3], WithinRel(5.0, 1e-12));
        CHECK_THAT(f[4], WithinRel(0.5, 1e-12));
        CHECK(f[5] == 1.0);   // dark section: untouched, no division by zero
        CHECK_THAT(f[6], WithinRel(4.0, 1e-12));
        CHECK_THAT(f[7], WithinRel(4.0, 1e-12));
    }
    SECTION("equalizez uses z = 0 of the reference for every plane") {
        cpu::bleachFactors(sums.data(), 2, 2, 2, true, f.data());
        CHECK_THAT(f[1], WithinRel(2.0, 1e-12));   // 40 / 20
        CHECK_THAT(f[3], WithinRel(10.0, 1e-12));  // 40 / 4
        CHECK(f[5] == 1.0);
        CHECK_THAT(f[7], WithinRel(8.0, 1e-12));   // 40 / 5
    }
}

TEST_CASE("edgeApodize matches the per-element reference on a non-square section", "[sim_cpu_stages]") {
    constexpr IndexT nsec = 2, ny = 6, nx = 9;
    constexpr int nap = 2;
    std::vector<double> data = randomVector(static_cast<std::size_t>(nsec * ny * nx), 7);
    std::vector<double> ref = data;

    // reference: straightforward transcription of the cudasirecon kernel
    auto fact = [&](int i) { return 1.0 - std::sin((i + 0.5) / nap * kPi * 0.5); };
    for (IndexT s = 0; s < nsec; ++s) {
        double* img = ref.data() + s * ny * nx;
        for (IndexT k = 0; k < nx; ++k) {
            const double diff = (img[(ny - 1) * nx + k] - img[k]) * 0.5;
            for (int l = 0; l < nap; ++l) {
                img[l * nx + k] += diff * fact(l);
                img[(ny - 1 - l) * nx + k] -= diff * fact(l);
            }
        }
        for (IndexT l = 0; l < ny; ++l) {
            const double diff = (img[l * nx + nx - 1] - img[l * nx]) * 0.5;
            for (int k = 0; k < nap; ++k) {
                img[l * nx + k] += diff * fact(k);
                img[l * nx + nx - 1 - k] -= diff * fact(k);
            }
        }
    }

    cpu::edgeApodize(data.data(), nsec, ny, nx, nap);
    for (std::size_t i = 0; i < data.size(); ++i) CHECK_THAT(data[i], WithinAbs(ref[i], 1e-12));
}

TEST_CASE("edgeApodize clamps the border to the section size", "[sim_cpu_stages]") {
    // napodize larger than the image must not read or write out of bounds and
    // must leave a constant image constant
    std::vector<double> data(3 * 3, 2.5);
    cpu::edgeApodize(data.data(), 1, 3, 3, 10);
    for (double v : data) CHECK_THAT(v, WithinAbs(2.5, 1e-12));
}

TEST_CASE("cosineApodize is the outer product of two sine windows", "[sim_cpu_stages]") {
    constexpr IndexT ny = 5, nx = 8;
    std::vector<double> data(static_cast<std::size_t>(2 * ny * nx), 1.0);
    cpu::cosineApodize(data.data(), 2, ny, nx);
    for (IndexT s = 0; s < 2; ++s)
        for (IndexT y = 0; y < ny; ++y)
            for (IndexT x = 0; x < nx; ++x) {
                const double w = std::sin(kPi * (x + 0.5) / nx) * std::sin(kPi * (y + 0.5) / ny);
                CHECK_THAT(data[static_cast<std::size_t>((s * ny + y) * nx + x)], WithinAbs(w, 1e-12));
            }
}

TEST_CASE("separate equals the naive matrix product across block boundaries", "[sim_cpu_stages]") {
    // n deliberately not a multiple of the internal voxel block so the tail
    // path is exercised as well
    constexpr int nphases = 5, nbands = 5;
    constexpr IndexT n = 3 * 512 + 37;
    const std::vector<double> phases = randomVector(static_cast<std::size_t>(nphases * n), 11);
    const std::vector<double> mat = randomVector(static_cast<std::size_t>(nbands * nphases), 13);

    std::vector<double> bands(static_cast<std::size_t>(nbands * n), -1.0);
    cpu::separate(phases.data(), bands.data(), mat.data(), nphases, nbands, n);

    for (int b = 0; b < nbands; ++b)
        for (IndexT i = 0; i < n; ++i) {
            double acc = 0.0;
            for (int p = 0; p < nphases; ++p)
                acc += mat[static_cast<std::size_t>(b * nphases + p)] *
                       phases[static_cast<std::size_t>(p * n + i)];
            CHECK_THAT(bands[static_cast<std::size_t>(b * n + i)], WithinAbs(acc, 1e-12));
        }
}

TEST_CASE("separate with fewer bands than phases ignores the extra rows", "[sim_cpu_stages]") {
    constexpr int nphases = 3, nbands = 2;
    constexpr IndexT n = 4;
    const std::vector<double> phases = {1, 2, 3, 4,   10, 20, 30, 40,   100, 200, 300, 400};
    const std::vector<double> mat = {1, 0, 0,   0, 1, 1};
    std::vector<double> bands(static_cast<std::size_t>(nbands * n));
    cpu::separate(phases.data(), bands.data(), mat.data(), nphases, nbands, n);
    CHECK(bands[0] == 1.0);
    CHECK(bands[3] == 4.0);
    CHECK(bands[4] == 110.0);
    CHECK(bands[7] == 440.0);
}
