// Masked FFT registration (Padfield 2012). The FFT formulation is checked
// against a direct spatial evaluation of the same masked normalized
// cross-correlation, which is the definition the paper accelerates.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/registration.hpp"

using namespace sirius;
using Catch::Matchers::WithinAbs;

namespace {

    struct Volume {
        std::array<Index, 3> extent{1, 1, 1};
        std::vector<double> data;
        std::vector<std::uint8_t> mask;

        Index size() const { return extent[0] * extent[1] * extent[2]; }
        Index at(Index z, Index y, Index x) const { return (z * extent[1] + y) * extent[2] + x; }
        double operator()(Index z, Index y, Index x) const { return data[static_cast<std::size_t>(at(z, y, x))]; }

        BufferView<const double> view() const {
            return {data.data(), Shape{extent[0], extent[1], extent[2]}, Device::cpu()};
        }
        BufferView<const std::uint8_t> maskView() const {
            if (mask.empty()) return {};
            return {mask.data(), Shape{extent[0], extent[1], extent[2]}, Device::cpu()};
        }
    };

    Volume randomVolume(std::array<Index, 3> extent, std::mt19937& rng, bool masked = false,
                        double offset = 0.0) {
        Volume v;
        v.extent = extent;
        std::uniform_real_distribution<double> value(-1.0, 1.0);
        std::bernoulli_distribution valid(0.7);
        v.data.resize(static_cast<std::size_t>(v.size()));
        for (auto& d : v.data) d = value(rng) + offset;
        if (masked) {
            v.mask.resize(v.data.size());
            for (auto& m : v.mask) m = valid(rng) ? 1 : 0;
        }
        return v;
    }

    // The definition the Fourier form implements: over the voxels where the
    // two masks overlap at displacement `shift`, the Pearson correlation of
    // the two images.
    struct Direct { double correlation; Index overlap; };

    Direct directMaskedNcc(const Volume& fixed, const Volume& moving, std::array<Index, 3> shift) {
        double sf = 0, sm = 0, sff = 0, smm = 0, sfm = 0;
        Index n = 0;
        for (Index z = 0; z < moving.extent[0]; ++z)
            for (Index y = 0; y < moving.extent[1]; ++y)
                for (Index x = 0; x < moving.extent[2]; ++x) {
                    const Index fz = z + shift[0], fy = y + shift[1], fx = x + shift[2];
                    if (fz < 0 || fy < 0 || fx < 0) continue;
                    if (fz >= fixed.extent[0] || fy >= fixed.extent[1] || fx >= fixed.extent[2]) continue;
                    const auto mi = static_cast<std::size_t>(moving.at(z, y, x));
                    const auto fi = static_cast<std::size_t>(fixed.at(fz, fy, fx));
                    if (!moving.mask.empty() && moving.mask[mi] == 0) continue;
                    if (!fixed.mask.empty() && fixed.mask[fi] == 0) continue;
                    const double f = fixed.data[fi];
                    const double m = moving.data[mi];
                    sf += f; sm += m; sff += f * f; smm += m * m; sfm += f * m;
                    ++n;
                }
        if (n == 0) return {0.0, 0};
        const double dn = static_cast<double>(n);
        const double num = sfm - sf * sm / dn;
        const double fden = std::max(sff - sf * sf / dn, 0.0);
        const double mden = std::max(smm - sm * sm / dn, 0.0);
        const double den = std::sqrt(fden * mden);
        if (den <= 0.0) return {0.0, n};
        return {std::clamp(num / den, -1.0, 1.0), n};
    }

    // Largest absolute deviation between the FFT map and the direct
    // evaluation, over the displacements whose overlap is big enough that the
    // coefficient is numerically meaningful.
    double compareToDirect(const MaskedNccResult& ncc, const Volume& fixed, const Volume& moving,
                           Index minOverlap) {
        const Shape& s = ncc.correlation.shape();
        double worst = 0.0;
        for (Index z = 0; z < s[0]; ++z)
            for (Index y = 0; y < s[1]; ++y)
                for (Index x = 0; x < s[2]; ++x) {
                    const Index i = (z * s[1] + y) * s[2] + x;
                    const auto shift = ncc.shiftAt({z, y, x});
                    const Direct d = directMaskedNcc(fixed, moving, shift);
                    REQUIRE(static_cast<Index>(ncc.overlap.data()[i]) == d.overlap);
                    if (d.overlap < minOverlap) continue;
                    worst = std::max(worst, std::abs(ncc.correlation.data()[i] - d.correlation));
                }
        return worst;
    }

    // Copy a sub-block out of a volume; the ground-truth displacement of the
    // block relative to the source is -origin.
    Volume crop(const Volume& src, std::array<Index, 3> origin, std::array<Index, 3> extent) {
        Volume out;
        out.extent = extent;
        out.data.resize(static_cast<std::size_t>(out.size()));
        for (Index z = 0; z < extent[0]; ++z)
            for (Index y = 0; y < extent[1]; ++y)
                for (Index x = 0; x < extent[2]; ++x)
                    out.data[static_cast<std::size_t>(out.at(z, y, x))] =
                        src(z + origin[0], y + origin[1], x + origin[2]);
        return out;
    }

} // namespace

TEST_CASE("nextFastFFTSize returns the next 2/3/5/7-smooth size", "[registration]") {
    CHECK(nextFastFFTSize(0) == 1);
    CHECK(nextFastFFTSize(1) == 1);
    CHECK(nextFastFFTSize(16) == 16);
    CHECK(nextFastFFTSize(17) == 18);   // 2 * 3^2
    CHECK(nextFastFFTSize(11) == 12);
    CHECK(nextFastFFTSize(13) == 14);   // 2 * 7
    CHECK(nextFastFFTSize(1021) == 1024);
    for (Index n = 1; n < 400; ++n) {
        const Index m = nextFastFFTSize(n);
        REQUIRE(m >= n);
        Index r = m;
        for (Index f : {Index(2), Index(3), Index(5), Index(7)})
            while (r % f == 0) r /= f;
        REQUIRE(r == 1);
    }
}

TEST_CASE("unmasked correlation matches the direct normalized cross-correlation",
          "[registration]") {
    std::mt19937 rng(7);
    const Volume fixed = randomVolume({1, 12, 15}, rng);
    const Volume moving = randomVolume({1, 7, 5}, rng);

    const auto ncc = maskedNormalizedCrossCorrelation<double>(fixed.view(), moving.view(), {}, {});
    REQUIRE(ncc.correlation.shape() == Shape{1, 18, 19});
    CHECK(compareToDirect(ncc, fixed, moving, 4) < 1e-9);
}

TEST_CASE("masked correlation matches the direct masked normalized cross-correlation",
          "[registration]") {
    std::mt19937 rng(11);
    // A non-zero mean is the case a naive "zero the masked voxels and
    // correlate" implementation gets wrong: the zeros act as data.
    const Volume fixed = randomVolume({1, 14, 11}, rng, /*masked=*/true, /*offset=*/3.0);
    const Volume moving = randomVolume({1, 6, 8}, rng, /*masked=*/true, /*offset=*/-2.0);

    const auto ncc = maskedNormalizedCrossCorrelation<double>(
        fixed.view(), moving.view(), fixed.maskView(), moving.maskView());
    CHECK(compareToDirect(ncc, fixed, moving, 6) < 1e-9);
}

TEST_CASE("masked correlation matches the direct evaluation in 3D", "[registration]") {
    std::mt19937 rng(23);
    const Volume fixed = randomVolume({5, 9, 8}, rng, /*masked=*/true, /*offset=*/1.5);
    const Volume moving = randomVolume({3, 4, 5}, rng, /*masked=*/true);

    const auto ncc = maskedNormalizedCrossCorrelation<double>(
        fixed.view(), moving.view(), fixed.maskView(), moving.maskView());
    REQUIRE(ncc.correlation.shape() == Shape{7, 12, 12});
    CHECK(compareToDirect(ncc, fixed, moving, 8) < 1e-9);
}

TEST_CASE("registration recovers the translation of a cropped block", "[registration]") {
    std::mt19937 rng(101);
    const Volume scene = randomVolume({4, 40, 48}, rng);
    const std::array<Index, 3> origin{1, 12, 9};
    const Volume block = crop(scene, origin, {2, 14, 16});

    const auto result = registerTranslationMasked<double>(scene.view(), block.view(), {}, {});
    REQUIRE(result.valid);
    // moving[p] matches fixed[p + shift], and the block was taken from
    // `origin`, so the shift is the origin itself.
    CHECK(result.integerShift == origin);
    CHECK(result.correlation > 0.999);
    CHECK(result.overlap == static_cast<double>(block.size()));
    // The parabola through a noise-like peak is not perfectly symmetric, so
    // the refined shift sits a small fraction of a voxel off the integer one.
    for (int a = 0; a < 3; ++a)
        CHECK_THAT(result.shift[a], WithinAbs(static_cast<double>(origin[a]), 0.05));
}

TEST_CASE("masking rescues a registration that corrupted data would break", "[registration]") {
    // Two overlapping views of the same scene. The moving tile has a bright
    // artefact (a saturated block, or the zero fill left by a deskew) that
    // dominates an unmasked correlation and pulls the peak away; masking it
    // out is exactly the case the paper is about.
    std::mt19937 rng(1234);
    const Volume scene = randomVolume({1, 64, 64}, rng);
    const std::array<Index, 3> origin{0, 20, 24};
    Volume tile = crop(scene, origin, {1, 32, 32});

    tile.mask.assign(tile.data.size(), 1);
    for (Index y = 0; y < 14; ++y)
        for (Index x = 0; x < 32; ++x) {
            const auto i = static_cast<std::size_t>(tile.at(0, y, x));
            tile.data[i] = 60.0;      // constant, far outside the scene's range
            tile.mask[i] = 0;
        }

    MaskedNccOptions options;
    options.requiredOverlapFraction = 0.25;

    const auto unmasked = registerTranslationMasked<double>(scene.view(), tile.view(), {}, {}, options);
    const auto masked = registerTranslationMasked<double>(scene.view(), tile.view(), {},
                                                          tile.maskView(), options);
    REQUIRE(masked.valid);
    CHECK(masked.integerShift == origin);
    CHECK(masked.correlation > 0.999);
    // The unmasked run sees a strong constant plateau and lands elsewhere.
    CHECK(unmasked.integerShift != origin);
}

TEST_CASE("the search range and overlap filters constrain the peak", "[registration]") {
    std::mt19937 rng(55);
    const Volume scene = randomVolume({1, 32, 32}, rng);
    const std::array<Index, 3> origin{0, 11, 7};
    const Volume block = crop(scene, origin, {1, 10, 10});

    MaskedNccOptions tight;
    tight.maxShift = {0, 4, 4};
    const auto limited = registerTranslationMasked<double>(scene.view(), block.view(), {}, {}, tight);
    REQUIRE(limited.valid);
    CHECK(limited.integerShift != origin);
    CHECK(std::abs(limited.integerShift[1]) <= 4);
    CHECK(std::abs(limited.integerShift[2]) <= 4);

    MaskedNccOptions wide;
    wide.maxShift = {0, 20, 20};
    const auto found = registerTranslationMasked<double>(scene.view(), block.view(), {}, {}, wide);
    REQUIRE(found.valid);
    CHECK(found.integerShift == origin);

    // Demanding an overlap the images never reach leaves nothing to report.
    MaskedNccOptions impossible;
    impossible.requiredOverlapVoxels = 10 * 10 + 1;
    CHECK_FALSE(registerTranslationMasked<double>(scene.view(), block.view(), {}, {}, impossible).valid);
}

TEST_CASE("subpixel refinement follows a sub-voxel shift", "[registration]") {
    // A Gaussian blob sampled at two centres half a voxel apart: the integer
    // peak can only be one of the two neighbours, the parabola fit should
    // land between them.
    const Index n = 48;
    auto gaussian = [n](double cy, double cx) {
        Volume v;
        v.extent = {1, n, n};
        v.data.resize(static_cast<std::size_t>(n * n));
        for (Index y = 0; y < n; ++y)
            for (Index x = 0; x < n; ++x) {
                const double dy = static_cast<double>(y) - cy;
                const double dx = static_cast<double>(x) - cx;
                v.data[static_cast<std::size_t>(y * n + x)] = std::exp(-(dy * dy + dx * dx) / 32.0);
            }
        return v;
    };
    const Volume fixed = gaussian(24.0, 24.0);
    const Volume moving = gaussian(24.0 - 3.4, 24.0 - 2.5);   // moving must shift by (3.4, 2.5)

    const auto result = registerTranslationMasked<double>(fixed.view(), moving.view(), {}, {});
    REQUIRE(result.valid);
    CHECK_THAT(result.shift[1], WithinAbs(3.4, 0.15));
    CHECK_THAT(result.shift[2], WithinAbs(2.5, 0.15));
    CHECK(result.shift[0] == 0.0);
}

TEST_CASE("a reused correlator repeats itself exactly", "[registration]") {
    std::mt19937 rng(9);
    const Volume fixed = randomVolume({2, 20, 24}, rng, /*masked=*/true);
    const Volume moving = randomVolume({2, 12, 10}, rng, /*masked=*/true);

    MaskedCorrelator correlator(fixed.view().shape(), moving.view().shape());
    const auto e = correlator.correlationExtent();
    CHECK(e == std::array<Index, 3>{3, 31, 33});
    CHECK(correlator.paddedExtent() == std::array<Index, 3>{3, 32, 35});
    CHECK(correlator.workingBytes() > 0);

    Buffer<double> c1(Shape{e[0], e[1], e[2]}), o1(Shape{e[0], e[1], e[2]});
    Buffer<double> c2(Shape{e[0], e[1], e[2]}), o2(Shape{e[0], e[1], e[2]});
    correlator.correlate<double>(fixed.view(), moving.view(), fixed.maskView(), moving.maskView(),
                                 c1.view(), o1.view());
    correlator.correlate<double>(fixed.view(), moving.view(), fixed.maskView(), moving.maskView(),
                                 c2.view(), o2.view());
    for (Index i = 0; i < c1.size(); ++i) {
        REQUIRE(c1.data()[i] == c2.data()[i]);
        REQUIRE(o1.data()[i] == o2.data()[i]);
    }
}

TEST_CASE("integer pixel types register like doubles", "[registration]") {
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> value(0, 4000);
    const Index n = 40;
    std::vector<std::uint16_t> scene(static_cast<std::size_t>(n * n));
    for (auto& s : scene) s = static_cast<std::uint16_t>(value(rng));

    const Index bn = 12;
    const Index oy = 9, ox = 17;
    std::vector<std::uint16_t> block(static_cast<std::size_t>(bn * bn));
    for (Index y = 0; y < bn; ++y)
        for (Index x = 0; x < bn; ++x)
            block[static_cast<std::size_t>(y * bn + x)] =
                scene[static_cast<std::size_t>((y + oy) * n + (x + ox))];

    BufferView<const std::uint16_t> sceneView(scene.data(), Shape{n, n}, Device::cpu());
    BufferView<const std::uint16_t> blockView(block.data(), Shape{bn, bn}, Device::cpu());
    const auto result = registerTranslationMasked<std::uint16_t>(sceneView, blockView, {}, {});
    REQUIRE(result.valid);
    CHECK(result.integerShift == std::array<Index, 3>{0, oy, ox});
    CHECK(result.correlation > 0.999);
}

TEST_CASE("registration rejects malformed inputs", "[registration]") {
    std::mt19937 rng(2);
    const Volume fixed = randomVolume({1, 8, 8}, rng);
    const Volume moving = randomVolume({1, 4, 4}, rng);
    std::vector<std::uint8_t> wrongMask(9, 1);
    BufferView<const std::uint8_t> badMask(wrongMask.data(), Shape{3, 3}, Device::cpu());

    CHECK_THROWS_AS(maskedNormalizedCrossCorrelation<double>(fixed.view(), moving.view(), badMask, {}),
                    std::invalid_argument);

    Buffer<double> rank4(Shape{2, 2, 2, 2});
    CHECK_THROWS_AS(maskedNormalizedCrossCorrelation<double>(rank4.view(), moving.view(), {}, {}),
                    std::invalid_argument);
}
