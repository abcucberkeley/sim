// Richardson-Lucy deconvolution: a known object blurred with the Gaussian
// PSF must come back closer to the truth than the blurred input, with a
// monotonically settling iteration, in 3D and on a single plane.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/deconvolution.hpp"

using namespace sirius;
using Catch::Matchers::WithinAbs;

namespace {

    // Two bright beads on a dim background, the kind of scene deconvolution
    // is expected to sharpen without inventing structure.
    Buffer<float> beads(Index nz, Index ny, Index nx) {
        Buffer<float> v(Shape{nz, ny, nx});
        for (Index z = 0; z < nz; ++z)
            for (Index y = 0; y < ny; ++y)
                for (Index x = 0; x < nx; ++x) {
                    float val = 10.0f;
                    auto bead = [&](Index bz, Index by, Index bx, float amp) {
                        const double d2 = static_cast<double>((z - bz) * (z - bz) + (y - by) * (y - by) + (x - bx) * (x - bx));
                        if (d2 <= 2.25) val += amp;
                    };
                    bead(nz / 2, ny / 3, nx / 3, 500.0f);
                    bead(nz / 2, 2 * ny / 3, 2 * nx / 3, 300.0f);
                    v.data()[(z * ny + y) * nx + x] = val;
                }
        return v;
    }

    // Direct (spatial) convolution with a centred PSF, edges clamped, so the
    // FFT path is checked against an independent definition of the blur.
    Buffer<float> blurDirect(const Buffer<float>& v, const Buffer<float>& psf) {
        const Index nz = v.dim(0), ny = v.dim(1), nx = v.dim(2);
        const Index pz = psf.dim(0), py = psf.dim(1), px = psf.dim(2);
        Buffer<float> out(v.shape());
        for (Index z = 0; z < nz; ++z)
            for (Index y = 0; y < ny; ++y)
                for (Index x = 0; x < nx; ++x) {
                    double acc = 0.0;
                    for (Index kz = 0; kz < pz; ++kz)
                        for (Index ky = 0; ky < py; ++ky)
                            for (Index kx = 0; kx < px; ++kx) {
                                const Index sz = std::clamp<Index>(z - (kz - pz / 2), 0, nz - 1);
                                const Index sy = std::clamp<Index>(y - (ky - py / 2), 0, ny - 1);
                                const Index sx = std::clamp<Index>(x - (kx - px / 2), 0, nx - 1);
                                acc += psf.data()[(kz * py + ky) * px + kx] * v.data()[(sz * ny + sy) * nx + sx];
                            }
                    out.data()[(z * ny + y) * nx + x] = static_cast<float>(acc);
                }
        return out;
    }

    double rmse(const Buffer<float>& a, const Buffer<float>& b) {
        double s = 0.0;
        for (Index i = 0; i < a.size(); ++i) {
            const double d = a.data()[i] - b.data()[i];
            s += d * d;
        }
        return std::sqrt(s / static_cast<double>(a.size()));
    }

} // namespace

TEST_CASE("gaussianPsf is centred, positive and normalized", "[deconvolution]") {
    const Buffer<float> psf = gaussianPsf(7, 9, 9, 0.2, 0.1, 1.4, 520.0, 1.515);
    REQUIRE(psf.shape() == Shape{7, 9, 9});
    double sum = 0.0;
    float peak = 0.0f;
    Index argmax = 0;
    for (Index i = 0; i < psf.size(); ++i) {
        CHECK(psf.data()[i] > 0.0f);
        sum += psf.data()[i];
        if (psf.data()[i] > peak) { peak = psf.data()[i]; argmax = i; }
    }
    CHECK_THAT(sum, WithinAbs(1.0, 1e-5));
    CHECK(argmax == (3 * 9 + 4) * 9 + 4);
    // the axial width is the larger one
    CHECK(psf.data()[(3 * 9 + 4) * 9 + 5] < psf.data()[(4 * 9 + 4) * 9 + 4]);
}

TEST_CASE("Richardson-Lucy sharpens a blurred volume", "[deconvolution]") {
    const Buffer<float> truth = beads(9, 24, 24);
    const Buffer<float> psf = gaussianPsf(5, 7, 7, 0.2, 0.1, 1.2, 520.0, 1.33);
    const Buffer<float> blurred = blurDirect(truth, psf);
    const double errBlurred = rmse(blurred, truth);

    Buffer<float> estimate = blurred.clone();
    DeconvolutionOptions opt;
    opt.iterations = 30;
    const DeconvolutionResult r = richardsonLucy(estimate.view(), psf.view(), opt);

    CHECK(r.iterations == 30);
    CHECK_FALSE(r.stoppedEarly);
    CHECK_FALSE(r.ranOnGpu);
    REQUIRE(r.relativeChange.size() == 30);
    const double errDecon = rmse(estimate, truth);
    INFO("rmse blurred " << errBlurred << " deconvolved " << errDecon);
    // A flat-top bead is the hard case for Richardson-Lucy (it sharpens into
    // a peak before the plateau fills in), so the error to the truth drops
    // moderately here; the data fidelity, which the iteration maximizes,
    // must improve by a lot.
    CHECK(errDecon < 0.8 * errBlurred);
    CHECK(rmse(blurDirect(estimate, psf), blurred) < 0.3 * rmse(blurDirect(blurred, psf), blurred));
    // the iteration settles: the last change is well below the first
    CHECK(r.relativeChange.back() < 0.25 * r.relativeChange.front());
    for (std::size_t i = 5; i < r.relativeChange.size(); ++i) CHECK(r.relativeChange[i] <= r.relativeChange[i - 1] * 1.05);
    // no negative intensities, and the beads stay where they were
    Index argmax = 0;
    for (Index i = 0; i < estimate.size(); ++i) {
        CHECK(estimate.data()[i] >= 0.0f);
        if (estimate.data()[i] > estimate.data()[argmax]) argmax = i;
    }
    CHECK(argmax == (4 * 24 + 8) * 24 + 8);

    SECTION("total variation keeps the result close while smoothing") {
        Buffer<float> tv = blurred.clone();
        DeconvolutionOptions o = opt;
        o.tvLambda = 0.002;
        const DeconvolutionResult rt = richardsonLucy(tv.view(), psf.view(), o);
        CHECK(rt.iterations == 30);
        CHECK(rmse(tv, truth) < 0.7 * errBlurred);
    }
    SECTION("early stopping through the callback and the threshold") {
        Buffer<float> a = blurred.clone();
        DeconvolutionOptions o = opt;
        int calls = 0;
        o.onIteration = [&](int iteration, double) { ++calls; return iteration < 7; };
        const DeconvolutionResult ra = richardsonLucy(a.view(), psf.view(), o);
        CHECK(ra.stoppedEarly);
        CHECK(ra.iterations == 7);
        CHECK(calls == 7);

        Buffer<float> b = blurred.clone();
        DeconvolutionOptions p = opt;
        p.stopRelativeChange = r.relativeChange[9];   // reached after ten iterations
        const DeconvolutionResult rb = richardsonLucy(b.view(), psf.view(), p);
        CHECK(rb.stoppedEarly);
        CHECK(rb.iterations <= 11);
        CHECK(rb.iterations >= 2);
    }
    SECTION("zero iterations leave the input untouched") {
        Buffer<float> a = blurred.clone();
        DeconvolutionOptions o;
        o.iterations = 0;
        const DeconvolutionResult ra = richardsonLucy(a.view(), psf.view(), o);
        CHECK(ra.iterations == 0);
        for (Index i = 0; i < a.size(); ++i) CHECK_THAT(a.data()[i], WithinAbs(blurred.data()[i], 1e-4));
    }
}

TEST_CASE("Richardson-Lucy honours a cancel callback without disturbing the result",
          "[deconvolution][cancel]") {
    const Buffer<float> truth = beads(9, 24, 24);
    const Buffer<float> psf = gaussianPsf(5, 7, 7, 0.2, 0.1, 1.2, 520.0, 1.33);
    const Buffer<float> blurred = blurDirect(truth, psf);

    DeconvolutionOptions opt;
    opt.iterations = 12;
    Buffer<float> reference = blurred.clone();
    richardsonLucy(reference.view(), psf.view(), opt);

    SECTION("a predicate that never fires leaves every value identical") {
        int polls = 0;
        DeconvolutionOptions o = opt;
        o.cancelled = [&polls] { ++polls; return false; };
        Buffer<float> estimate = blurred.clone();
        const DeconvolutionResult r = richardsonLucy(estimate.view(), psf.view(), o);
        CHECK(r.iterations == 12);
        CHECK(polls >= 12);
        Index differing = 0;
        for (Index i = 0; i < estimate.size(); ++i)
            if (estimate.data()[i] != reference.data()[i]) ++differing;
        INFO(differing << " of " << estimate.size() << " values differ");
        CHECK(differing == 0);
    }

    SECTION("a predicate that fires throws and leaves the caller's view untouched") {
        DeconvolutionOptions o = opt;
        int polls = 0;
        o.cancelled = [&polls] { return ++polls > 4; };
        Buffer<float> estimate = blurred.clone();
        CHECK_THROWS_WITH(richardsonLucy(estimate.view(), psf.view(), o), Catch::Matchers::Equals("cancelled"));
        // it stopped inside the second iteration, far short of 12
        CHECK(polls == 5);
        Index changed = 0;
        for (Index i = 0; i < estimate.size(); ++i)
            if (estimate.data()[i] != blurred.data()[i]) ++changed;
        CHECK(changed == 0);
    }
}

TEST_CASE("Richardson-Lucy on a single plane uses the PSF's central plane", "[deconvolution]") {
    const Buffer<float> truth3 = beads(1, 32, 32);
    const Buffer<float> psf = gaussianPsf(5, 7, 7, 0.2, 0.1, 1.2, 520.0, 1.33);
    // blur with the central plane only, which is what a 2D deconvolution sees
    Buffer<float> psf2(Shape{1, 7, 7});
    double sum = 0.0;
    for (Index i = 0; i < 49; ++i) sum += psf.data()[2 * 49 + i];
    for (Index i = 0; i < 49; ++i) psf2.data()[i] = static_cast<float>(psf.data()[2 * 49 + i] / sum);
    const Buffer<float> blurred = blurDirect(truth3, psf2);

    Buffer<float> image(Shape{32, 32});   // a rank-2 view
    std::copy(blurred.data(), blurred.data() + blurred.size(), image.data());
    Buffer<float> truth(Shape{32, 32});
    std::copy(truth3.data(), truth3.data() + truth3.size(), truth.data());

    DeconvolutionOptions opt;
    opt.iterations = 25;
    const DeconvolutionResult r = richardsonLucy(image.view(), psf.view(), opt);   // 3D PSF, central plane used
    CHECK(r.iterations == 25);
    Buffer<float> blurred2(Shape{32, 32});
    std::copy(blurred.data(), blurred.data() + blurred.size(), blurred2.data());
    CHECK(rmse(image, truth) < 0.8 * rmse(blurred2, truth));
    Buffer<float> image3(Shape{1, 32, 32});
    std::copy(image.data(), image.data() + image.size(), image3.data());
    CHECK(rmse(blurDirect(image3, psf2), blurred) < 0.3 * rmse(blurDirect(blurred, psf2), blurred));

    SECTION("a PSF larger than the image is cropped, not rejected") {
        Buffer<float> big = blurred2.clone();
        const Buffer<float> wide = gaussianPsf(1, 41, 41, 0.2, 0.1, 1.2, 520.0, 1.33);
        DeconvolutionOptions o;
        o.iterations = 3;
        CHECK_NOTHROW(richardsonLucy(big.view(), wide.view(), o));
    }
    SECTION("a PSF without positive energy is rejected") {
        Buffer<float> bad(Shape{3, 3});
        std::fill(bad.data(), bad.data() + bad.size(), 0.0f);
        Buffer<float> img = blurred2.clone();
        CHECK_THROWS_AS(richardsonLucy(img.view(), bad.view()), std::invalid_argument);
    }
}
