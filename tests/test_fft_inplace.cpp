// In-place execution of sirius::FFT on the CPU backend. FFTW plans are either
// in-place or out-of-place and may only be executed the way they were
// planned, so aliased in/out must select a dedicated in-place plan; these
// cases would silently corrupt data (or trip FFTW's checks) otherwise.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <complex>
#include <random>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/fft.hpp"

using namespace sirius;
using Cplx = std::complex<double>;

namespace {
    struct Case {
        std::vector<int> dims;
        int howmany;
    };

    Buffer<Cplx> randomBuffer(Shape shape, unsigned seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        Buffer<Cplx> b(shape);
        for (Index i = 0; i < b.size(); ++i) b.data()[i] = Cplx(d(gen), d(gen));
        return b;
    }

    double maxAbsDiff(const Buffer<Cplx>& a, const Buffer<Cplx>& b) {
        double m = 0.0;
        for (Index i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a.data()[i] - b.data()[i]));
        return m;
    }
} // namespace

TEST_CASE("FFT in-place execution matches out-of-place on the CPU", "[fft][inplace]") {
    // Non-power-of-two and batched sizes make FFTW pick the multi-step
    // algorithms where in-place and out-of-place plans differ the most.
    const Case c = GENERATE(Case{{64}, 1}, Case{{15, 17}, 1}, Case{{4, 6, 10}, 1}, Case{{9, 12}, 3});
    INFO("rank " << c.dims.size() << " howmany " << c.howmany);
    const auto rigor = GENERATE(PlanRigor::Estimate, PlanRigor::Measure);
    FFT fft(c.dims, c.howmany, rigor);

    const Buffer<Cplx> in = randomBuffer(fft.shape(), 11);
    Buffer<Cplx> reference(fft.shape());
    fft.fft(in, reference);

    Buffer<Cplx> work = in.clone();
    fft.fft(work, work);                            // in == out
    REQUIRE(maxAbsDiff(work, reference) < 1e-9);

    fft.ifft(work, work, /*normalize=*/true);       // and back, in place again
    REQUIRE(maxAbsDiff(work, in) < 1e-10);

    SECTION("raw-pointer interface aliases the same way") {
        Buffer<Cplx> raw = in.clone();
        fft.fft(raw.data(), raw.data());
        REQUIRE(maxAbsDiff(raw, reference) < 1e-9);
    }
    SECTION("repeated in-place calls reuse the lazily created plan") {
        for (int i = 0; i < 3; ++i) {
            Buffer<Cplx> again = in.clone();
            fft.fft(again, again);
            REQUIRE(maxAbsDiff(again, reference) < 1e-9);
        }
    }
}
