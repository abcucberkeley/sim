#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <Eigen/Core>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include "sirius/fft.hpp"

using namespace sirius;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

namespace {
    constexpr double pi = 3.14159265358979323846;

    // Tolerance for double-precision FFT comparisons.
    // FFTW is accurate to ~machine epsilon * log2(N), this is comfortably above that.
    constexpr double kTol = 1e-10;

    // Build a complex sinusoid: x[k] = exp(2*pi*i * freq * k / n)
    // Its FFT has a single unit spike at bin `freq` and zeros elsewhere.
    Eigen::VectorXcd make_sinusoid(Eigen::Index n, int freq) {
        Eigen::VectorXcd v(n);
        for (Eigen::Index k = 0; k < n; ++k)
            v[k] = std::exp(std::complex<double>(0.0, 2.0 * pi * freq * k / static_cast<double>(n)));
        return v;
    }

    // Normalise an inverse FFT result (FFTW unnormalised convention: IFFT(FFT(x)) = N*x)
    Eigen::VectorXcd normalise(const Eigen::VectorXcd& v) {
        return v / static_cast<double>(v.size());
    }

    // Max absolute error between two vectors
    double max_abs_error(const Eigen::VectorXcd& a, const Eigen::VectorXcd& b) {
        return (a - b).cwiseAbs().maxCoeff();
    }
}

// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------

TEST_CASE("FFT1D construction", "[fft1d]") {
    SECTION("Valid sizes construct without throwing") {
        auto n = GENERATE(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024);
        REQUIRE_NOTHROW(FFT1D(n));
    }

    SECTION("All PlanRigor values construct successfully") {
        auto rigor = GENERATE(
            PlanRigor::Estimate,
            PlanRigor::Measure,
            PlanRigor::Patient,
            PlanRigor::Exhaustive
        );
        REQUIRE_NOTHROW(FFT1D(64, rigor));
    }

    SECTION("Non-power-of-two sizes are valid for FFTW") {
        // FFTW handles arbitrary sizes, not just powers of two
        auto n = GENERATE(3, 5, 7, 100, 127, 1000);
        REQUIRE_NOTHROW(FFT1D(n));
    }

    SECTION("Size zero throws") {
        REQUIRE_THROWS_AS(FFT1D(0), std::invalid_argument);
    }

    SECTION("Negative size throws") {
        REQUIRE_THROWS_AS(FFT1D(-1), std::invalid_argument);
    }
}

// -----------------------------------------------------------------------
// Move semantics
// -----------------------------------------------------------------------

TEST_CASE("FFT1D move semantics", "[fft1d]") {
    SECTION("Move construction leaves object in valid state") {
        FFT1D a(64);
        FFT1D b(std::move(a));

        // b should work correctly
        Eigen::VectorXcd in  = make_sinusoid(64, 3);
        Eigen::VectorXcd out(64);
        REQUIRE_NOTHROW(b.forward(in, out));
    }

    SECTION("Move assignment leaves object in valid state") {
        FFT1D a(64);
        FFT1D b(32);
        b = std::move(a);

        Eigen::VectorXcd in  = make_sinusoid(64, 3);
        Eigen::VectorXcd out(64);
        REQUIRE_NOTHROW(b.forward(in, out));
    }
}

// -----------------------------------------------------------------------
// Buffer size validation
// -----------------------------------------------------------------------

TEST_CASE("FFT1D rejects mismatched buffer sizes", "[fft1d]") {
    FFT1D fft(64);

    SECTION("forward: input too small") {
        Eigen::VectorXcd in(32), out(64);
        REQUIRE_THROWS_AS(fft.forward(in, out), std::invalid_argument);
    }

    SECTION("forward: output too small") {
        Eigen::VectorXcd in(64), out(32);
        REQUIRE_THROWS_AS(fft.forward(in, out), std::invalid_argument);
    }

    SECTION("inverse: input too large") {
        Eigen::VectorXcd in(128), out(64);
        REQUIRE_THROWS_AS(fft.inverse(in, out), std::invalid_argument);
    }
}

// -----------------------------------------------------------------------
// Linearity (superposition)
// -----------------------------------------------------------------------

TEST_CASE("FFT1D linearity - FFT(a*x + b*y) == a*FFT(x) + b*FFT(y)", "[fft1d][correctness]") {
    const Eigen::Index n = 128;
    FFT1D fft(n);

    Eigen::VectorXcd x = Eigen::VectorXcd::Random(n);
    Eigen::VectorXcd y = Eigen::VectorXcd::Random(n);
    const std::complex<double> a(2.0, -1.0);
    const std::complex<double> b(0.5,  3.0);

    Eigen::VectorXcd combined(n), out_combined(n);
    combined = a * x + b * y;
    fft.forward(combined, out_combined);

    Eigen::VectorXcd out_x(n), out_y(n);
    fft.forward(x, out_x);
    fft.forward(y, out_y);
    Eigen::VectorXcd expected = a * out_x + b * out_y;

    REQUIRE(max_abs_error(out_combined, expected) < kTol);
}

// -----------------------------------------------------------------------
// Impulse response - delta function
// -----------------------------------------------------------------------

TEST_CASE("FFT1D of delta function is flat spectrum", "[fft1d][correctness]") {
    // x[0]=1, x[k]=0 for k>0  =>  X[f] = 1 for all f
    const Eigen::Index n = 256;
    FFT1D fft(n);

    Eigen::VectorXcd in = Eigen::VectorXcd::Zero(n);
    in[0] = 1.0;

    Eigen::VectorXcd out(n);
    fft.forward(in, out);

    for (Eigen::Index f = 0; f < n; ++f)
        REQUIRE_THAT(std::abs(out[f]), Catch::Matchers::WithinAbs(1.0, kTol));
}

// -----------------------------------------------------------------------
// Single-frequency sinusoid - spike in spectrum
// -----------------------------------------------------------------------

TEST_CASE("FFT1D of sinusoid produces spike at correct bin", "[fft1d][correctness]") {
    const Eigen::Index n = 128;
    FFT1D fft(n);

    auto freq = GENERATE(0, 1, 5, 10, 63);
    INFO("Frequency bin: " << freq);

    Eigen::VectorXcd in  = make_sinusoid(n, freq);
    Eigen::VectorXcd out(n);
    fft.forward(in, out);

    for (Eigen::Index f = 0; f < n; ++f) {
        double expected_mag = (f == freq) ? static_cast<double>(n) : 0.0;
        REQUIRE_THAT(std::abs(out[f]), Catch::Matchers::WithinAbs(expected_mag, kTol * n));
    }
}

// -----------------------------------------------------------------------
// Forward-inverse round-trip
// -----------------------------------------------------------------------

TEST_CASE("FFT1D forward-inverse round-trip recovers original signal", "[fft1d][correctness]") {
    auto n = GENERATE(16, 64, 128, 256, 1024);
    INFO("N = " << n);

    FFT1D fft(n);
    Eigen::VectorXcd original = Eigen::VectorXcd::Random(n);

    Eigen::VectorXcd freq_domain(n), recovered(n);
    fft.forward(original, freq_domain);
    fft.inverse(freq_domain, recovered);

    // FFTW uses unnormalised convention: IFFT(FFT(x)) = N * x
    REQUIRE(max_abs_error(normalise(recovered), original) < kTol);
}

// -----------------------------------------------------------------------
// Parseval's theorem - energy conservation
// -----------------------------------------------------------------------

TEST_CASE("FFT1D satisfies Parseval's theorem", "[fft1d][correctness]") {
    // sum|x[k]|^2 == (1/N) * sum|X[f]|^2
    const Eigen::Index n = 256;
    FFT1D fft(n);

    Eigen::VectorXcd in = Eigen::VectorXcd::Random(n);
    Eigen::VectorXcd out(n);
    fft.forward(in, out);

    double energy_time = in.squaredNorm();
    double energy_freq = out.squaredNorm() / static_cast<double>(n);

    REQUIRE_THAT(energy_freq, Catch::Matchers::WithinRel(energy_time, 1e-10));
}

// -----------------------------------------------------------------------
// Shift theorem - time shift = linear phase in frequency
// -----------------------------------------------------------------------

TEST_CASE("FFT1D shift theorem - time shift is linear phase in frequency", "[fft1d][correctness]") {
    // If y[k] = x[k - d] (circular), then Y[f] = X[f] * exp(-2*pi*i*f*d/N)
    const Eigen::Index n = 64;
    const int d = 5; // shift amount
    FFT1D fft(n);

    Eigen::VectorXcd x = Eigen::VectorXcd::Random(n);

    // circular shift by d
    Eigen::VectorXcd y(n);
    for (Eigen::Index k = 0; k < n; ++k)
        y[k] = x[(k - d + n) % n];

    Eigen::VectorXcd X(n), Y(n);
    fft.forward(x, X);
    fft.forward(y, Y);

    // verify Y[f] == X[f] * exp(-2*pi*i*f*d/N) for each bin f
    for (Eigen::Index f = 0; f < n; ++f) {
        std::complex<double> phase = std::exp(std::complex<double>(0.0, -2.0 * pi * f * d / static_cast<double>(n)));
        std::complex<double> expected = X[f] * phase;
        REQUIRE_THAT(std::abs(Y[f] - expected), Catch::Matchers::WithinAbs(0.0, kTol * n));
    }
}

// -----------------------------------------------------------------------
// Conjugate symmetry - real-valued input
// -----------------------------------------------------------------------

TEST_CASE("FFT1D conjugate symmetry for real-valued input", "[fft1d][correctness]") {
    // If x is real, then X[N-f] == conj(X[f])
    const Eigen::Index n = 128;
    FFT1D fft(n);

    // real-valued input stored as complex with zero imaginary part
    Eigen::VectorXcd in(n);
    for (Eigen::Index k = 0; k < n; ++k)
        in[k] = std::complex<double>(std::cos(2.0 * pi * 3 * k / n) + 0.5, 0.0);

    Eigen::VectorXcd out(n);
    fft.forward(in, out);

    for (Eigen::Index f = 1; f < n / 2; ++f) {
        std::complex<double> diff = out[n - f] - std::conj(out[f]);
        REQUIRE_THAT(std::abs(diff), Catch::Matchers::WithinAbs(0.0, kTol * n));
    }
}

// -----------------------------------------------------------------------
// DC component
// -----------------------------------------------------------------------

TEST_CASE("FFT1D DC bin equals sum of input", "[fft1d][correctness]") {
    // X[0] = sum(x[k])
    const Eigen::Index n = 64;
    FFT1D fft(n);

    Eigen::VectorXcd in = Eigen::VectorXcd::Random(n);
    Eigen::VectorXcd out(n);
    fft.forward(in, out);

    std::complex<double> dc_expected = in.sum();
    REQUIRE_THAT(std::abs(out[0] - dc_expected), Catch::Matchers::WithinAbs(0.0, kTol * n));
}

// -----------------------------------------------------------------------
// Misaligned input via Eigen segment (exercises execute_safe fallback)
// -----------------------------------------------------------------------

TEST_CASE("FFT1D handles misaligned input via segment", "[fft1d][alignment]") {
    // vec.segment(1, n) points to an offset address - likely misaligned for SIMD.
    // execute_safe should detect this and use the copy fallback.
    const Eigen::Index n = 64;
    FFT1D fft(n);

    // allocate n+1, take a segment starting at index 1 to force offset address
    Eigen::VectorXcd padded = Eigen::VectorXcd::Random(n + 1);
    Eigen::VectorXcd in_contiguous = padded.segment(1, n); // copy into contiguous for reference
    auto in_segment = padded.segment(1, n);                // non-owning, potentially misaligned

    Eigen::VectorXcd out_ref(n), out_seg(n);
    fft.forward(in_contiguous, out_ref);
    fft.forward(in_segment,    out_seg);    // goes through execute_safe

    REQUIRE(max_abs_error(out_ref, out_seg) < kTol);
}

// -----------------------------------------------------------------------
// Multiple independent FFT1D objects at the same size
// -----------------------------------------------------------------------

TEST_CASE("Multiple FFT1D objects at same size produce identical results", "[fft1d]") {
    const Eigen::Index n = 128;
    FFT1D fft_a(n);
    FFT1D fft_b(n, PlanRigor::Estimate);

    Eigen::VectorXcd in = Eigen::VectorXcd::Random(n);
    Eigen::VectorXcd out_a(n), out_b(n);

    fft_a.forward(in, out_a);
    fft_b.forward(in, out_b);

    REQUIRE(max_abs_error(out_a, out_b) < kTol);
}

// -----------------------------------------------------------------------
// Reuse - same plan, multiple executions
// -----------------------------------------------------------------------

TEST_CASE("FFT1D plan can be reused across multiple calls", "[fft1d]") {
    const Eigen::Index n = 64;
    FFT1D fft(n);

    for (int iter = 0; iter < 5; ++iter) {
        Eigen::VectorXcd in  = make_sinusoid(n, iter);
        Eigen::VectorXcd out(n);
        fft.forward(in, out);

        // spike should be at bin `iter`
        REQUIRE_THAT(std::abs(out[iter]),
            Catch::Matchers::WithinAbs(static_cast<double>(n), kTol * n));
    }
}

// -----------------------------------------------------------------------
// Normalization flag
// -----------------------------------------------------------------------

TEST_CASE("FFT1D normalize=true: IFFT(FFT(x)) == x without manual scaling", "[fft1d][normalize]") {
    auto n = GENERATE(16, 64, 256);
    INFO("N = " << n);

    FFT1D fft(n, PlanRigor::Measure, true);
    Eigen::VectorXcd original = Eigen::VectorXcd::Random(n);

    Eigen::VectorXcd freq_domain(n), recovered(n);
    fft.forward(original, freq_domain);
    fft.inverse(freq_domain, recovered);

    REQUIRE(max_abs_error(recovered, original) < kTol);
}

TEST_CASE("FFT1D normalize=false (default): inverse is unnormalized", "[fft1d][normalize]") {
    // With normalize=false, IFFT(FFT(x)) = N*x — the raw FFTW convention.
    const Eigen::Index n = 64;
    FFT1D fft(n); // normalize defaults to false

    Eigen::VectorXcd original = Eigen::VectorXcd::Random(n);
    Eigen::VectorXcd freq_domain(n), recovered(n);
    fft.forward(original, freq_domain);
    fft.inverse(freq_domain, recovered);

    // recovered should equal N * original, not original
    REQUIRE(max_abs_error(recovered, static_cast<double>(n) * original) < kTol);
}

TEST_CASE("FFT1D normalize flag does not affect forward transform", "[fft1d][normalize]") {
    // forward output should be identical regardless of the normalize flag
    const Eigen::Index n = 64;
    FFT1D fft_raw(n, PlanRigor::Measure, false);
    FFT1D fft_norm(n, PlanRigor::Measure, true);

    Eigen::VectorXcd in = Eigen::VectorXcd::Random(n);
    Eigen::VectorXcd out_raw(n), out_norm(n);
    fft_raw.forward(in, out_raw);
    fft_norm.forward(in, out_norm);

    REQUIRE(max_abs_error(out_raw, out_norm) < kTol);
}

// -----------------------------------------------------------------------
// Wisdom save / load
// -----------------------------------------------------------------------

namespace {
    // Use a fixed temp path; clean up in the test.
    const std::string kWisdomPath =
        (std::filesystem::temp_directory_path() / "sirius_test_wisdom.fftw").string();
}

TEST_CASE("FFT1D saveWisdom writes a file and loadWisdom reads it back", "[fft1d][wisdom]") {
    std::remove(kWisdomPath.c_str());

    // Plan and save
    FFT1D fft(128);
    REQUIRE_NOTHROW(FFT1D::saveWisdom(kWisdomPath.c_str()));

    // File must exist after saving
    FILE* f = std::fopen(kWisdomPath.c_str(), "r");
    REQUIRE(f != nullptr);
    if (f) std::fclose(f);

    // Load and verify a subsequent plan still produces correct results
    REQUIRE_NOTHROW(FFT1D::loadWisdom(kWisdomPath.c_str()));

    FFT1D fft2(128);
    Eigen::VectorXcd in = Eigen::VectorXcd::Zero(128);
    in[0] = 1.0;
    Eigen::VectorXcd out(128);
    fft2.forward(in, out);

    // delta input -> flat spectrum with magnitude 1
    for (Eigen::Index f = 0; f < 128; ++f)
        REQUIRE_THAT(std::abs(out[f]), Catch::Matchers::WithinAbs(1.0, kTol));

    std::remove(kWisdomPath.c_str());
}

TEST_CASE("FFT1D loadWisdom on missing file does not throw", "[fft1d][wisdom]") {
    const std::string missing =
        (std::filesystem::temp_directory_path() / "sirius_nonexistent_wisdom.fftw").string();
    REQUIRE_NOTHROW(FFT1D::loadWisdom(missing.c_str()));
}

TEST_CASE("FFT1D saveWisdom to invalid path throws", "[fft1d][wisdom]") {
    const std::string bad =
        (std::filesystem::temp_directory_path() / "sirius_nonexistent_subdir" / "wisdom.fftw").string();
    REQUIRE_THROWS_AS(
        FFT1D::saveWisdom(bad.c_str()),
        std::runtime_error
    );
}
