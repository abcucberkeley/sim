#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <Eigen/Core>
#include <cmath>
#include <cstring>
#include <cstdio>
#include "sirius/fft.hpp"

using namespace sirius;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

namespace {
    constexpr double pi = 3.14159265358979323846;
    constexpr double kTol = 1e-10;

    // Fill with a 3D complex sinusoid: x[d,r,c] = exp(2πi*(f1*d/D + f2*r/R + f3*c/C))
    // Its 3D FFT has a single spike of magnitude D*R*C at bin (f1, f2, f3).
    void fill_sinusoid(FFTWBuffer3D& buf, int f1, int f2, int f3) {
        auto D = buf.depth(), R = buf.rows(), C = buf.cols();
        for (Eigen::Index d = 0; d < D; ++d)
            for (Eigen::Index r = 0; r < R; ++r)
                for (Eigen::Index c = 0; c < C; ++c)
                    buf(d, r, c) = std::exp(std::complex<double>(0.0,
                        2.0 * pi * (f1 * d / static_cast<double>(D)
                                  + f2 * r / static_cast<double>(R)
                                  + f3 * c / static_cast<double>(C))));
    }

    // Deterministic complex fill — avoids rand() while still exercising non-trivial values.
    void fill_deterministic(FFTWBuffer3D& buf) {
        auto D = buf.depth(), R = buf.rows(), C = buf.cols();
        for (Eigen::Index d = 0; d < D; ++d)
            for (Eigen::Index r = 0; r < R; ++r)
                for (Eigen::Index c = 0; c < C; ++c)
                    buf(d, r, c) = std::complex<double>(
                        std::cos(d * 1.3 + r * 0.7 + c * 2.1),
                        std::sin(d * 0.9 + r * 1.5 + c * 0.3));
    }

    void fill_zero(FFTWBuffer3D& buf) {
        std::memset(buf.data(), 0, static_cast<size_t>(buf.size()) * sizeof(fftw_complex));
    }

    double max_abs_error(const FFTWBuffer3D& a, const FFTWBuffer3D& b) {
        auto* pa = reinterpret_cast<const std::complex<double>*>(a.data());
        auto* pb = reinterpret_cast<const std::complex<double>*>(b.data());
        double err = 0.0;
        for (Eigen::Index i = 0; i < a.size(); ++i)
            err = std::max(err, std::abs(pa[i] - pb[i]));
        return err;
    }

    double sum_sq_norm(const FFTWBuffer3D& buf) {
        auto* p = reinterpret_cast<const std::complex<double>*>(buf.data());
        double s = 0.0;
        for (Eigen::Index i = 0; i < buf.size(); ++i)
            s += std::norm(p[i]); // std::norm returns |z|²
        return s;
    }
}

// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------

TEST_CASE("FFT3D construction", "[fft3d]") {
    SECTION("Valid sizes construct without throwing") {
        REQUIRE_NOTHROW(FFT3D(4, 4, 4));
        REQUIRE_NOTHROW(FFT3D(2, 8, 16));
        REQUIRE_NOTHROW(FFT3D(1, 1, 1));
    }

    SECTION("Non-cubic dimensions are valid") {
        REQUIRE_NOTHROW(FFT3D(3, 5, 7));
        REQUIRE_NOTHROW(FFT3D(8, 16, 32));
    }

    SECTION("Non-power-of-two sizes are valid for FFTW") {
        REQUIRE_NOTHROW(FFT3D(6, 10, 15));
    }

    SECTION("All PlanRigor values construct successfully") {
        auto rigor = GENERATE(
            PlanRigor::Estimate,
            PlanRigor::Measure,
            PlanRigor::Patient,
            PlanRigor::Exhaustive
        );
        REQUIRE_NOTHROW(FFT3D(4, 4, 4, rigor));
    }

    SECTION("Zero depth throws") {
        REQUIRE_THROWS_AS(FFT3D(0, 4, 4), std::invalid_argument);
    }

    SECTION("Zero rows throws") {
        REQUIRE_THROWS_AS(FFT3D(4, 0, 4), std::invalid_argument);
    }

    SECTION("Zero cols throws") {
        REQUIRE_THROWS_AS(FFT3D(4, 4, 0), std::invalid_argument);
    }

    SECTION("Negative dimensions throw") {
        REQUIRE_THROWS_AS(FFT3D(-1, 4, 4), std::invalid_argument);
        REQUIRE_THROWS_AS(FFT3D(4, -1, 4), std::invalid_argument);
        REQUIRE_THROWS_AS(FFT3D(4, 4, -1), std::invalid_argument);
    }
}

// -----------------------------------------------------------------------
// Move semantics
// -----------------------------------------------------------------------

TEST_CASE("FFT3D move semantics", "[fft3d]") {
    SECTION("Move constructor leaves object in valid state") {
        FFT3D a(4, 8, 8);
        FFT3D b(std::move(a));

        FFTWBuffer3D in(4, 8, 8), out(4, 8, 8);
        fill_zero(in);
        in(0, 0, 0) = 1.0;
        REQUIRE_NOTHROW(b.forward(in, out));
    }

    SECTION("Move assignment leaves object in valid state") {
        FFT3D a(4, 8, 8);
        FFT3D b(2, 2, 2);
        b = std::move(a);

        FFTWBuffer3D in(4, 8, 8), out(4, 8, 8);
        fill_zero(in);
        in(0, 0, 0) = 1.0;
        REQUIRE_NOTHROW(b.forward(in, out));
    }
}

// -----------------------------------------------------------------------
// Buffer dimension validation
// -----------------------------------------------------------------------

TEST_CASE("FFT3D rejects mismatched buffer dimensions", "[fft3d]") {
    FFT3D fft(4, 8, 16);

    SECTION("forward: wrong depth") {
        FFTWBuffer3D in(2, 8, 16), out(4, 8, 16);
        REQUIRE_THROWS_AS(fft.forward(in, out), std::invalid_argument);
    }

    SECTION("forward: wrong rows") {
        FFTWBuffer3D in(4, 4, 16), out(4, 8, 16);
        REQUIRE_THROWS_AS(fft.forward(in, out), std::invalid_argument);
    }

    SECTION("forward: wrong cols") {
        FFTWBuffer3D in(4, 8, 8), out(4, 8, 16);
        REQUIRE_THROWS_AS(fft.forward(in, out), std::invalid_argument);
    }

    SECTION("inverse: wrong output dimensions") {
        FFTWBuffer3D in(4, 8, 16), out(4, 8, 8);
        REQUIRE_THROWS_AS(fft.inverse(in, out), std::invalid_argument);
    }
}

// -----------------------------------------------------------------------
// 3D impulse response
// -----------------------------------------------------------------------

TEST_CASE("FFT3D of 3D delta is flat spectrum", "[fft3d][correctness]") {
    // x[0,0,0]=1, rest=0 → X[f1,f2,f3]=1 for all (f1,f2,f3)
    const Eigen::Index D = 4, R = 8, C = 8;
    FFT3D fft(D, R, C);

    FFTWBuffer3D in(D, R, C), out(D, R, C);
    fill_zero(in);
    in(0, 0, 0) = 1.0;
    fft.forward(in, out);

    for (Eigen::Index d = 0; d < D; ++d)
        for (Eigen::Index r = 0; r < R; ++r)
            for (Eigen::Index c = 0; c < C; ++c)
                REQUIRE_THAT(std::abs(out(d, r, c)),
                    Catch::Matchers::WithinAbs(1.0, kTol));
}

// -----------------------------------------------------------------------
// 3D sinusoid — single spike in spectrum
// -----------------------------------------------------------------------

TEST_CASE("FFT3D of 3D sinusoid produces spike at correct bin", "[fft3d][correctness]") {
    const Eigen::Index D = 8, R = 8, C = 8;
    FFT3D fft(D, R, C);
    const double N = static_cast<double>(D * R * C);

    auto f1 = GENERATE(0, 1, 3);
    auto f2 = GENERATE(0, 2, 7);
    auto f3 = GENERATE(0, 4, 7);
    INFO("f1=" << f1 << " f2=" << f2 << " f3=" << f3);

    FFTWBuffer3D in(D, R, C), out(D, R, C);
    fill_sinusoid(in, f1, f2, f3);
    fft.forward(in, out);

    for (Eigen::Index d = 0; d < D; ++d)
        for (Eigen::Index r = 0; r < R; ++r)
            for (Eigen::Index c = 0; c < C; ++c) {
                double expected = (d == f1 && r == f2 && c == f3) ? N : 0.0;
                REQUIRE_THAT(std::abs(out(d, r, c)),
                    Catch::Matchers::WithinAbs(expected, kTol * N));
            }
}

// -----------------------------------------------------------------------
// Linearity
// -----------------------------------------------------------------------

TEST_CASE("FFT3D linearity - FFT(a*X + b*Y) == a*FFT(X) + b*FFT(Y)", "[fft3d][correctness]") {
    const Eigen::Index D = 4, R = 6, C = 8;
    FFT3D fft(D, R, C);

    FFTWBuffer3D X(D, R, C), Y(D, R, C);
    fill_deterministic(X);
    // Y is a shifted version of X for variety
    for (Eigen::Index d = 0; d < D; ++d)
        for (Eigen::Index r = 0; r < R; ++r)
            for (Eigen::Index c = 0; c < C; ++c)
                Y(d, r, c) = std::complex<double>(
                    std::sin(d * 2.1 + r * 0.4 + c * 1.7),
                    std::cos(d * 0.5 + r * 2.3 + c * 0.8));

    const std::complex<double> a(2.0, -1.0), b(0.5, 3.0);

    FFTWBuffer3D combined(D, R, C);
    auto* px = reinterpret_cast<const std::complex<double>*>(X.data());
    auto* py = reinterpret_cast<const std::complex<double>*>(Y.data());
    auto* pc = reinterpret_cast<std::complex<double>*>(combined.data());
    for (Eigen::Index i = 0; i < D * R * C; ++i)
        pc[i] = a * px[i] + b * py[i];

    FFTWBuffer3D out_combined(D, R, C), out_X(D, R, C), out_Y(D, R, C);
    fft.forward(combined, out_combined);
    fft.forward(X, out_X);
    fft.forward(Y, out_Y);

    auto* pox = reinterpret_cast<const std::complex<double>*>(out_X.data());
    auto* poy = reinterpret_cast<const std::complex<double>*>(out_Y.data());
    auto* poc = reinterpret_cast<const std::complex<double>*>(out_combined.data());
    double err = 0.0;
    for (Eigen::Index i = 0; i < D * R * C; ++i)
        err = std::max(err, std::abs(poc[i] - (a * pox[i] + b * poy[i])));

    REQUIRE(err < kTol);
}

// -----------------------------------------------------------------------
// Forward-inverse round-trip
// -----------------------------------------------------------------------

TEST_CASE("FFT3D forward-inverse round-trip recovers original signal", "[fft3d][correctness]") {
    auto D = GENERATE(4, 8);
    auto R = GENERATE(4, 6);
    auto C = GENERATE(4, 8);
    INFO("D=" << D << " R=" << R << " C=" << C);

    SECTION("manual normalization") {
        FFT3D fft(D, R, C);
        FFTWBuffer3D original(D, R, C), freq(D, R, C), recovered(D, R, C);
        fill_deterministic(original);

        fft.forward(original, freq);
        fft.inverse(freq, recovered);

        double N = static_cast<double>(D * R * C);
        auto* po = reinterpret_cast<const std::complex<double>*>(original.data());
        auto* pr = reinterpret_cast<std::complex<double>*>(recovered.data());
        for (Eigen::Index i = 0; i < D * R * C; ++i)
            pr[i] /= N;

        REQUIRE(max_abs_error(recovered, original) < kTol);
    }

    SECTION("built-in normalization") {
        FFT3D fft(D, R, C, PlanRigor::Measure, true);
        FFTWBuffer3D original(D, R, C), freq(D, R, C), recovered(D, R, C);
        fill_deterministic(original);

        fft.forward(original, freq);
        fft.inverse(freq, recovered);

        REQUIRE(max_abs_error(recovered, original) < kTol);
    }
}

// -----------------------------------------------------------------------
// Parseval's theorem
// -----------------------------------------------------------------------

TEST_CASE("FFT3D satisfies Parseval's theorem", "[fft3d][correctness]") {
    // sum|x[d,r,c]|² == (1/N) * sum|X[f1,f2,f3]|²  where N = D*R*C
    const Eigen::Index D = 4, R = 6, C = 8;
    FFT3D fft(D, R, C);

    FFTWBuffer3D in(D, R, C), out(D, R, C);
    fill_deterministic(in);
    fft.forward(in, out);

    double energy_space = sum_sq_norm(in);
    double energy_freq  = sum_sq_norm(out) / static_cast<double>(D * R * C);

    REQUIRE_THAT(energy_freq, Catch::Matchers::WithinRel(energy_space, 1e-10));
}

// -----------------------------------------------------------------------
// Separability — FFT3D(u⊗v⊗w)[f1,f2,f3] = FFT1D(u)[f1] * FFT1D(v)[f2] * FFT1D(w)[f3]
// -----------------------------------------------------------------------

TEST_CASE("FFT3D is separable along depth, rows, and cols", "[fft3d][correctness]") {
    const Eigen::Index D = 8, R = 8, C = 8;
    FFT1D fft1d_d(D), fft1d_r(R), fft1d_c(C);
    FFT3D fft3d(D, R, C);

    // Build 1D signals
    Eigen::VectorXcd u(D), v(R), w(C);
    for (Eigen::Index i = 0; i < D; ++i) u[i] = std::complex<double>(std::cos(i * 1.1), std::sin(i * 0.7));
    for (Eigen::Index i = 0; i < R; ++i) v[i] = std::complex<double>(std::cos(i * 0.5), std::sin(i * 1.3));
    for (Eigen::Index i = 0; i < C; ++i) w[i] = std::complex<double>(std::cos(i * 2.0), std::sin(i * 0.3));

    // Build outer product X[d,r,c] = u[d] * v[r] * w[c]
    FFTWBuffer3D X(D, R, C);
    for (Eigen::Index d = 0; d < D; ++d)
        for (Eigen::Index r = 0; r < R; ++r)
            for (Eigen::Index c = 0; c < C; ++c)
                X(d, r, c) = u[d] * v[r] * w[c];

    FFTWBuffer3D X_fft(D, R, C);
    fft3d.forward(X, X_fft);

    // Compute 1D FFTs of each axis
    Eigen::VectorXcd U(D), V(R), W(C);
    fft1d_d.forward(u, U);
    fft1d_r.forward(v, V);
    fft1d_c.forward(w, W);

    // Verify X_fft[f1,f2,f3] == U[f1] * V[f2] * W[f3]
    double err = 0.0;
    for (Eigen::Index f1 = 0; f1 < D; ++f1)
        for (Eigen::Index f2 = 0; f2 < R; ++f2)
            for (Eigen::Index f3 = 0; f3 < C; ++f3)
                err = std::max(err, std::abs(X_fft(f1, f2, f3) - U[f1] * V[f2] * W[f3]));

    REQUIRE(err < kTol * D * R * C);
}

// -----------------------------------------------------------------------
// Normalization flag
// -----------------------------------------------------------------------

TEST_CASE("FFT3D normalize=false (default): inverse is unnormalized", "[fft3d][normalize]") {
    const Eigen::Index D = 4, R = 4, C = 4;
    FFT3D fft(D, R, C);

    FFTWBuffer3D original(D, R, C), freq(D, R, C), recovered(D, R, C);
    fill_deterministic(original);
    fft.forward(original, freq);
    fft.inverse(freq, recovered);

    double N = static_cast<double>(D * R * C);
    auto* po = reinterpret_cast<const std::complex<double>*>(original.data());
    auto* pr = reinterpret_cast<const std::complex<double>*>(recovered.data());
    double err = 0.0;
    for (Eigen::Index i = 0; i < D * R * C; ++i)
        err = std::max(err, std::abs(pr[i] - N * po[i]));

    REQUIRE(err < kTol);
}

TEST_CASE("FFT3D normalize flag does not affect forward transform", "[fft3d][normalize]") {
    const Eigen::Index D = 4, R = 4, C = 4;
    FFT3D fft_raw(D, R, C, PlanRigor::Measure, false);
    FFT3D fft_norm(D, R, C, PlanRigor::Measure, true);

    FFTWBuffer3D in(D, R, C), out_raw(D, R, C), out_norm(D, R, C);
    fill_deterministic(in);
    fft_raw.forward(in, out_raw);
    fft_norm.forward(in, out_norm);

    REQUIRE(max_abs_error(out_raw, out_norm) < kTol);
}

// -----------------------------------------------------------------------
// Plan reuse
// -----------------------------------------------------------------------

TEST_CASE("FFT3D plan can be reused across multiple calls", "[fft3d]") {
    const Eigen::Index D = 4, R = 4, C = 4;
    FFT3D fft(D, R, C);

    for (int f1 = 0; f1 < 2; ++f1) {
        for (int f2 = 0; f2 < 2; ++f2) {
            for (int f3 = 0; f3 < 2; ++f3) {
                FFTWBuffer3D in(D, R, C), out(D, R, C);
                fill_sinusoid(in, f1, f2, f3);
                fft.forward(in, out);

                double N = static_cast<double>(D * R * C);
                REQUIRE_THAT(std::abs(out(f1, f2, f3)),
                    Catch::Matchers::WithinAbs(N, kTol * N));
            }
        }
    }
}

// -----------------------------------------------------------------------
// Wisdom
// -----------------------------------------------------------------------

namespace {
    const char* kWisdom3DPath = "/tmp/sirius_test_wisdom_3d.fftw";
}

TEST_CASE("FFT3D saveWisdom and loadWisdom", "[fft3d][wisdom]") {
    std::remove(kWisdom3DPath);

    FFT3D fft(8, 8, 8);
    REQUIRE_NOTHROW(FFT3D::saveWisdom(kWisdom3DPath));

    FILE* f = std::fopen(kWisdom3DPath, "r");
    REQUIRE(f != nullptr);
    if (f) std::fclose(f);

    REQUIRE_NOTHROW(FFT3D::loadWisdom(kWisdom3DPath));

    FFT3D fft2(8, 8, 8);
    FFTWBuffer3D in(8, 8, 8), out(8, 8, 8);
    fill_zero(in);
    in(0, 0, 0) = 1.0;
    fft2.forward(in, out);

    for (Eigen::Index d = 0; d < 8; ++d)
        for (Eigen::Index r = 0; r < 8; ++r)
            for (Eigen::Index c = 0; c < 8; ++c)
                REQUIRE_THAT(std::abs(out(d, r, c)),
                    Catch::Matchers::WithinAbs(1.0, kTol));

    std::remove(kWisdom3DPath);
}

TEST_CASE("FFT3D saveWisdom to invalid path throws", "[fft3d][wisdom]") {
    REQUIRE_THROWS_AS(
        FFT3D::saveWisdom("/nonexistent_dir/wisdom.fftw"),
        std::runtime_error
    );
}
