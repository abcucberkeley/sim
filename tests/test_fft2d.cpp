#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <Eigen/Core>
#include <cmath>
#include <cstdio>
#include "sirius/fft.hpp"

using namespace sirius;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

namespace {
    constexpr double pi = 3.14159265358979323846;
    constexpr double kTol = 1e-10;

    // Build a 2D complex sinusoid: x[r,c] = exp(2*pi*i*(f1*r/R + f2*c/C))
    // Its 2D FFT has a single spike of magnitude R*C at bin (f1, f2).
    RowMatrixXcd make_sinusoid_2d(Eigen::Index rows, Eigen::Index cols, int f1, int f2) {
        RowMatrixXcd m(rows, cols);
        for (Eigen::Index r = 0; r < rows; ++r)
            for (Eigen::Index c = 0; c < cols; ++c)
                m(r, c) = std::exp(std::complex<double>(0.0,
                    2.0 * pi * (f1 * r / static_cast<double>(rows)
                              + f2 * c / static_cast<double>(cols))));
        return m;
    }

    double max_abs_error(const RowMatrixXcd& a, const RowMatrixXcd& b) {
        return (a - b).cwiseAbs().maxCoeff();
    }

    double max_abs_error(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
        return (a - b).cwiseAbs().maxCoeff();
    }
}

// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------

TEST_CASE("FFT2D construction", "[fft2d]") {
    SECTION("Valid sizes construct without throwing") {
        REQUIRE_NOTHROW(FFT2D(4, 4));
        REQUIRE_NOTHROW(FFT2D(8, 16));
        REQUIRE_NOTHROW(FFT2D(1, 1));
    }

    SECTION("Non-square dimensions are valid") {
        auto rows = GENERATE(3, 5, 64, 100);
        auto cols = GENERATE(7, 16, 50, 128);
        REQUIRE_NOTHROW(FFT2D(rows, cols));
    }

    SECTION("Non-power-of-two sizes are valid for FFTW") {
        REQUIRE_NOTHROW(FFT2D(100, 150));
        REQUIRE_NOTHROW(FFT2D(127, 33));
    }

    SECTION("All PlanRigor values construct successfully") {
        auto rigor = GENERATE(
            PlanRigor::Estimate,
            PlanRigor::Measure,
            PlanRigor::Patient,
            PlanRigor::Exhaustive
        );
        REQUIRE_NOTHROW(FFT2D(16, 16, rigor));
    }

    SECTION("Zero rows throws") {
        REQUIRE_THROWS_AS(FFT2D(0, 8), std::invalid_argument);
    }

    SECTION("Zero cols throws") {
        REQUIRE_THROWS_AS(FFT2D(8, 0), std::invalid_argument);
    }

    SECTION("Negative dimensions throw") {
        REQUIRE_THROWS_AS(FFT2D(-1, 8), std::invalid_argument);
        REQUIRE_THROWS_AS(FFT2D(8, -1), std::invalid_argument);
    }
}

// -----------------------------------------------------------------------
// Move semantics
// -----------------------------------------------------------------------

TEST_CASE("FFT2D move semantics", "[fft2d]") {
    SECTION("Move construction leaves object in valid state") {
        FFT2D a(32, 32);
        FFT2D b(std::move(a));

        RowMatrixXcd in  = RowMatrixXcd::Zero(32, 32);
        RowMatrixXcd out(32, 32);
        in(0, 0) = 1.0;
        REQUIRE_NOTHROW(b.forward(in, out));
    }

    SECTION("Move assignment leaves object in valid state") {
        FFT2D a(32, 32);
        FFT2D b(16, 16);
        b = std::move(a);

        RowMatrixXcd in  = RowMatrixXcd::Zero(32, 32);
        RowMatrixXcd out(32, 32);
        in(0, 0) = 1.0;
        REQUIRE_NOTHROW(b.forward(in, out));
    }
}

// -----------------------------------------------------------------------
// Buffer dimension validation
// -----------------------------------------------------------------------

TEST_CASE("FFT2D rejects mismatched buffer dimensions", "[fft2d]") {
    FFT2D fft(16, 32);

    SECTION("forward row-major: wrong rows") {
        RowMatrixXcd in(8, 32), out(16, 32);
        REQUIRE_THROWS_AS(fft.forward(in, out), std::invalid_argument);
    }

    SECTION("forward row-major: wrong cols") {
        RowMatrixXcd in(16, 16), out(16, 32);
        REQUIRE_THROWS_AS(fft.forward(in, out), std::invalid_argument);
    }

    SECTION("inverse row-major: wrong output size") {
        RowMatrixXcd in(16, 32), out(16, 16);
        REQUIRE_THROWS_AS(fft.inverse(in, out), std::invalid_argument);
    }

    SECTION("forward col-major: wrong dimensions") {
        Eigen::MatrixXcd in(8, 32), out(16, 32);
        REQUIRE_THROWS_AS(fft.forward(in, out), std::invalid_argument);
    }
}

// -----------------------------------------------------------------------
// 2D impulse response — delta at origin produces flat spectrum
// -----------------------------------------------------------------------

TEST_CASE("FFT2D of 2D delta is flat spectrum", "[fft2d][correctness]") {
    // x[0,0]=1, rest=0 → X[f1,f2]=1 for all f1,f2
    const Eigen::Index rows = 16, cols = 32;
    FFT2D fft(rows, cols);

    RowMatrixXcd in  = RowMatrixXcd::Zero(rows, cols);
    RowMatrixXcd out(rows, cols);
    in(0, 0) = 1.0;
    fft.forward(in, out);

    for (Eigen::Index f1 = 0; f1 < rows; ++f1)
        for (Eigen::Index f2 = 0; f2 < cols; ++f2)
            REQUIRE_THAT(std::abs(out(f1, f2)), Catch::Matchers::WithinAbs(1.0, kTol));
}

// -----------------------------------------------------------------------
// 2D sinusoid — single spike in spectrum
// -----------------------------------------------------------------------

TEST_CASE("FFT2D of 2D sinusoid produces spike at correct bin", "[fft2d][correctness]") {
    const Eigen::Index rows = 16, cols = 16;
    FFT2D fft(rows, cols);
    const double N = static_cast<double>(rows * cols);

    auto f1 = GENERATE(0, 1, 3, 7);
    auto f2 = GENERATE(0, 2, 5, 15);
    INFO("f1=" << f1 << " f2=" << f2);

    RowMatrixXcd in  = make_sinusoid_2d(rows, cols, f1, f2);
    RowMatrixXcd out(rows, cols);
    fft.forward(in, out);

    for (Eigen::Index r = 0; r < rows; ++r)
        for (Eigen::Index c = 0; c < cols; ++c) {
            double expected = (r == f1 && c == f2) ? N : 0.0;
            REQUIRE_THAT(std::abs(out(r, c)), Catch::Matchers::WithinAbs(expected, kTol * N));
        }
}

// -----------------------------------------------------------------------
// Linearity
// -----------------------------------------------------------------------

TEST_CASE("FFT2D linearity - FFT(a*X + b*Y) == a*FFT(X) + b*FFT(Y)", "[fft2d][correctness]") {
    const Eigen::Index rows = 16, cols = 24;
    FFT2D fft(rows, cols);

    RowMatrixXcd X = RowMatrixXcd::Random(rows, cols);
    RowMatrixXcd Y = RowMatrixXcd::Random(rows, cols);
    const std::complex<double> a(2.0, -1.0), b(0.5, 3.0);

    RowMatrixXcd combined = a * X + b * Y;
    RowMatrixXcd out_combined(rows, cols);
    fft.forward(combined, out_combined);

    RowMatrixXcd out_X(rows, cols), out_Y(rows, cols);
    fft.forward(X, out_X);
    fft.forward(Y, out_Y);

    REQUIRE(max_abs_error(out_combined, a * out_X + b * out_Y) < kTol);
}

// -----------------------------------------------------------------------
// Forward-inverse round-trip
// -----------------------------------------------------------------------

TEST_CASE("FFT2D forward-inverse round-trip recovers original signal", "[fft2d][correctness]") {
    auto rows = GENERATE(8, 16, 32);
    auto cols = GENERATE(8, 24, 32);
    INFO("rows=" << rows << " cols=" << cols);

    SECTION("manual normalization (RowMatrixXcd)") {
        FFT2D fft(rows, cols);
        RowMatrixXcd original = RowMatrixXcd::Random(rows, cols);
        RowMatrixXcd freq(rows, cols), recovered(rows, cols);

        fft.forward(original, freq);
        fft.inverse(freq, recovered);

        double N = static_cast<double>(rows * cols);
        REQUIRE(max_abs_error(recovered / N, original) < kTol);
    }

    SECTION("built-in normalization (RowMatrixXcd)") {
        FFT2D fft(rows, cols, PlanRigor::Measure, true);
        RowMatrixXcd original = RowMatrixXcd::Random(rows, cols);
        RowMatrixXcd freq(rows, cols), recovered(rows, cols);

        fft.forward(original, freq);
        fft.inverse(freq, recovered);

        REQUIRE(max_abs_error(recovered, original) < kTol);
    }

    SECTION("manual normalization (MatrixXcd)") {
        FFT2D fft(rows, cols);
        Eigen::MatrixXcd original = Eigen::MatrixXcd::Random(rows, cols);
        Eigen::MatrixXcd freq(rows, cols), recovered(rows, cols);

        fft.forward(original, freq);
        fft.inverse(freq, recovered);

        double N = static_cast<double>(rows * cols);
        REQUIRE(max_abs_error(recovered / N, original) < kTol);
    }
}

// -----------------------------------------------------------------------
// Row-major and col-major overloads agree
// -----------------------------------------------------------------------

TEST_CASE("FFT2D RowMatrixXcd and MatrixXcd overloads produce identical results", "[fft2d]") {
    const Eigen::Index rows = 16, cols = 20;
    FFT2D fft(rows, cols);

    // Use the same underlying data, different storage order
    RowMatrixXcd in_row = RowMatrixXcd::Random(rows, cols);
    Eigen::MatrixXcd in_col = in_row; // Eigen converts storage order on assignment

    RowMatrixXcd out_row(rows, cols);
    Eigen::MatrixXcd out_col(rows, cols);
    fft.forward(in_row, out_row);
    fft.forward(in_col, out_col);

    // Results must match element-wise
    REQUIRE(max_abs_error(out_row, RowMatrixXcd(out_col)) < kTol);
}

// -----------------------------------------------------------------------
// Parseval's theorem
// -----------------------------------------------------------------------

TEST_CASE("FFT2D satisfies Parseval's theorem", "[fft2d][correctness]") {
    // sum|x[r,c]|² == (1/(R*C)) * sum|X[f1,f2]|²
    const Eigen::Index rows = 16, cols = 24;
    FFT2D fft(rows, cols);

    RowMatrixXcd in  = RowMatrixXcd::Random(rows, cols);
    RowMatrixXcd out(rows, cols);
    fft.forward(in, out);

    double energy_space = in.squaredNorm();
    double energy_freq  = out.squaredNorm() / static_cast<double>(rows * cols);

    REQUIRE_THAT(energy_freq, Catch::Matchers::WithinRel(energy_space, 1e-10));
}

// -----------------------------------------------------------------------
// Separability — FFT2D(u * v^T) = FFT1D(u) * FFT1D(v)^T
// -----------------------------------------------------------------------

TEST_CASE("FFT2D is separable along rows and cols", "[fft2d][correctness]") {
    // If X = u * v^T (outer product), then X[f1,f2] = U[f1] * V[f2]
    // where U = FFT1D(u), V = FFT1D(v).
    const Eigen::Index rows = 16, cols = 24;
    FFT1D fft1d_r(rows);
    FFT1D fft1d_c(cols);
    FFT2D fft2d(rows, cols);

    Eigen::VectorXcd u = Eigen::VectorXcd::Random(rows);
    Eigen::VectorXcd v = Eigen::VectorXcd::Random(cols);

    // Build outer product as row-major matrix
    RowMatrixXcd X(rows, cols);
    for (Eigen::Index r = 0; r < rows; ++r)
        for (Eigen::Index c = 0; c < cols; ++c)
            X(r, c) = u[r] * v[c];

    RowMatrixXcd X_fft(rows, cols);
    fft2d.forward(X, X_fft);

    Eigen::VectorXcd U(rows), V(cols);
    fft1d_r.forward(u, U);
    fft1d_c.forward(v, V);

    // X_fft[f1,f2] should equal U[f1] * V[f2]
    double err = 0.0;
    for (Eigen::Index f1 = 0; f1 < rows; ++f1)
        for (Eigen::Index f2 = 0; f2 < cols; ++f2)
            err = std::max(err, std::abs(X_fft(f1, f2) - U[f1] * V[f2]));

    REQUIRE(err < kTol * rows * cols);
}

// -----------------------------------------------------------------------
// Normalization flag
// -----------------------------------------------------------------------

TEST_CASE("FFT2D normalize=false (default): inverse is unnormalized", "[fft2d][normalize]") {
    const Eigen::Index rows = 16, cols = 16;
    FFT2D fft(rows, cols);

    RowMatrixXcd original = RowMatrixXcd::Random(rows, cols);
    RowMatrixXcd freq(rows, cols), recovered(rows, cols);
    fft.forward(original, freq);
    fft.inverse(freq, recovered);

    double N = static_cast<double>(rows * cols);
    REQUIRE(max_abs_error(recovered, N * original) < kTol);
}

TEST_CASE("FFT2D normalize flag does not affect forward transform", "[fft2d][normalize]") {
    const Eigen::Index rows = 16, cols = 16;
    FFT2D fft_raw(rows, cols, PlanRigor::Measure, false);
    FFT2D fft_norm(rows, cols, PlanRigor::Measure, true);

    RowMatrixXcd in = RowMatrixXcd::Random(rows, cols);
    RowMatrixXcd out_raw(rows, cols), out_norm(rows, cols);
    fft_raw.forward(in, out_raw);
    fft_norm.forward(in, out_norm);

    REQUIRE(max_abs_error(out_raw, out_norm) < kTol);
}

// -----------------------------------------------------------------------
// Plan reuse
// -----------------------------------------------------------------------

TEST_CASE("FFT2D plan can be reused across multiple calls", "[fft2d]") {
    const Eigen::Index rows = 16, cols = 16;
    FFT2D fft(rows, cols);

    for (int f1 = 0; f1 < 4; ++f1) {
        for (int f2 = 0; f2 < 4; ++f2) {
            RowMatrixXcd in  = make_sinusoid_2d(rows, cols, f1, f2);
            RowMatrixXcd out(rows, cols);
            fft.forward(in, out);

            double N = static_cast<double>(rows * cols);
            REQUIRE_THAT(std::abs(out(f1, f2)),
                Catch::Matchers::WithinAbs(N, kTol * N));
        }
    }
}

// -----------------------------------------------------------------------
// Wisdom
// -----------------------------------------------------------------------

namespace {
    const char* kWisdom2DPath = "/tmp/sirius_test_wisdom_2d.fftw";
}

TEST_CASE("FFT2D saveWisdom and loadWisdom", "[fft2d][wisdom]") {
    std::remove(kWisdom2DPath);

    FFT2D fft(32, 32);
    REQUIRE_NOTHROW(FFT2D::saveWisdom(kWisdom2DPath));

    FILE* f = std::fopen(kWisdom2DPath, "r");
    REQUIRE(f != nullptr);
    if (f) std::fclose(f);

    REQUIRE_NOTHROW(FFT2D::loadWisdom(kWisdom2DPath));

    // Plan created after loading wisdom should still produce correct results
    FFT2D fft2(32, 32);
    RowMatrixXcd in  = RowMatrixXcd::Zero(32, 32);
    RowMatrixXcd out(32, 32);
    in(0, 0) = 1.0;
    fft2.forward(in, out);

    for (Eigen::Index f1 = 0; f1 < 32; ++f1)
        for (Eigen::Index f2 = 0; f2 < 32; ++f2)
            REQUIRE_THAT(std::abs(out(f1, f2)), Catch::Matchers::WithinAbs(1.0, kTol));

    std::remove(kWisdom2DPath);
}

TEST_CASE("FFT2D saveWisdom to invalid path throws", "[fft2d][wisdom]") {
    REQUIRE_THROWS_AS(
        FFT2D::saveWisdom("/nonexistent_dir/wisdom.fftw"),
        std::runtime_error
    );
}
