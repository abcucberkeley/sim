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
    constexpr double kTol = 1e-10;

    using RowMatrixXcd =
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // View a rank-2 row-major tensor as a row-major Eigen matrix (no copy).
    // TensorXcd<2> is RowMajor (see sirius::TensorXcd), so storage matches RowMatrixXcd.
    Eigen::Map<RowMatrixXcd> as_mat(TensorXcd<2>& t) {
        return Eigen::Map<RowMatrixXcd>(t.data(), t.dimension(0), t.dimension(1));
    }
    Eigen::Map<const RowMatrixXcd> as_mat(const TensorXcd<2>& t) {
        return Eigen::Map<const RowMatrixXcd>(t.data(), t.dimension(0), t.dimension(1));
    }

    TensorXcd<2> random_tensor(Eigen::Index rows, Eigen::Index cols) {
        TensorXcd<2> t(rows, cols);
        as_mat(t) = RowMatrixXcd::Random(rows, cols);
        return t;
    }

    // Build a 2D complex sinusoid: x[r,c] = exp(2*pi*i*(f1*r/R + f2*c/C))
    // Its 2D FFT has a single spike of magnitude R*C at bin (f1, f2).
    TensorXcd<2> make_sinusoid_2d(Eigen::Index rows, Eigen::Index cols, int f1, int f2) {
        TensorXcd<2> t(rows, cols);
        for (Eigen::Index r = 0; r < rows; ++r)
            for (Eigen::Index c = 0; c < cols; ++c)
                t(r, c) = std::exp(std::complex<double>(0.0,
                    2.0 * pi * (f1 * r / static_cast<double>(rows)
                              + f2 * c / static_cast<double>(cols))));
        return t;
    }

    double max_abs_error(const TensorXcd<2>& a, const TensorXcd<2>& b) {
        return (as_mat(a) - as_mat(b)).cwiseAbs().maxCoeff();
    }
}

// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------

TEST_CASE("FFT 2D construction", "[fft2d]") {
    SECTION("Valid sizes construct without throwing") {
        REQUIRE_NOTHROW(FFT({4, 4}));
        REQUIRE_NOTHROW(FFT({8, 16}));
        REQUIRE_NOTHROW(FFT({1, 1}));
    }

    SECTION("Non-square dimensions are valid") {
        auto rows = GENERATE(3, 5, 64, 100);
        auto cols = GENERATE(7, 16, 50, 128);
        REQUIRE_NOTHROW(FFT({rows, cols}));
    }

    SECTION("Non-power-of-two sizes are valid for FFTW") {
        REQUIRE_NOTHROW(FFT({100, 150}));
        REQUIRE_NOTHROW(FFT({127, 33}));
    }

    SECTION("All PlanRigor values construct successfully") {
        auto rigor = GENERATE(
            PlanRigor::Estimate,
            PlanRigor::Measure,
            PlanRigor::Patient,
            PlanRigor::Exhaustive
        );
        REQUIRE_NOTHROW(FFT({16, 16}, 1, rigor));
    }
}

// -----------------------------------------------------------------------
// Move semantics
// -----------------------------------------------------------------------

TEST_CASE("FFT 2D move semantics", "[fft2d]") {
    SECTION("Move construction leaves object in valid state") {
        FFT a({32, 32});
        FFT b(std::move(a));

        TensorXcd<2> in(32, 32); in.setZero();
        in(0, 0) = 1.0;
        TensorXcd<2> out(32, 32);
        REQUIRE_NOTHROW(b.fft(in, out));
    }

    SECTION("Move assignment leaves object in valid state") {
        FFT a({32, 32});
        FFT b({16, 16});
        b = std::move(a);

        TensorXcd<2> in(32, 32); in.setZero();
        in(0, 0) = 1.0;
        TensorXcd<2> out(32, 32);
        REQUIRE_NOTHROW(b.fft(in, out));
    }
}

// -----------------------------------------------------------------------
// 2D impulse response — delta at origin produces flat spectrum
// -----------------------------------------------------------------------

TEST_CASE("FFT 2D of 2D delta is flat spectrum", "[fft2d][correctness]") {
    // x[0,0]=1, rest=0 → X[f1,f2]=1 for all f1,f2
    const Eigen::Index rows = 16, cols = 32;
    FFT fft({static_cast<int>(rows), static_cast<int>(cols)});

    TensorXcd<2> in(rows, cols); in.setZero();
    in(0, 0) = 1.0;
    TensorXcd<2> out(rows, cols);
    fft.fft(in, out);

    for (Eigen::Index f1 = 0; f1 < rows; ++f1)
        for (Eigen::Index f2 = 0; f2 < cols; ++f2)
            REQUIRE_THAT(std::abs(out(f1, f2)), Catch::Matchers::WithinAbs(1.0, kTol));
}

// -----------------------------------------------------------------------
// 2D sinusoid — single spike in spectrum
// -----------------------------------------------------------------------

TEST_CASE("FFT 2D of 2D sinusoid produces spike at correct bin", "[fft2d][correctness]") {
    const Eigen::Index rows = 16, cols = 16;
    FFT fft({static_cast<int>(rows), static_cast<int>(cols)});
    const double N = static_cast<double>(rows * cols);

    auto f1 = GENERATE(0, 1, 3, 7);
    auto f2 = GENERATE(0, 2, 5, 15);
    INFO("f1=" << f1 << " f2=" << f2);

    TensorXcd<2> in  = make_sinusoid_2d(rows, cols, f1, f2);
    TensorXcd<2> out(rows, cols);
    fft.fft(in, out);

    for (Eigen::Index r = 0; r < rows; ++r)
        for (Eigen::Index c = 0; c < cols; ++c) {
            double expected = (r == f1 && c == f2) ? N : 0.0;
            REQUIRE_THAT(std::abs(out(r, c)), Catch::Matchers::WithinAbs(expected, kTol * N));
        }
}

// -----------------------------------------------------------------------
// Linearity
// -----------------------------------------------------------------------

TEST_CASE("FFT 2D linearity - fft(a*X + b*Y) == a*fft(X) + b*fft(Y)", "[fft2d][correctness]") {
    const Eigen::Index rows = 16, cols = 24;
    FFT fft({static_cast<int>(rows), static_cast<int>(cols)});

    TensorXcd<2> X = random_tensor(rows, cols);
    TensorXcd<2> Y = random_tensor(rows, cols);
    const std::complex<double> a(2.0, -1.0), b(0.5, 3.0);

    TensorXcd<2> combined(rows, cols);
    as_mat(combined) = a * as_mat(X) + b * as_mat(Y);

    TensorXcd<2> out_combined(rows, cols), out_X(rows, cols), out_Y(rows, cols);
    fft.fft(combined, out_combined);
    fft.fft(X, out_X);
    fft.fft(Y, out_Y);

    RowMatrixXcd expected = a * as_mat(out_X) + b * as_mat(out_Y);
    REQUIRE((as_mat(out_combined) - expected).cwiseAbs().maxCoeff() < kTol);
}

// -----------------------------------------------------------------------
// Forward-inverse round-trip
// -----------------------------------------------------------------------

TEST_CASE("FFT 2D forward-inverse round-trip recovers original signal", "[fft2d][correctness]") {
    auto rows = GENERATE(8, 16, 32);
    auto cols = GENERATE(8, 24, 32);
    INFO("rows=" << rows << " cols=" << cols);

    SECTION("manual normalization") {
        FFT fft({rows, cols});
        TensorXcd<2> original = random_tensor(rows, cols);
        TensorXcd<2> freq(rows, cols), recovered(rows, cols);

        fft.fft(original, freq);
        fft.ifft(freq, recovered); // unnormalized

        double N = static_cast<double>(rows * cols);
        REQUIRE((as_mat(recovered) / N - as_mat(original)).cwiseAbs().maxCoeff() < kTol);
    }

    SECTION("built-in normalization") {
        FFT fft({rows, cols});
        TensorXcd<2> original = random_tensor(rows, cols);
        TensorXcd<2> freq(rows, cols), recovered(rows, cols);

        fft.fft(original, freq);
        fft.ifft(freq, recovered, /*normalize=*/true);

        REQUIRE(max_abs_error(recovered, original) < kTol);
    }
}

// -----------------------------------------------------------------------
// Parseval's theorem
// -----------------------------------------------------------------------

TEST_CASE("FFT 2D satisfies Parseval's theorem", "[fft2d][correctness]") {
    // sum|x[r,c]|² == (1/(R*C)) * sum|X[f1,f2]|²
    const Eigen::Index rows = 16, cols = 24;
    FFT fft({static_cast<int>(rows), static_cast<int>(cols)});

    TensorXcd<2> in  = random_tensor(rows, cols);
    TensorXcd<2> out(rows, cols);
    fft.fft(in, out);

    double energy_space = as_mat(in).squaredNorm();
    double energy_freq  = as_mat(out).squaredNorm() / static_cast<double>(rows * cols);

    REQUIRE_THAT(energy_freq, Catch::Matchers::WithinRel(energy_space, 1e-10));
}

// -----------------------------------------------------------------------
// Separability — FFT2D(u * v^T) = FFT1D(u) * FFT1D(v)^T
// -----------------------------------------------------------------------

TEST_CASE("FFT 2D is separable along rows and cols", "[fft2d][correctness]") {
    // If X = u * v^T (outer product), then X[f1,f2] = U[f1] * V[f2]
    // where U = FFT1D(u), V = FFT1D(v).
    const Eigen::Index rows = 16, cols = 24;
    FFT fft1d_r({static_cast<int>(rows)});
    FFT fft1d_c({static_cast<int>(cols)});
    FFT fft2d({static_cast<int>(rows), static_cast<int>(cols)});

    Eigen::VectorXcd u = Eigen::VectorXcd::Random(rows);
    Eigen::VectorXcd v = Eigen::VectorXcd::Random(cols);

    // Build outer product as a 2D tensor
    TensorXcd<2> X(rows, cols);
    for (Eigen::Index r = 0; r < rows; ++r)
        for (Eigen::Index c = 0; c < cols; ++c)
            X(r, c) = u[r] * v[c];

    TensorXcd<2> X_fft(rows, cols);
    fft2d.fft(X, X_fft);

    TensorXcd<1> u_t(rows), v_t(cols), U(rows), V(cols);
    std::copy(u.data(), u.data() + rows, u_t.data());
    std::copy(v.data(), v.data() + cols, v_t.data());
    fft1d_r.fft(u_t, U);
    fft1d_c.fft(v_t, V);

    // X_fft[f1,f2] should equal U[f1] * V[f2]
    double err = 0.0;
    for (Eigen::Index f1 = 0; f1 < rows; ++f1)
        for (Eigen::Index f2 = 0; f2 < cols; ++f2)
            err = std::max(err, std::abs(X_fft(f1, f2) - U(f1) * V(f2)));

    REQUIRE(err < kTol * rows * cols);
}

// -----------------------------------------------------------------------
// Normalization flag (per-call on ifft)
// -----------------------------------------------------------------------

TEST_CASE("FFT 2D ifft(normalize=false) default: inverse is unnormalized", "[fft2d][normalize]") {
    const Eigen::Index rows = 16, cols = 16;
    FFT fft({static_cast<int>(rows), static_cast<int>(cols)});

    TensorXcd<2> original = random_tensor(rows, cols);
    TensorXcd<2> freq(rows, cols), recovered(rows, cols);
    fft.fft(original, freq);
    fft.ifft(freq, recovered); // default normalize=false

    double N = static_cast<double>(rows * cols);
    REQUIRE((as_mat(recovered) - N * as_mat(original)).cwiseAbs().maxCoeff() < kTol);
}

// -----------------------------------------------------------------------
// Plan reuse
// -----------------------------------------------------------------------

TEST_CASE("FFT 2D plan can be reused across multiple calls", "[fft2d]") {
    const Eigen::Index rows = 16, cols = 16;
    FFT fft({static_cast<int>(rows), static_cast<int>(cols)});

    for (int f1 = 0; f1 < 4; ++f1) {
        for (int f2 = 0; f2 < 4; ++f2) {
            TensorXcd<2> in  = make_sinusoid_2d(rows, cols, f1, f2);
            TensorXcd<2> out(rows, cols);
            fft.fft(in, out);

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
    const std::string kWisdom2DPath =
        (std::filesystem::temp_directory_path() / "sirius_test_wisdom_2d.fftw").string();
}

TEST_CASE("FFT 2D saveWisdom and loadWisdom", "[fft2d][wisdom]") {
    std::remove(kWisdom2DPath.c_str());

    FFT fft({32, 32});
    REQUIRE_NOTHROW(FFT::saveWisdom(kWisdom2DPath));

    FILE* f = std::fopen(kWisdom2DPath.c_str(), "r");
    REQUIRE(f != nullptr);
    if (f) std::fclose(f);

    REQUIRE_NOTHROW(FFT::loadWisdom(kWisdom2DPath));

    // Plan created after loading wisdom should still produce correct results
    FFT fft2({32, 32});
    TensorXcd<2> in(32, 32); in.setZero();
    in(0, 0) = 1.0;
    TensorXcd<2> out(32, 32);
    fft2.fft(in, out);

    for (Eigen::Index f1 = 0; f1 < 32; ++f1)
        for (Eigen::Index f2 = 0; f2 < 32; ++f2)
            REQUIRE_THAT(std::abs(out(f1, f2)), Catch::Matchers::WithinAbs(1.0, kTol));

    std::remove(kWisdom2DPath.c_str());
}
