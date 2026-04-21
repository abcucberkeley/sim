#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <Eigen/Core>
#include <cmath>
#include <cstring>
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

    // Fill with a 3D complex sinusoid: x[d,r,c] = exp(2πi*(f1*d/D + f2*r/R + f3*c/C))
    // Its 3D FFT has a single spike of magnitude D*R*C at bin (f1, f2, f3).
    void fill_sinusoid(TensorXcd<3>& buf, int f1, int f2, int f3) {
        const Eigen::Index D = buf.dimension(0);
        const Eigen::Index R = buf.dimension(1);
        const Eigen::Index C = buf.dimension(2);
        for (Eigen::Index d = 0; d < D; ++d)
            for (Eigen::Index r = 0; r < R; ++r)
                for (Eigen::Index c = 0; c < C; ++c)
                    buf(d, r, c) = std::exp(std::complex<double>(0.0,
                        2.0 * pi * (f1 * d / static_cast<double>(D)
                                  + f2 * r / static_cast<double>(R)
                                  + f3 * c / static_cast<double>(C))));
    }

    // Deterministic complex fill — avoids rand() while still exercising non-trivial values.
    void fill_deterministic(TensorXcd<3>& buf) {
        const Eigen::Index D = buf.dimension(0);
        const Eigen::Index R = buf.dimension(1);
        const Eigen::Index C = buf.dimension(2);
        for (Eigen::Index d = 0; d < D; ++d)
            for (Eigen::Index r = 0; r < R; ++r)
                for (Eigen::Index c = 0; c < C; ++c)
                    buf(d, r, c) = std::complex<double>(
                        std::cos(d * 1.3 + r * 0.7 + c * 2.1),
                        std::sin(d * 0.9 + r * 1.5 + c * 0.3));
    }

    double max_abs_error(const TensorXcd<3>& a, const TensorXcd<3>& b) {
        double err = 0.0;
        for (Eigen::Index i = 0; i < a.size(); ++i)
            err = std::max(err, std::abs(a.data()[i] - b.data()[i]));
        return err;
    }

    double sum_sq_norm(const TensorXcd<3>& buf) {
        double s = 0.0;
        for (Eigen::Index i = 0; i < buf.size(); ++i)
            s += std::norm(buf.data()[i]); // std::norm returns |z|²
        return s;
    }
}

// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------

TEST_CASE("FFT 3D construction", "[fft3d]") {
    SECTION("Valid sizes construct without throwing") {
        REQUIRE_NOTHROW(FFT({4, 4, 4}));
        REQUIRE_NOTHROW(FFT({2, 8, 16}));
        REQUIRE_NOTHROW(FFT({1, 1, 1}));
    }

    SECTION("Non-cubic dimensions are valid") {
        REQUIRE_NOTHROW(FFT({3, 5, 7}));
        REQUIRE_NOTHROW(FFT({8, 16, 32}));
    }

    SECTION("Non-power-of-two sizes are valid for FFTW") {
        REQUIRE_NOTHROW(FFT({6, 10, 15}));
    }

    SECTION("All PlanRigor values construct successfully") {
        auto rigor = GENERATE(
            PlanRigor::Estimate,
            PlanRigor::Measure,
            PlanRigor::Patient,
            PlanRigor::Exhaustive
        );
        REQUIRE_NOTHROW(FFT({4, 4, 4}, 1, rigor));
    }

    SECTION("Rank > 3 throws invalid_argument") {
        REQUIRE_THROWS_AS(FFT({2, 2, 2, 2}), std::invalid_argument);
    }
}

// -----------------------------------------------------------------------
// Move semantics
// -----------------------------------------------------------------------

TEST_CASE("FFT 3D move semantics", "[fft3d]") {
    SECTION("Move constructor leaves object in valid state") {
        FFT a({4, 8, 8});
        FFT b(std::move(a));

        TensorXcd<3> in(4, 8, 8); in.setZero();
        in(0, 0, 0) = 1.0;
        TensorXcd<3> out(4, 8, 8);
        REQUIRE_NOTHROW(b.fft(in, out));
    }

    SECTION("Move assignment leaves object in valid state") {
        FFT a({4, 8, 8});
        FFT b({2, 2, 2});
        b = std::move(a);

        TensorXcd<3> in(4, 8, 8); in.setZero();
        in(0, 0, 0) = 1.0;
        TensorXcd<3> out(4, 8, 8);
        REQUIRE_NOTHROW(b.fft(in, out));
    }
}

// -----------------------------------------------------------------------
// 3D impulse response
// -----------------------------------------------------------------------

TEST_CASE("FFT 3D of 3D delta is flat spectrum", "[fft3d][correctness]") {
    // x[0,0,0]=1, rest=0 → X[f1,f2,f3]=1 for all (f1,f2,f3)
    const Eigen::Index D = 4, R = 8, C = 8;
    FFT fft({static_cast<int>(D), static_cast<int>(R), static_cast<int>(C)});

    TensorXcd<3> in(D, R, C); in.setZero();
    in(0, 0, 0) = 1.0;
    TensorXcd<3> out(D, R, C);
    fft.fft(in, out);

    for (Eigen::Index d = 0; d < D; ++d)
        for (Eigen::Index r = 0; r < R; ++r)
            for (Eigen::Index c = 0; c < C; ++c)
                REQUIRE_THAT(std::abs(out(d, r, c)),
                    Catch::Matchers::WithinAbs(1.0, kTol));
}

// -----------------------------------------------------------------------
// 3D sinusoid — single spike in spectrum
// -----------------------------------------------------------------------

TEST_CASE("FFT 3D of 3D sinusoid produces spike at correct bin", "[fft3d][correctness]") {
    const Eigen::Index D = 8, R = 8, C = 8;
    FFT fft({static_cast<int>(D), static_cast<int>(R), static_cast<int>(C)});
    const double N = static_cast<double>(D * R * C);

    auto f1 = GENERATE(0, 1, 3);
    auto f2 = GENERATE(0, 2, 7);
    auto f3 = GENERATE(0, 4, 7);
    INFO("f1=" << f1 << " f2=" << f2 << " f3=" << f3);

    TensorXcd<3> in(D, R, C), out(D, R, C);
    fill_sinusoid(in, f1, f2, f3);
    fft.fft(in, out);

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

TEST_CASE("FFT 3D linearity - fft(a*X + b*Y) == a*fft(X) + b*fft(Y)", "[fft3d][correctness]") {
    const Eigen::Index D = 4, R = 6, C = 8;
    FFT fft({static_cast<int>(D), static_cast<int>(R), static_cast<int>(C)});

    TensorXcd<3> X(D, R, C), Y(D, R, C);
    fill_deterministic(X);
    // Y is a different deterministic fill for variety
    for (Eigen::Index d = 0; d < D; ++d)
        for (Eigen::Index r = 0; r < R; ++r)
            for (Eigen::Index c = 0; c < C; ++c)
                Y(d, r, c) = std::complex<double>(
                    std::sin(d * 2.1 + r * 0.4 + c * 1.7),
                    std::cos(d * 0.5 + r * 2.3 + c * 0.8));

    const std::complex<double> a(2.0, -1.0), b(0.5, 3.0);

    TensorXcd<3> combined(D, R, C);
    const auto* px = X.data();
    const auto* py = Y.data();
    auto* pc = combined.data();
    for (Eigen::Index i = 0; i < X.size(); ++i)
        pc[i] = a * px[i] + b * py[i];

    TensorXcd<3> out_combined(D, R, C), out_X(D, R, C), out_Y(D, R, C);
    fft.fft(combined,    out_combined);
    fft.fft(X,           out_X);
    fft.fft(Y,           out_Y);

    const auto* pox = out_X.data();
    const auto* poy = out_Y.data();
    const auto* poc = out_combined.data();
    double err = 0.0;
    for (Eigen::Index i = 0; i < out_combined.size(); ++i)
        err = std::max(err, std::abs(poc[i] - (a * pox[i] + b * poy[i])));

    REQUIRE(err < kTol);
}

// -----------------------------------------------------------------------
// Forward-inverse round-trip
// -----------------------------------------------------------------------

TEST_CASE("FFT 3D forward-inverse round-trip recovers original signal", "[fft3d][correctness]") {
    auto D = GENERATE(4, 8);
    auto R = GENERATE(4, 6);
    auto C = GENERATE(4, 8);
    INFO("D=" << D << " R=" << R << " C=" << C);

    SECTION("manual normalization") {
        FFT fft({D, R, C});
        TensorXcd<3> original(D, R, C), freq(D, R, C), recovered(D, R, C);
        fill_deterministic(original);

        fft.fft(original, freq);
        fft.ifft(freq, recovered); // unnormalized

        double N = static_cast<double>(D * R * C);
        auto* pr = recovered.data();
        for (Eigen::Index i = 0; i < recovered.size(); ++i)
            pr[i] /= N;

        REQUIRE(max_abs_error(recovered, original) < kTol);
    }

    SECTION("built-in normalization") {
        FFT fft({D, R, C});
        TensorXcd<3> original(D, R, C), freq(D, R, C), recovered(D, R, C);
        fill_deterministic(original);

        fft.fft(original, freq);
        fft.ifft(freq, recovered, /*normalize=*/true);

        REQUIRE(max_abs_error(recovered, original) < kTol);
    }
}

// -----------------------------------------------------------------------
// Parseval's theorem
// -----------------------------------------------------------------------

TEST_CASE("FFT 3D satisfies Parseval's theorem", "[fft3d][correctness]") {
    // sum|x[d,r,c]|² == (1/N) * sum|X[f1,f2,f3]|²  where N = D*R*C
    const Eigen::Index D = 4, R = 6, C = 8;
    FFT fft({static_cast<int>(D), static_cast<int>(R), static_cast<int>(C)});

    TensorXcd<3> in(D, R, C), out(D, R, C);
    fill_deterministic(in);
    fft.fft(in, out);

    double energy_space = sum_sq_norm(in);
    double energy_freq  = sum_sq_norm(out) / static_cast<double>(D * R * C);

    REQUIRE_THAT(energy_freq, Catch::Matchers::WithinRel(energy_space, 1e-10));
}

// -----------------------------------------------------------------------
// Separability — FFT3D(u⊗v⊗w)[f1,f2,f3] = FFT1D(u)[f1] * FFT1D(v)[f2] * FFT1D(w)[f3]
// -----------------------------------------------------------------------

TEST_CASE("FFT 3D is separable along depth, rows, and cols", "[fft3d][correctness]") {
    const Eigen::Index D = 8, R = 8, C = 8;
    FFT fft1d_d({static_cast<int>(D)});
    FFT fft1d_r({static_cast<int>(R)});
    FFT fft1d_c({static_cast<int>(C)});
    FFT fft3d({static_cast<int>(D), static_cast<int>(R), static_cast<int>(C)});

    // Build 1D signals as rank-1 tensors
    TensorXcd<1> u(D), v(R), w(C);
    for (Eigen::Index i = 0; i < D; ++i) u(i) = std::complex<double>(std::cos(i * 1.1), std::sin(i * 0.7));
    for (Eigen::Index i = 0; i < R; ++i) v(i) = std::complex<double>(std::cos(i * 0.5), std::sin(i * 1.3));
    for (Eigen::Index i = 0; i < C; ++i) w(i) = std::complex<double>(std::cos(i * 2.0), std::sin(i * 0.3));

    // Build outer product X[d,r,c] = u[d] * v[r] * w[c]
    TensorXcd<3> X(D, R, C);
    for (Eigen::Index d = 0; d < D; ++d)
        for (Eigen::Index r = 0; r < R; ++r)
            for (Eigen::Index c = 0; c < C; ++c)
                X(d, r, c) = u(d) * v(r) * w(c);

    TensorXcd<3> X_fft(D, R, C);
    fft3d.fft(X, X_fft);

    // Compute 1D FFTs of each axis
    TensorXcd<1> U(D), V(R), W(C);
    fft1d_d.fft(u, U);
    fft1d_r.fft(v, V);
    fft1d_c.fft(w, W);

    // Verify X_fft[f1,f2,f3] == U[f1] * V[f2] * W[f3]
    double err = 0.0;
    for (Eigen::Index f1 = 0; f1 < D; ++f1)
        for (Eigen::Index f2 = 0; f2 < R; ++f2)
            for (Eigen::Index f3 = 0; f3 < C; ++f3)
                err = std::max(err, std::abs(X_fft(f1, f2, f3) - U(f1) * V(f2) * W(f3)));

    REQUIRE(err < kTol * D * R * C);
}

// -----------------------------------------------------------------------
// Normalization flag (per-call on ifft)
// -----------------------------------------------------------------------

TEST_CASE("FFT 3D ifft(normalize=false) default: inverse is unnormalized", "[fft3d][normalize]") {
    const Eigen::Index D = 4, R = 4, C = 4;
    FFT fft({static_cast<int>(D), static_cast<int>(R), static_cast<int>(C)});

    TensorXcd<3> original(D, R, C), freq(D, R, C), recovered(D, R, C);
    fill_deterministic(original);
    fft.fft(original, freq);
    fft.ifft(freq, recovered); // default normalize=false

    double N = static_cast<double>(D * R * C);
    const auto* po = original.data();
    const auto* pr = recovered.data();
    double err = 0.0;
    for (Eigen::Index i = 0; i < recovered.size(); ++i)
        err = std::max(err, std::abs(pr[i] - N * po[i]));

    REQUIRE(err < kTol);
}

// -----------------------------------------------------------------------
// Plan reuse
// -----------------------------------------------------------------------

TEST_CASE("FFT 3D plan can be reused across multiple calls", "[fft3d]") {
    const Eigen::Index D = 4, R = 4, C = 4;
    FFT fft({static_cast<int>(D), static_cast<int>(R), static_cast<int>(C)});

    for (int f1 = 0; f1 < 2; ++f1) {
        for (int f2 = 0; f2 < 2; ++f2) {
            for (int f3 = 0; f3 < 2; ++f3) {
                TensorXcd<3> in(D, R, C), out(D, R, C);
                fill_sinusoid(in, f1, f2, f3);
                fft.fft(in, out);

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
    const std::string kWisdom3DPath =
        (std::filesystem::temp_directory_path() / "sirius_test_wisdom_3d.fftw").string();
}

TEST_CASE("FFT 3D saveWisdom and loadWisdom", "[fft3d][wisdom]") {
    std::remove(kWisdom3DPath.c_str());

    FFT fft({8, 8, 8});
    REQUIRE_NOTHROW(FFT::saveWisdom(kWisdom3DPath));

    FILE* f = std::fopen(kWisdom3DPath.c_str(), "r");
    REQUIRE(f != nullptr);
    if (f) std::fclose(f);

    REQUIRE_NOTHROW(FFT::loadWisdom(kWisdom3DPath));

    FFT fft2({8, 8, 8});
    TensorXcd<3> in(8, 8, 8); in.setZero();
    in(0, 0, 0) = 1.0;
    TensorXcd<3> out(8, 8, 8);
    fft2.fft(in, out);

    for (Eigen::Index d = 0; d < 8; ++d)
        for (Eigen::Index r = 0; r < 8; ++r)
            for (Eigen::Index c = 0; c < 8; ++c)
                REQUIRE_THAT(std::abs(out(d, r, c)),
                    Catch::Matchers::WithinAbs(1.0, kTol));

    std::remove(kWisdom3DPath.c_str());
}
