#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <Eigen/Core>
#include <fftw3.h>

#include "sirius/fft_buffers.hpp"

using namespace sirius;

// -----------------------------------------------------------------------
// FFTWBuffer1D — construction
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer1D construction", "[fft_buffers][1d]") {
    SECTION("Valid sizes construct without throwing") {
        auto n = GENERATE(1, 2, 64, 1024);
        REQUIRE_NOTHROW(FFTWBuffer1D(n));
    }

    SECTION("Zero size throws") {
        REQUIRE_THROWS_AS(FFTWBuffer1D(0), std::invalid_argument);
    }

    SECTION("Negative size throws") {
        REQUIRE_THROWS_AS(FFTWBuffer1D(-1), std::invalid_argument);
    }
}

// -----------------------------------------------------------------------
// FFTWBuffer1D — accessors
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer1D accessors", "[fft_buffers][1d]") {
    FFTWBuffer1D buf(64);

    SECTION("size() returns constructed size") {
        REQUIRE(buf.size() == 64);
    }

    SECTION("data() is non-null") {
        REQUIRE(buf.data() != nullptr);
    }

    SECTION("const data() is non-null") {
        const FFTWBuffer1D& cbuf = buf;
        REQUIRE(cbuf.data() != nullptr);
    }

    SECTION("as_eigen() maps correct size") {
        REQUIRE(buf.as_eigen().size() == 64);
    }

    SECTION("as_eigen() aliases the same memory as data()") {
        auto m = buf.as_eigen();
        REQUIRE(reinterpret_cast<fftw_complex*>(m.data()) == buf.data());
    }

    SECTION("writes via as_eigen() are visible through data()") {
        buf.as_eigen()[3] = std::complex<double>(1.5, -2.5);
        auto* raw = reinterpret_cast<std::complex<double>*>(buf.data());
        REQUIRE(raw[3] == std::complex<double>(1.5, -2.5));
    }
}

// -----------------------------------------------------------------------
// FFTWBuffer1D — move semantics
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer1D move semantics", "[fft_buffers][1d]") {
    SECTION("Move constructor transfers ownership and zeros source") {
        FFTWBuffer1D a(64);
        fftw_complex* ptr = a.data();

        FFTWBuffer1D b(std::move(a));

        REQUIRE(b.data() == ptr);
        REQUIRE(b.size() == 64);
        REQUIRE(a.data() == nullptr);
        REQUIRE(a.size() == 0);
    }

    SECTION("Move assignment transfers ownership and zeros source") {
        FFTWBuffer1D a(64);
        FFTWBuffer1D b(32);
        fftw_complex* ptr = a.data();

        b = std::move(a);

        REQUIRE(b.data() == ptr);
        REQUIRE(b.size() == 64);
        REQUIRE(a.data() == nullptr);
        REQUIRE(a.size() == 0);
    }

    SECTION("Self-assignment is a no-op") {
        FFTWBuffer1D a(64);
        fftw_complex* ptr = a.data();

        a = std::move(a);

        REQUIRE(a.data() == ptr);
        REQUIRE(a.size() == 64);
    }
}

// -----------------------------------------------------------------------
// FFTWBuffer2D — construction
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer2D construction", "[fft_buffers][2d]") {
    SECTION("Valid dimensions construct without throwing") {
        REQUIRE_NOTHROW(FFTWBuffer2D(1, 1));
        REQUIRE_NOTHROW(FFTWBuffer2D(4, 8));
        REQUIRE_NOTHROW(FFTWBuffer2D(128, 256));
    }

    SECTION("Zero rows throws") {
        REQUIRE_THROWS_AS(FFTWBuffer2D(0, 8), std::invalid_argument);
    }

    SECTION("Zero cols throws") {
        REQUIRE_THROWS_AS(FFTWBuffer2D(4, 0), std::invalid_argument);
    }

    SECTION("Negative rows throws") {
        REQUIRE_THROWS_AS(FFTWBuffer2D(-1, 8), std::invalid_argument);
    }

    SECTION("Negative cols throws") {
        REQUIRE_THROWS_AS(FFTWBuffer2D(4, -1), std::invalid_argument);
    }
}

// -----------------------------------------------------------------------
// FFTWBuffer2D — accessors
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer2D accessors", "[fft_buffers][2d]") {
    FFTWBuffer2D buf(4, 8);

    SECTION("rows() and cols() return constructed values") {
        REQUIRE(buf.rows() == 4);
        REQUIRE(buf.cols() == 8);
    }

    SECTION("size() equals rows * cols") {
        REQUIRE(buf.size() == 32);
    }

    SECTION("data() is non-null") {
        REQUIRE(buf.data() != nullptr);
    }

    SECTION("const data() is non-null") {
        const FFTWBuffer2D& cbuf = buf;
        REQUIRE(cbuf.data() != nullptr);
    }

    SECTION("as_eigen() maps correct dimensions") {
        auto m = buf.as_eigen();
        REQUIRE(m.rows() == 4);
        REQUIRE(m.cols() == 8);
    }

    SECTION("as_eigen() aliases the same memory as data()") {
        auto m = buf.as_eigen();
        REQUIRE(reinterpret_cast<fftw_complex*>(m.data()) == buf.data());
    }
}

// -----------------------------------------------------------------------
// FFTWBuffer2D — row-major memory layout
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer2D as_eigen() uses row-major layout matching FFTW C-order", "[fft_buffers][2d]") {
    // FFTW 2D plans assume C-order (row-major): element (r, c) is at offset r*cols + c.
    // as_eigen() must map with RowMajor so Eigen and FFTW agree on element positions.
    FFTWBuffer2D buf(3, 4);
    auto m = buf.as_eigen();

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            m(r, c) = std::complex<double>(r * 10.0 + c, 0.0);

    auto* raw = reinterpret_cast<std::complex<double>*>(buf.data());
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            REQUIRE(raw[r * 4 + c] == std::complex<double>(r * 10.0 + c, 0.0));
}

// -----------------------------------------------------------------------
// FFTWBuffer2D — move semantics
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer2D move semantics", "[fft_buffers][2d]") {
    SECTION("Move constructor transfers ownership, dimensions, and size") {
        FFTWBuffer2D a(4, 8);
        fftw_complex* ptr = a.data();

        FFTWBuffer2D b(std::move(a));

        REQUIRE(b.data() == ptr);
        REQUIRE(b.rows() == 4);
        REQUIRE(b.cols() == 8);
        REQUIRE(b.size() == 32);
        REQUIRE(a.data() == nullptr);
        REQUIRE(a.rows() == 0);
        REQUIRE(a.cols() == 0);
        REQUIRE(a.size() == 0);
    }

    SECTION("Move assignment transfers ownership, dimensions, and size") {
        FFTWBuffer2D a(4, 8);
        FFTWBuffer2D b(2, 2);
        fftw_complex* ptr = a.data();

        b = std::move(a);

        REQUIRE(b.data() == ptr);
        REQUIRE(b.rows() == 4);
        REQUIRE(b.cols() == 8);
        REQUIRE(b.size() == 32);
        REQUIRE(a.data() == nullptr);
        REQUIRE(a.rows() == 0);
        REQUIRE(a.cols() == 0);
        REQUIRE(a.size() == 0);
    }

    SECTION("Self-assignment is a no-op") {
        FFTWBuffer2D a(4, 8);
        fftw_complex* ptr = a.data();

        a = std::move(a);

        REQUIRE(a.data() == ptr);
        REQUIRE(a.rows() == 4);
        REQUIRE(a.cols() == 8);
        REQUIRE(a.size() == 32);
    }
}

// -----------------------------------------------------------------------
// FFTWBuffer3D — construction
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer3D construction", "[fft_buffers][3d]") {
    SECTION("Valid dimensions construct without throwing") {
        REQUIRE_NOTHROW(FFTWBuffer3D(1, 1, 1));
        REQUIRE_NOTHROW(FFTWBuffer3D(4, 8, 16));
        REQUIRE_NOTHROW(FFTWBuffer3D(32, 64, 128));
    }

    SECTION("Zero depth throws") {
        REQUIRE_THROWS_AS(FFTWBuffer3D(0, 4, 4), std::invalid_argument);
    }

    SECTION("Zero rows throws") {
        REQUIRE_THROWS_AS(FFTWBuffer3D(4, 0, 4), std::invalid_argument);
    }

    SECTION("Zero cols throws") {
        REQUIRE_THROWS_AS(FFTWBuffer3D(4, 4, 0), std::invalid_argument);
    }

    SECTION("Negative depth throws") {
        REQUIRE_THROWS_AS(FFTWBuffer3D(-1, 4, 4), std::invalid_argument);
    }

    SECTION("Negative rows throws") {
        REQUIRE_THROWS_AS(FFTWBuffer3D(4, -1, 4), std::invalid_argument);
    }

    SECTION("Negative cols throws") {
        REQUIRE_THROWS_AS(FFTWBuffer3D(4, 4, -1), std::invalid_argument);
    }
}

// -----------------------------------------------------------------------
// FFTWBuffer3D — accessors
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer3D accessors", "[fft_buffers][3d]") {
    FFTWBuffer3D buf(2, 4, 8);

    SECTION("depth(), rows(), cols() return constructed values") {
        REQUIRE(buf.depth() == 2);
        REQUIRE(buf.rows() == 4);
        REQUIRE(buf.cols() == 8);
    }

    SECTION("size() equals depth * rows * cols") {
        REQUIRE(buf.size() == 64);
    }

    SECTION("data() is non-null") {
        REQUIRE(buf.data() != nullptr);
    }

    SECTION("const data() is non-null") {
        const FFTWBuffer3D& cbuf = buf;
        REQUIRE(cbuf.data() != nullptr);
    }
}

// -----------------------------------------------------------------------
// FFTWBuffer3D — operator()(z, r, c)
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer3D operator() element access", "[fft_buffers][3d]") {
    FFTWBuffer3D buf(3, 4, 5);

    SECTION("write via operator() is visible at correct raw offset") {
        buf(1, 2, 3) = std::complex<double>(7.0, -3.0);
        auto* raw = reinterpret_cast<std::complex<double>*>(buf.data());
        // z=1, r=2, c=3 → flat offset = 1*4*5 + 2*5 + 3 = 33
        REQUIRE(raw[1 * 4 * 5 + 2 * 5 + 3] == std::complex<double>(7.0, -3.0));
    }

    SECTION("write via raw pointer is visible through operator()") {
        auto* raw = reinterpret_cast<std::complex<double>*>(buf.data());
        raw[0 * 4 * 5 + 3 * 5 + 4] = std::complex<double>(1.5, 2.5); // z=0, r=3, c=4
        REQUIRE(buf(0, 3, 4) == std::complex<double>(1.5, 2.5));
    }

    SECTION("operator() is consistent with slice()(r, c)") {
        buf(2, 1, 3) = std::complex<double>(4.0, -1.0);
        REQUIRE(buf.slice(2)(1, 3) == std::complex<double>(4.0, -1.0));
    }

    SECTION("independent writes to different elements do not interfere") {
        buf(0, 0, 0) = std::complex<double>(1.0, 0.0);
        buf(1, 2, 3) = std::complex<double>(2.0, 0.0);
        buf(2, 3, 4) = std::complex<double>(3.0, 0.0);

        REQUIRE(buf(0, 0, 0) == std::complex<double>(1.0, 0.0));
        REQUIRE(buf(1, 2, 3) == std::complex<double>(2.0, 0.0));
        REQUIRE(buf(2, 3, 4) == std::complex<double>(3.0, 0.0));
    }

    SECTION("first and last elements are accessible") {
        buf(0, 0, 0)         = std::complex<double>(10.0, 0.0);
        buf(2, 3, 4)         = std::complex<double>(20.0, 0.0);
        REQUIRE(buf(0, 0, 0) == std::complex<double>(10.0, 0.0));
        REQUIRE(buf(2, 3, 4) == std::complex<double>(20.0, 0.0));
    }
}

// -----------------------------------------------------------------------
// FFTWBuffer3D — slice()
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer3D slice() returns correct 2D view", "[fft_buffers][3d]") {
    FFTWBuffer3D buf(3, 4, 5);

    SECTION("slice dimensions match rows and cols") {
        auto s = buf.slice(0);
        REQUIRE(s.rows() == 4);
        REQUIRE(s.cols() == 5);
    }

    SECTION("slice(z) points into the correct offset in the flat buffer") {
        // plane z starts at raw offset z * rows * cols
        for (Eigen::Index z = 0; z < 3; ++z) {
            auto s = buf.slice(z);
            auto* expected_ptr = reinterpret_cast<std::complex<double>*>(buf.data()) + z * 4 * 5;
            REQUIRE(s.data() == expected_ptr);
        }
    }

    SECTION("writes via slice(z) are visible through data() and vice versa") {
        auto s0 = buf.slice(0);
        auto s1 = buf.slice(1);

        s0(1, 2) = std::complex<double>(3.0, -1.0);
        s1(0, 0) = std::complex<double>(7.0,  2.0);

        auto* raw = reinterpret_cast<std::complex<double>*>(buf.data());
        REQUIRE(raw[1 * 5 + 2]       == std::complex<double>(3.0, -1.0)); // z=0, r=1, c=2
        REQUIRE(raw[1 * 4 * 5 + 0]   == std::complex<double>(7.0,  2.0)); // z=1, r=0, c=0
    }

    SECTION("slices of different planes are independent views") {
        auto s0 = buf.slice(0);
        auto s2 = buf.slice(2);

        s0.setZero();
        s2.setConstant(std::complex<double>(1.0, 0.0));

        // s0 should still be zero
        REQUIRE(s0.cwiseAbs().maxCoeff() == 0.0);
        // s2 should be all ones
        REQUIRE(s2.cwiseAbs().minCoeff() == 1.0);
    }
}

TEST_CASE("FFTWBuffer3D slice() uses row-major layout matching FFTW C-order", "[fft_buffers][3d]") {
    // Within each plane, element (r, c) must be at flat offset z*rows*cols + r*cols + c
    FFTWBuffer3D buf(2, 3, 4);

    for (Eigen::Index z = 0; z < 2; ++z) {
        auto s = buf.slice(z);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
                s(r, c) = std::complex<double>(z * 100.0 + r * 10.0 + c, 0.0);
    }

    auto* raw = reinterpret_cast<std::complex<double>*>(buf.data());
    for (Eigen::Index z = 0; z < 2; ++z)
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
                REQUIRE(raw[z * 3 * 4 + r * 4 + c] ==
                        std::complex<double>(z * 100.0 + r * 10.0 + c, 0.0));
}

// -----------------------------------------------------------------------
// FFTWBuffer3D — move semantics
// -----------------------------------------------------------------------

TEST_CASE("FFTWBuffer3D move semantics", "[fft_buffers][3d]") {
    SECTION("Move constructor transfers ownership, all dimensions, and size") {
        FFTWBuffer3D a(2, 4, 8);
        fftw_complex* ptr = a.data();

        FFTWBuffer3D b(std::move(a));

        REQUIRE(b.data() == ptr);
        REQUIRE(b.depth() == 2);
        REQUIRE(b.rows() == 4);
        REQUIRE(b.cols() == 8);
        REQUIRE(b.size() == 64);
        REQUIRE(a.data() == nullptr);
        REQUIRE(a.depth() == 0);
        REQUIRE(a.rows() == 0);
        REQUIRE(a.cols() == 0);
        REQUIRE(a.size() == 0);
    }

    SECTION("Move assignment transfers ownership, all dimensions, and size") {
        FFTWBuffer3D a(2, 4, 8);
        FFTWBuffer3D b(1, 1, 1);
        fftw_complex* ptr = a.data();

        b = std::move(a);

        REQUIRE(b.data() == ptr);
        REQUIRE(b.depth() == 2);
        REQUIRE(b.rows() == 4);
        REQUIRE(b.cols() == 8);
        REQUIRE(b.size() == 64);
        REQUIRE(a.data() == nullptr);
        REQUIRE(a.depth() == 0);
        REQUIRE(a.rows() == 0);
        REQUIRE(a.cols() == 0);
        REQUIRE(a.size() == 0);
    }

    SECTION("Self-assignment is a no-op") {
        FFTWBuffer3D a(2, 4, 8);
        fftw_complex* ptr = a.data();

        a = std::move(a);

        REQUIRE(a.data() == ptr);
        REQUIRE(a.depth() == 2);
        REQUIRE(a.rows() == 4);
        REQUIRE(a.cols() == 8);
        REQUIRE(a.size() == 64);
    }

    SECTION("Slices of moved-into buffer still point into correct memory") {
        FFTWBuffer3D a(2, 4, 8);
        auto s0_before = a.slice(0);
        s0_before(0, 0) = std::complex<double>(42.0, 0.0);

        FFTWBuffer3D b(std::move(a));
        auto s0_after = b.slice(0);

        REQUIRE(s0_after(0, 0) == std::complex<double>(42.0, 0.0));
    }
}
