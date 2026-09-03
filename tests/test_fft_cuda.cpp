// cuFFT backend of sirius::FFT, checked against the FFTW backend on the same
// input. Skips when no CUDA device is usable.

#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <random>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/fft.hpp"

using namespace sirius;
using Cplx = std::complex<double>;

namespace {
    Device gpuOrSkip() {
        if (!cudaAvailable()) SKIP("no CUDA device available");
        return Device::cuda(0);
    }

    Buffer<Cplx> randomHost(Shape shape, unsigned seed) {
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
}

TEST_CASE("FFT on CUDA matches FFTW", "[fft][cuda]") {
    const Device gpu = gpuOrSkip();
    // (dims, howmany) covering 1D/2D/3D and batching
    struct Case { std::vector<int> dims; int howmany; };
    const Case cases[] = {{{64}, 1}, {{15, 17}, 1}, {{4, 8, 8}, 1}, {{32}, 5}, {{8, 8}, 3}};

    for (const Case& c : cases) {
        INFO("dims rank " << c.dims.size() << " howmany " << c.howmany);
        FFT cpu(c.dims, c.howmany, PlanRigor::Estimate);
        FFT cuda(c.dims, c.howmany, PlanRigor::Estimate, gpu);
        REQUIRE(cuda.device() == gpu);
        REQUIRE(cuda.size() == cpu.size());
        REQUIRE(cuda.shape() == cpu.shape());

        Buffer<Cplx> in = randomHost(cpu.shape(), 7);
        Buffer<Cplx> outCpu(cpu.shape());
        cpu.fft(in, outCpu);

        Stream stream(gpu);
        Buffer<Cplx> dIn = toDevice(in, gpu, stream);
        Buffer<Cplx> dOut(cpu.shape(), gpu, HostMemory::Pageable, stream);
        cuda.fft(dIn, dOut, stream);
        Buffer<Cplx> outGpu = dOut.to(Device::cpu(), stream);
        stream.synchronize();
        REQUIRE(maxAbsDiff(outCpu, outGpu) < 1e-9);

        // inverse with normalization recovers the input
        Buffer<Cplx> dBack(cpu.shape(), gpu, HostMemory::Pageable, stream);
        cuda.ifft(dOut, dBack, true, stream);
        Buffer<Cplx> back = dBack.to(Device::cpu(), stream);
        stream.synchronize();
        REQUIRE(maxAbsDiff(in, back) < 1e-10);

        // unnormalized inverse scales by N
        cuda.ifft(dOut, dBack, false, stream);
        back = dBack.to(Device::cpu(), stream);
        stream.synchronize();
        Index total = 1;
        for (int d : c.dims) total *= d;
        for (Index i = 0; i < in.size(); ++i)
            REQUIRE(std::abs(back.data()[i] - static_cast<double>(total) * in.data()[i]) < 1e-8);
    }
}

TEST_CASE("FFT rejects views on the wrong device or of the wrong size", "[fft][cuda]") {
    const Device gpu = gpuOrSkip();
    FFT cuda({16}, 1, PlanRigor::Estimate, gpu);
    FFT cpu({16}, 1, PlanRigor::Estimate);

    Buffer<Cplx> host(Shape{16});
    Buffer<Cplx> dev(Shape{16}, gpu);
    Buffer<Cplx> devSmall(Shape{8}, gpu);

    REQUIRE_THROWS_AS(cuda.fft(host, host), std::invalid_argument);   // host memory on a CUDA plan
    REQUIRE_THROWS_AS(cpu.fft(dev, dev), std::invalid_argument);      // device memory on a CPU plan
    REQUIRE_THROWS_AS(cuda.fft(dev, devSmall), std::invalid_argument);
    REQUIRE_NOTHROW(cuda.fft(dev, dev));   // in-place is allowed
    Stream::null().synchronize();
}

TEST_CASE("FFT plan on CUDA can be moved", "[fft][cuda]") {
    const Device gpu = gpuOrSkip();
    FFT a({8, 8}, 1, PlanRigor::Estimate, gpu);
    FFT b = std::move(a);
    Buffer<Cplx> dev(Shape{8, 8}, gpu);
    fill(dev, Cplx(1.0, 0.0));
    Buffer<Cplx> out(Shape{8, 8}, gpu);
    b.fft(dev, out);
    auto h = toEigen<2>(out);
    REQUIRE(std::abs(h(0, 0) - Cplx(64.0, 0.0)) < 1e-9);   // DC bin = sum
    REQUIRE(std::abs(h(1, 1)) < 1e-9);
}
