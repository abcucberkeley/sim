// cuFFT backend of sirius::RealFFT, checked against the FFTW backend on the
// same input. Skips when no CUDA device is usable.

#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <random>
#include <vector>

#include "sirius/buffer.hpp"
#include "sirius/real_fft.hpp"

using namespace sirius;
using Cplx = std::complex<double>;

namespace {
    Device gpuOrSkip() {
        if (!cudaAvailable()) SKIP("no CUDA device available");
        return Device::cuda(0);
    }

    Buffer<double> randomReal(Index n, unsigned seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> d(-1.0, 1.0);
        Buffer<double> b(Shape{n});
        for (Index i = 0; i < n; ++i) b.data()[i] = d(gen);
        return b;
    }
}

TEST_CASE("RealFFT on CUDA matches FFTW", "[real_fft][cuda]") {
    const Device gpu = gpuOrSkip();
    struct Case { std::vector<int> dims; int howmany; };
    const Case cases[] = {{{64}, 1}, {{16, 18}, 1}, {{4, 8, 10}, 1}, {{32}, 4}, {{6, 8, 8}, 5}};

    for (const Case& c : cases) {
        INFO("dims rank " << c.dims.size() << " howmany " << c.howmany);
        RealFFT cpu(c.dims, c.howmany, PlanRigor::Estimate);
        RealFFT cuda(c.dims, c.howmany, PlanRigor::Estimate, gpu);
        REQUIRE(cuda.device() == gpu);
        REQUIRE(cuda.fullRealSize() == cpu.fullRealSize());
        REQUIRE(cuda.fullComplexSize() == cpu.fullComplexSize());

        Buffer<double> in = randomReal(cpu.fullRealSize(), 11);
        Buffer<Cplx> outCpu(Shape{cpu.fullComplexSize()});
        cpu.rfft(in.data(), outCpu.data());

        Stream stream(gpu);
        Buffer<double> dIn = toDevice(in, gpu, stream);
        Buffer<Cplx> dOut(Shape{cpu.fullComplexSize()}, gpu, HostMemory::Pageable, stream);
        cuda.rfft(dIn.data(), dOut.data(), stream);
        Buffer<Cplx> outGpu = dOut.to(Device::cpu(), stream);
        stream.synchronize();

        double maxDiff = 0.0;
        for (Index i = 0; i < outCpu.size(); ++i)
            maxDiff = std::max(maxDiff, std::abs(outCpu.data()[i] - outGpu.data()[i]));
        REQUIRE(maxDiff < 1e-9);

        // normalized inverse recovers the input, and preserves its complex input
        Buffer<Cplx> spectrumCopy = dOut.clone(stream);
        Buffer<double> dBack(Shape{cpu.fullRealSize()}, gpu, HostMemory::Pageable, stream);
        cuda.irfft(dOut.data(), dBack.data(), true, stream);
        Buffer<double> back = dBack.to(Device::cpu(), stream);
        Buffer<Cplx> spectrumAfter = dOut.to(Device::cpu(), stream);
        Buffer<Cplx> spectrumBefore = spectrumCopy.to(Device::cpu(), stream);
        stream.synchronize();

        for (Index i = 0; i < in.size(); ++i)
            REQUIRE(std::abs(back.data()[i] - in.data()[i]) < 1e-10);
        for (Index i = 0; i < spectrumBefore.size(); ++i)
            REQUIRE(spectrumAfter.data()[i] == spectrumBefore.data()[i]);
    }
}
