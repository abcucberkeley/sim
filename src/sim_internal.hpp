#ifndef SIRIUS_SIM_INTERNAL_HPP
#define SIRIUS_SIM_INTERNAL_HPP

// Internal seam between the device-agnostic SIM reconstruction driver
// (sim_reconstruction.cpp) and the per-device stage implementations
// (sim_cpu.cpp, cuda/sim_kernels.cu). The driver owns all buffers, FFT plans
// and the host-side control flow (k0 bracket search, parabola fits, parameter
// bookkeeping); a SimBackend runs the data-parallel stages on its device.
//
// Pointer arguments refer to memory on the backend's device unless the name
// says host*. Host-pointer arguments are small (per-plane or per-order
// scalars) and are consumed synchronously. CUDA stages are enqueued on the
// stream given at construction; results returned by value (reductions) imply
// a synchronization.

#include <memory>

#include "sirius/device.hpp"
#include "sim_math.hpp"

namespace sirius::simdetail {

    struct ModampSums {
        Cd xy{0, 0};        // sum conj(ov0) * ov1 * exp(i*phase-ramp)
        double sumX = 0;    // sum |ov0|^2
        double sumY = 0;    // sum |ov1|^2
    };

    class SimBackend {
    public:
        virtual ~SimBackend() = default;

        // data[i] = (data[i] - sub) * mul (background + inscale)
        virtual void scaleShift(double* data, IndexT n, double sub, double mul) = 0;

        // hostSums[p] = sum over plane p; data is (nplanes, planeElems)
        virtual void planeSums(const double* data, IndexT nplanes, IndexT planeElems,
                               double* hostSums) = 0;

        // data[p][:] *= hostFactors[p]
        virtual void scalePlanes(double* data, IndexT nplanes, IndexT planeElems,
                                 const double* hostFactors) = 0;

        // per-section edge blend / cosine window; data is (nsec, ny, nx)
        virtual void edgeApodize(double* data, IndexT nsec, IndexT ny, IndexT nx,
                                 int napodize) = 0;
        virtual void cosineApodize(double* data, IndexT nsec, IndexT ny, IndexT nx) = 0;

        // bands[b] = sum_p hostMat[b*nphases+p] * phases[p]; volumes of n reals
        virtual void separate(const double* phases, double* bands, const double* hostMat,
                              int nphases, int nbands, IndexT n) = 0;

        // Fill the (nz, ny, nx) overlap volumes for one order pair; planes with
        // |z| > zdistcutoff stay zero (volumes are pre-zeroed by the driver).
        // Null bandIm1 selects the order-0 gather.
        virtual void makeOverlaps(const OverlapCtx& ctx,
                                  const Cd* bandRe1, const Cd* bandIm1,
                                  const Cd* bandRe2, const Cd* bandIm2,
                                  Cd* ov0, Cd* ov1, int zdistcutoff) = 0;

        // plane[y, x] = sum_z ov0[z, y, x] * conj(ov1[z, y, x])
        virtual void crossCorrelate(const Cd* ov0, const Cd* ov1, Cd* plane,
                                    IndexT nz, IndexT ny, IndexT nx) = 0;

        // Reductions for the modulation amplitude: the ramp angle per voxel is
        // angleX*(ix - nx/2) + angleY*(iy - ny/2). Synchronizes.
        virtual ModampSums modampReduce(const Cd* ov0, const Cd* ov1,
                                        IndexT nz, IndexT ny, IndexT nx,
                                        double angleX, double angleY) = 0;

        // Wiener-filter all orders of one direction in place. bands is the
        // direction's contiguous (2*norders-1, nz, ny, nxh) storage;
        // hostZd/hostConjamp are per-order (norders) host arrays.
        virtual void filterBands(const FilterCtx& ctx, Cd* bands,
                                 const int* hostZd, const Cd* hostConjamp) = 0;

        // Embed one order band into the pre-zeroed big Fourier grid.
        virtual void moveBand(const MoveCtx& ctx, const Cd* bandRe, const Cd* bandIm,
                              Cd* big) = 0;

        // out += (order 0: Re big) or 2*(Re big * cos - Im big * sin) with the
        // per-voxel angle angleX*(ix - xdim/2) + angleY*(iy - ydim/2).
        virtual void accumulate(double* out, const Cd* big, int order,
                                double angleX, double angleY,
                                IndexT zdim, IndexT ydim, IndexT xdim) = 0;

        // Block until all enqueued work is done (no-op on the CPU).
        virtual void synchronize() = 0;
    };

    std::unique_ptr<SimBackend> makeCpuSimBackend();
    // Defined in cuda/sim_kernels.cu when built with CUDA; the CPU-only build
    // provides a stub that throws. `stream` must outlive the backend.
    std::unique_ptr<SimBackend> makeCudaSimBackend(Device device, const Stream& stream);

} // namespace sirius::simdetail

#endif // SIRIUS_SIM_INTERNAL_HPP
