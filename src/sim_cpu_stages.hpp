#ifndef SIRIUS_SIM_CPU_STAGES_HPP
#define SIRIUS_SIM_CPU_STAGES_HPP

// OpenMP implementations of the data-parallel SIM preprocessing stages on
// host memory. Single source of truth for the CPU: the public Eigen API
// (preprocess.hpp / separation.hpp) and the CpuSimBackend of the
// reconstruction pipeline both call these, so the two never drift apart and
// the reference tests of the public API cover the pipeline's kernels too.
//
// Layout: `data` is (nsec, ny, nx) row-major with nsec = ndirs*nphases*nz
// sections in (direction, phase, z) order, i.e. section (d, p, z) is
// data + ((d*nphases + p)*nz + z) * ny*nx.

#include "sim_math.hpp"

namespace sirius::simdetail::cpu {

    // data[i] = (data[i] - sub) * mul
    void scaleShift(double* data, IndexT n, double sub, double mul);

    // sums[p] = sum over section p of data (nplanes, planeElems)
    void planeSums(const double* data, IndexT nplanes, IndexT planeElems, double* sums);

    // data[p][:] *= factors[p]
    void scalePlanes(double* data, IndexT nplanes, IndexT planeElems, const double* factors);

    // Bleach-correction factors from the section sums S(d, p, z): every
    // section is scaled so its sum matches the direction-0/phase-0 reference
    // of the same z (or of z = 0 when equalizez). Dark sections (sum 0) keep
    // factor 1 rather than dividing by zero. sums/factors are (ndirs*nphases*nz).
    void bleachFactors(const double* sums, IndexT ndirs, IndexT nphases, IndexT nz,
                       bool equalizez, double* factors);

    // Edge apodization: blend opposite edges of every section over a
    // napodize-wide border (top/bottom rows first, then left/right columns;
    // the column pass sees the row pass's result). No-op for napodize <= 0.
    void edgeApodize(double* data, IndexT nsec, IndexT ny, IndexT nx, int napodize);

    // Separable sine window: section *= sin(pi*(x+0.5)/nx) * sin(pi*(y+0.5)/ny)
    void cosineApodize(double* data, IndexT nsec, IndexT ny, IndexT nx);

    // Band separation: bands[b][i] = sum_p mat[b*nphases + p] * phases[p][i]
    // for nbands output and nphases input volumes of n reals each; mat is
    // the row-major (nbands, nphases) separation matrix.
    void separate(const double* phases, double* bands, const double* mat,
                  int nphases, int nbands, IndexT n);

} // namespace sirius::simdetail::cpu

#endif // SIRIUS_SIM_CPU_STAGES_HPP
