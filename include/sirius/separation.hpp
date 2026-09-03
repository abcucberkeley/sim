#ifndef SIRIUS_SEPARATION_HPP
#define SIRIUS_SEPARATION_HPP

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace sirius {

    // Band separation ("un-mixing"). Each raw SIM image is a phase-weighted
    // superposition of the information bands:
    //
    //   D_j = B_0 + sum_o [ cos(o*phi_j) * re_o + sin(o*phi_j) * im_o ]
    //
    // The separation matrix M (2*norders-1, nphases) recovers the bands as
    // band_i = sum_j M[i, j] * D_j. Row 0 is the widefield order; rows
    // 2o-1 / 2o are the cosine / sine parts of order o, from which the
    // complex side bands follow as bandplus = re + i*im, bandminus = re - i*im.

    // Phase sequence helpers (radians): phi_j = 2*pi*j/nphases + offset, and
    // phi_j = j*step + offset.
    Eigen::VectorXd idealPhases(int nphases, double offset = 0.0);
    Eigen::VectorXd steppedPhases(int nphases, double step, double offset = 0.0);

    // Ideal, equally spaced phases: the analytic cos/sin matrix (the
    // "makematrix" convention of cudasirecon; rows are NOT orthonormalized --
    // band_0 comes out scaled by nphases and the side bands by nphases/2,
    // which the reconstruction's global input scale compensates).
    Eigen::MatrixXd separationMatrix(int nphases, int norders);

    // Arbitrary phases: the forward mixing matrix F[j] = [1, cos(o*phi_j),
    // sin(o*phi_j), ...] is pseudo-inverted (least squares; nphases may
    // exceed 2*norders-1) and rescaled to the same convention as above, so
    // for ideal phases this reproduces separationMatrix(nphases, norders)
    // exactly.
    Eigen::MatrixXd separationMatrix(const Eigen::VectorXd& phases, int norders);

    // Un-mix one direction's phase stack (nphases, nz, ny, nx) into real
    // bands (2*norders-1, nz, ny, nx) with the given separation matrix.
    // (CPU convenience; the reconstruction pipeline runs this step through
    // its device backend.)
    Eigen::Tensor<double, 4, Eigen::RowMajor>
    separateBands(const Eigen::Tensor<double, 4, Eigen::RowMajor>& phaseStack,
                  const Eigen::MatrixXd& matrix);

} // namespace sirius

#endif // SIRIUS_SEPARATION_HPP
