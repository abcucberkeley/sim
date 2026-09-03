#include "sirius/separation.hpp"
#include "sirius/constants.hpp"

#include "sim_cpu_stages.hpp"

#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>

namespace sirius {

    namespace {
        void requireSizes(int nphases, int norders) {
            if (norders < 1)
                throw std::invalid_argument("separationMatrix: norders must be >= 1");
            if (nphases < 2 * norders - 1)
                throw std::invalid_argument("separationMatrix: need nphases >= 2*norders-1 to separate " +
                                            std::to_string(norders) + " orders");
        }
    } // namespace

    Eigen::VectorXd idealPhases(int nphases, double offset) {
        if (nphases < 1) throw std::invalid_argument("idealPhases: nphases must be >= 1");
        Eigen::VectorXd phases(nphases);
        for (int j = 0; j < nphases; ++j)
            phases(j) = 2.0 * kPi * j / nphases + offset;
        return phases;
    }

    Eigen::VectorXd steppedPhases(int nphases, double step, double offset) {
        if (nphases < 1) throw std::invalid_argument("steppedPhases: nphases must be >= 1");
        Eigen::VectorXd phases(nphases);
        for (int j = 0; j < nphases; ++j)
            phases(j) = j * step + offset;
        return phases;
    }

    Eigen::MatrixXd separationMatrix(int nphases, int norders) {
        requireSizes(nphases, norders);
        const double phi = 2.0 * kPi / nphases;
        Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(2 * norders - 1, nphases);
        mat.row(0).setOnes();
        for (int order = 1; order < norders; ++order) {
            for (int j = 0; j < nphases; ++j) {
                mat(2 * order - 1, j) = std::cos(j * order * phi);
                mat(2 * order, j) = std::sin(j * order * phi);
            }
        }
        return mat;
    }

    Eigen::MatrixXd separationMatrix(const Eigen::VectorXd& phases, int norders) {
        const int nphases = static_cast<int>(phases.size());
        requireSizes(nphases, norders);
        const int nbands = 2 * norders - 1;

        // forward mixing matrix: D = F * bands (bands in the re/im basis)
        Eigen::MatrixXd fwd(nphases, nbands);
        fwd.col(0).setOnes();
        for (int order = 1; order < norders; ++order) {
            for (int j = 0; j < nphases; ++j) {
                fwd(j, 2 * order - 1) = std::cos(order * phases(j));
                fwd(j, 2 * order) = std::sin(order * phases(j));
            }
        }

        // least-squares inverse, rescaled to the cudasirecon convention
        // (row 0 sums to nphases, side-band rows to nphases/2) so that ideal
        // phases reproduce the analytic matrix exactly
        Eigen::MatrixXd pinv = fwd.completeOrthogonalDecomposition().pseudoInverse();
        pinv.row(0) *= static_cast<double>(nphases);
        pinv.bottomRows(nbands - 1) *= 0.5 * static_cast<double>(nphases);
        return pinv;
    }

    Eigen::Tensor<double, 4, Eigen::RowMajor>
    separateBands(const Eigen::Tensor<double, 4, Eigen::RowMajor>& phaseStack,
                  const Eigen::MatrixXd& matrix) {
        using Index = Eigen::Index;
        const Index nphases = phaseStack.dimension(0);
        const Index nz = phaseStack.dimension(1);
        const Index ny = phaseStack.dimension(2);
        const Index nx = phaseStack.dimension(3);
        if (matrix.cols() != nphases)
            throw std::invalid_argument("separateBands: matrix has " + std::to_string(matrix.cols()) +
                                        " columns but the stack has " + std::to_string(nphases) +
                                        " phases");
        const Index nbands = matrix.rows();
        const Index n = nz * ny * nx;

        // the shared kernel takes the matrix row-major and contiguous
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rowMajor = matrix;

        Eigen::Tensor<double, 4, Eigen::RowMajor> bands(nbands, nz, ny, nx);
        if (n > 0)
            simdetail::cpu::separate(phaseStack.data(), bands.data(), rowMajor.data(),
                                     static_cast<int>(nphases), static_cast<int>(nbands), n);
        return bands;
    }

} // namespace sirius
