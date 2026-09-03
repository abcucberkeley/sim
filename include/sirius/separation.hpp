#ifndef SIRIUS_SEPARATION_HPP
#define SIRIUS_SEPARATION_HPP

#include "sirius/tensor_util.hpp"

#include <Eigen/Core>

namespace sirius {
    // PatternModel allows modality specific matrix building
    //  - phases: 
    struct PatternModel {
        Eigen::MatrixXd phases; // (nphases, nbasis)
        Eigen::MatrixXd harmonics; // (norders-1, nbasis)
    };

    // PatternModel builders
    PatternModel customPattern(const Eigen::VectorXd& phases, int norders); // phi_j = phases[j]
    PatternModel idealPattern(int nphases, int norders, double offset = 0.); // phi_j = 2*pi*j/nphases + offset
    PatternModel steppedPattern(int nphases, int norders, double step, double offset = 0.); // phi_j = j*step + offset

} // namespace sirius

#endif // SIRIUS_SEPARATION_HPP