#include "sirius/separation.hpp"
#include "sirius/constants.hpp"

#include <Eigen/Dense>
#include <cassert>
#include <cmath>

namespace sirius {

    namespace {
        // h = {{1}, {2}, ..., {norders-1}}
        Eigen::MatrixXi scalarHarmonics(int norders) {
            Eigen::MatrixXi h(norders-1, 1);
            for (int i = 1; i < norders; ++i) h(i-1,0) = i;
            return h;
        }
    } // anonymous namespace

    PatternModel customPattern(const Eigen::VectorXd& phases, int norders) {
        PatternModel m;
        m.phases = phases;
        m.harmonics = scalarHarmonics(norders);
        return m;
    }

    // phi_j = 2*pi*j/nphases
    PatternModel idealPattern(int nphases, int norders, double offset = 0.) {
        Eigen::VectorXd phases(nphases);
        for (int j = 0; j < nphases; ++j) {
            phases(j) = 2.0 * kPi * j / nphases + offset;
        }
        return customPattern(phases, norders);
    }

    // phi_j = j*step + offset
    PatternModel steppedPattern(int nphases, int norders, double step, double offset = 0.) {
        Eigen::VectorXd phases(nphases);
        for (int j = 0; j < nphases; ++j) {
            phases(j) = j * step + offset;
        }
        return customPattern(phases, norders);
    }

    


} // namespace sirius