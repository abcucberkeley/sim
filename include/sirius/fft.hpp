#ifndef SIRIUS_FFT_HPP
#define SIRIUS_FFT_HPP

#include <Eigen/Core>

namespace sirius {
    Eigen::VectorXcd fft1d(const Eigen::VectorXcd& vec);
    Eigen::MatrixXcd fft2d(const Eigen::MatrixXcd& mat);
} // namespace sirius

#endif // SIRIUS_FFT_HPP