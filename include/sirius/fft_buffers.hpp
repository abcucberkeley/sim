#ifndef SIRIUS_FFT_BUFFERS_HPP
#define SIRIUS_FFT_BUFFERS_HPP

#include <cassert>
#include <stdexcept>
#include <fftw3.h>
#include <limits>
#include <Eigen/Core>

namespace sirius {

    // 1D Complex buffer
    class FFTWBuffer1D {
    public:
        FFTWBuffer1D(Eigen::Index n): n_(n) {
            if (n <= 0) throw std::invalid_argument("Size must be positive");
            if (n > static_cast<Eigen::Index>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("FFT dimension exceeds FFTW's int limit.");
            }
            data_ = fftw_alloc_complex(static_cast<size_t>(n));
            if (!data_) throw std::bad_alloc();
        }

        ~FFTWBuffer1D() {fftw_free(data_);}

        // delete copy constructor
        FFTWBuffer1D(const FFTWBuffer1D&) = delete;
        FFTWBuffer1D& operator=(const FFTWBuffer1D&) = delete;

        // move constructor
        FFTWBuffer1D(FFTWBuffer1D&& other) noexcept: data_(other.data_), n_(other.n_) {
            other.data_ = nullptr;
            other.n_ = 0;
        }
        FFTWBuffer1D& operator=(FFTWBuffer1D&& other) noexcept {
            if (this != &other) {
                fftw_free(data_);
                data_ = other.data_;
                n_ = other.n_;
                other.data_ = nullptr;
                other.n_ = 0;
            }
            return *this;
        }

        // access raw data pointers
        fftw_complex* data() {return data_;}
        const fftw_complex* data() const {return data_;}

        // lightweight eigen conversion
        Eigen::Map<Eigen::VectorXcd> as_eigen() {
            return {reinterpret_cast<std::complex<double>*>(data_), n_};
        }

        Eigen::Index size() const {return n_;}

    private:
        fftw_complex* data_ = nullptr;
        Eigen::Index n_ = 0;
    };

    // 2D Complex buffer
    class FFTWBuffer2D {
    public:
        using EigenMap = Eigen::Map<
            Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

        FFTWBuffer2D(Eigen::Index rows, Eigen::Index cols): rows_(rows), cols_(cols), size_(rows * cols) {
            if (rows <= 0 || cols <= 0) throw std::invalid_argument("Size must be positive");
            auto indmax = static_cast<Eigen::Index>(std::numeric_limits<int>::max());
            if (rows > indmax || cols > indmax || size_ > indmax) {
                throw std::invalid_argument("FFT dimension exceeds FFTW's int limit.");
            }
            data_ = fftw_alloc_complex(static_cast<size_t>(size_));
            if (!data_) throw std::bad_alloc();
        }

        ~FFTWBuffer2D() {fftw_free(data_);}

        // delete copy constructor
        FFTWBuffer2D(const FFTWBuffer2D&) = delete;
        FFTWBuffer2D& operator=(const FFTWBuffer2D&) = delete;

        // move constructor
        FFTWBuffer2D(FFTWBuffer2D&& other) noexcept
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_), size_(other.size_) {
            other.data_ = nullptr;
            other.rows_ = 0;
            other.cols_ = 0;
            other.size_ = 0;
        }

        FFTWBuffer2D& operator=(FFTWBuffer2D&& other) noexcept {
            if (this != &other) {
                fftw_free(data_);
                data_ = other.data_;
                rows_ = other.rows_;
                cols_ = other.cols_;
                size_ = other.size_;
                other.data_ = nullptr;
                other.rows_ = 0;
                other.cols_ = 0;
                other.size_ = 0;
            }
            return *this;
        }

        // access raw data pointers
        fftw_complex* data() {return data_;}
        const fftw_complex* data() const {return data_;}

        // lightweight eigen conversion — row-major to match FFTW's C-order layout
        EigenMap as_eigen() {
            return {reinterpret_cast<std::complex<double>*>(data_), rows_, cols_};
        }

        Eigen::Index rows() const {return rows_;}
        Eigen::Index cols() const {return cols_;}
        Eigen::Index size() const {return size_;}

    private:
        fftw_complex* data_ = nullptr;
        Eigen::Index rows_ = 0;
        Eigen::Index cols_ = 0;
        Eigen::Index size_ = 0;
    };

    // 3D Complex buffer
    class FFTWBuffer3D {
    public:
        using SliceMap = Eigen::Map<
            Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

        FFTWBuffer3D(Eigen::Index depth, Eigen::Index rows, Eigen::Index cols)
        : depth_(depth), rows_(rows), cols_(cols) {
            if (depth <= 0 || rows <= 0 || cols <= 0)
                throw std::invalid_argument("Size must be positive");
            auto indmax = static_cast<Eigen::Index>(std::numeric_limits<int>::max());
            if (depth > indmax || rows > indmax || cols > indmax)
                throw std::invalid_argument("FFT dimension exceeds FFTW's int limit.");
            // staged check to avoid overflow before the product is formed
            if (depth > indmax / rows || depth * rows > indmax / cols)
                throw std::invalid_argument("FFT dimension exceeds FFTW's int limit.");
            size_ = depth * rows * cols;
            data_ = fftw_alloc_complex(static_cast<size_t>(size_));
            if (!data_) throw std::bad_alloc();
        }

        ~FFTWBuffer3D() {fftw_free(data_);}

        // delete copy constructor
        FFTWBuffer3D(const FFTWBuffer3D&) = delete;
        FFTWBuffer3D& operator=(const FFTWBuffer3D&) = delete;

        // move constructor
        FFTWBuffer3D(FFTWBuffer3D&& other) noexcept
        : data_(other.data_), depth_(other.depth_), rows_(other.rows_),
          cols_(other.cols_), size_(other.size_) {
            other.data_ = nullptr;
            other.depth_ = 0;
            other.rows_ = 0;
            other.cols_ = 0;
            other.size_ = 0;
        }

        FFTWBuffer3D& operator=(FFTWBuffer3D&& other) noexcept {
            if (this != &other) {
                fftw_free(data_);
                data_ = other.data_;
                depth_ = other.depth_;
                rows_ = other.rows_;
                cols_ = other.cols_;
                size_ = other.size_;
                other.data_ = nullptr;
                other.depth_ = 0;
                other.rows_ = 0;
                other.cols_ = 0;
                other.size_ = 0;
            }
            return *this;
        }

        // access raw data pointers
        fftw_complex* data() {return data_;}
        const fftw_complex* data() const {return data_;}

        // return a 2D row-major Eigen map for plane z (non-owning view into this buffer)
        SliceMap slice(Eigen::Index z) {
            assert(z >= 0 && z < depth_);
            Eigen::Index stride = rows_ * cols_;
            return {reinterpret_cast<std::complex<double>*>(data_) + z * stride, rows_, cols_};
        }

        // element access
        std::complex<double>& operator()(Eigen::Index z, Eigen::Index r, Eigen::Index c) noexcept {
            assert(z >= 0 && z < depth_ && r >= 0 && r < rows_ && c >= 0 && c < cols_);
            return reinterpret_cast<std::complex<double>*>(data_)[z * rows_ * cols_ + r * cols_ + c];
        }


        Eigen::Index depth() const {return depth_;}
        Eigen::Index rows() const {return rows_;}
        Eigen::Index cols() const {return cols_;}
        Eigen::Index size() const {return size_;}

    private:
        fftw_complex* data_ = nullptr;
        Eigen::Index depth_ = 0;
        Eigen::Index rows_ = 0;
        Eigen::Index cols_ = 0;
        Eigen::Index size_ = 0;
    };


}


#endif // SIRIUS_FFT_BUFFERS_HPP
