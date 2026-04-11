#ifndef SIRIUS_FFT_HPP
#define SIRIUS_FFT_HPP

#include <memory>
#include <string>
#include <Eigen/Core>

namespace sirius {

    // Row-major complex matrix — matches FFTW's expected C-order layout for 2D plans.
    // Prefer this over Eigen::MatrixXcd when calling FFT2D for zero-copy execution.
    using RowMatrixXcd = Eigen::Matrix<std::complex<double>,
                                       Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Planning rigor controls the time FFTW spends searching for an optimal plan.
    // Higher rigor = better runtime FFT performance, but longer one-time planning cost.
    // Use Estimate for exploratory work; Measure or Patient for production runs on
    // fixed-size transforms that execute many times.
    enum class PlanRigor {
        Estimate,   // No measurement. Fast planning, suboptimal execution.
        Measure,    // Measure a few strategies. Good balance (seconds of planning).
        Patient,    // Measure many strategies. Better plan, slower to create.
        Exhaustive, // Try everything. Rarely worth it over Patient.
    };

    class FFT1D {
    public:
        explicit FFT1D(Eigen::Index n, PlanRigor rigor = PlanRigor::Measure, bool normalize = false);
        ~FFT1D(); 

        // delete copy constructors
        FFT1D(const FFT1D&) = delete;
        FFT1D& operator=(const FFT1D&) = delete;

        // move ops defined in .cpp for the same reason as the destructor
        FFT1D(FFT1D&&) noexcept;
        FFT1D& operator=(FFT1D&&) noexcept;

        // Forward and inverse FFT1D transforms.
        // std::complex<double> is layout-compatible with fftw_complex per the C++ standard
        // and FFTW documentation, so Eigen data is passed directly without copying.
        void forward(const Eigen::VectorXcd& in, Eigen::VectorXcd& out) const;
        void inverse(const Eigen::VectorXcd& in, Eigen::VectorXcd& out) const;

        // Save/load fft wisdom
        static void loadWisdom(const std::string& path);
        static void saveWisdom(const std::string& path);

    private:
        // Use Pimpl (pointer to implementation) pattern for fftw plan vars
        // otherwise, fftw details would have to be exposed to the consumer of the header
        struct Impl; // holds forward_plan, inverse_plan and Eigen index n
        std::unique_ptr<Impl> impl_;
        bool normalize_ = false;
    };

    class FFT2D {
    public:
        explicit FFT2D(Eigen::Index rows, Eigen::Index cols,
                       PlanRigor rigor = PlanRigor::Measure, bool normalize = false);
        ~FFT2D();

        // delete copy constructors
        FFT2D(const FFT2D&) = delete;
        FFT2D& operator=(const FFT2D&) = delete;

        // move ops defined in .cpp for the same reason as the destructor
        FFT2D(FFT2D&&) noexcept;
        FFT2D& operator=(FFT2D&&) noexcept;

        // Zero-copy path: RowMatrixXcd matches FFTW's C-order layout.
        // Executes directly when aligned; falls back to scratch buffers when not.
        void forward(const RowMatrixXcd& in, RowMatrixXcd& out) const;
        void inverse(const RowMatrixXcd& in, RowMatrixXcd& out) const;

        // Column-major convenience overloads — always copies due to layout conversion.
        void forward(const Eigen::MatrixXcd& in, Eigen::MatrixXcd& out) const;
        void inverse(const Eigen::MatrixXcd& in, Eigen::MatrixXcd& out) const;

        // Wisdom is global FFTW state — these are equivalent to FFT1D::loadWisdom/saveWisdom.
        static void loadWisdom(const std::string& path);
        static void saveWisdom(const std::string& path);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        bool normalize_ = false;
    };

} // namespace sirius

#endif // SIRIUS_FFT_HPP