#include "sirius/fft.hpp"
#include <cstring>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <Eigen/Core>
#include <fftw3.h>

namespace sirius {
    namespace {
        // RAII for fft plan
        // Note: fftw_plan_s is a struct and fftw_plan is a pointer to that struct
        struct FFTWPlanDeleter {
            void operator()(fftw_plan plan) const {fftw_destroy_plan(plan);}
        };
        using PlanPtr = std::unique_ptr<fftw_plan_s, FFTWPlanDeleter>;

        // FFTW's planner modifies global state — must be serialized across all instances
        std::mutex s_planner_mutex;

        // map plan rigor to fftw flags
        unsigned int toFFTWFlag(PlanRigor r) {
            switch (r) {
                case PlanRigor::Estimate:   return FFTW_ESTIMATE;
                case PlanRigor::Measure:    return FFTW_MEASURE;
                case PlanRigor::Patient:    return FFTW_PATIENT;
                case PlanRigor::Exhaustive: return FFTW_EXHAUSTIVE;
            }
            throw std::invalid_argument("Unknown PlanRigor value");
        }

        // safe execution in case unaligned buffers with offset are passed
        void execute_safe(fftw_plan plan, int plan_alignment,
            const Eigen::VectorXcd& in, Eigen::VectorXcd& out) {
                auto* in_ptr = reinterpret_cast<fftw_complex*>(const_cast<std::complex<double>*>(in.data()));
                auto* out_ptr = reinterpret_cast<fftw_complex*>(out.data());

                // check if aligned
                bool aligned = fftw_alignment_of(reinterpret_cast<double*>(in_ptr))  == plan_alignment
                && fftw_alignment_of(reinterpret_cast<double*>(out_ptr)) == plan_alignment;

                // if aligned, simply execute otherwise need to make copies
                if (aligned) {
                    fftw_execute_dft(plan, in_ptr, out_ptr);
                } else {
                    // caller passed a misaligned buffer (e.g. vec.segment(1,n) or a non-owning Map)
                    // copy into FFTW-aligned scratch, execute there, copy result back
                    Eigen::Index n = in.size();
                    FFTWBuffer1D tmp_in(n), tmp_out(n);
                    // this will call assignment operator of Map
                    // which will write contents of in, into tmp_in.data_
                    tmp_in.as_eigen() = in;
                    fftw_execute_dft(plan, tmp_in.data(), tmp_out.data());
                    out = tmp_out.as_eigen();
                }
        }

        // 2D counterpart of execute_safe — only valid for row-major input/output.
        // Executes directly when both buffers match plan alignment, otherwise copies
        // through FFTW-aligned scratch (e.g. when a RowMatrixXcd comes from operator new).
        void execute_safe_2d(fftw_plan plan, int plan_alignment,
            const RowMatrixXcd& in, RowMatrixXcd& out,
            Eigen::Index rows, Eigen::Index cols) {
                auto* in_ptr  = reinterpret_cast<fftw_complex*>(const_cast<std::complex<double>*>(in.data()));
                auto* out_ptr = reinterpret_cast<fftw_complex*>(out.data());

                bool aligned = fftw_alignment_of(reinterpret_cast<double*>(in_ptr))  == plan_alignment
                            && fftw_alignment_of(reinterpret_cast<double*>(out_ptr)) == plan_alignment;

                if (aligned) {
                    fftw_execute_dft(plan, in_ptr, out_ptr);
                } else {
                    FFTWBuffer2D tmp_in(rows, cols), tmp_out(rows, cols);
                    tmp_in.as_eigen() = in;  // row-major → row-major aligned copy
                    fftw_execute_dft(plan, tmp_in.data(), tmp_out.data());
                    out = tmp_out.as_eigen();
                }
        }

        // Shared wisdom helpers — FFTW wisdom is global state; both FFT1D and FFT2D
        // route through here so they all share the same planner mutex.
        void loadWisdomImpl(const std::string& path) {
            std::lock_guard<std::mutex> lock(s_planner_mutex);
            fftw_import_wisdom_from_filename(path.c_str()); // returns 0 on missing file, silently ok
        }

        void saveWisdomImpl(const std::string& path) {
            std::lock_guard<std::mutex> lock(s_planner_mutex);
            if (!fftw_export_wisdom_to_filename(path.c_str()))
                throw std::runtime_error("Failed to save FFTW wisdom to: " + path);
        }

    } // anonymous namespace

    struct FFT1D::Impl {
        PlanPtr forward_plan;
        PlanPtr inverse_plan;
        Eigen::Index n;
        int alignment; // fftw_alignment_of at plan creating time
    };

    FFT1D::FFT1D(Eigen::Index n, PlanRigor rigor, bool normalize)
    : impl_(std::make_unique<Impl>()), normalize_(normalize) {
        // Always allocate temp buffers — passing nullptr is only valid with FFTW_ESTIMATE,
        // but conditioning on rigor is fragile.
        FFTWBuffer1D in_buf(n);
        FFTWBuffer1D out_buf(n);
        unsigned int flags = toFFTWFlag(rigor);

        std::lock_guard<std::mutex> lock(s_planner_mutex);
        impl_->forward_plan = PlanPtr(fftw_plan_dft_1d(static_cast<int>(n), in_buf.data(), out_buf.data(), FFTW_FORWARD,  flags));
        impl_->inverse_plan = PlanPtr(fftw_plan_dft_1d(static_cast<int>(n), in_buf.data(), out_buf.data(), FFTW_BACKWARD, flags));
        impl_->n = n;
        impl_->alignment = fftw_alignment_of(reinterpret_cast<double*>(in_buf.data()));

        if (!impl_->forward_plan || !impl_->inverse_plan)
            throw std::runtime_error("FFTW failed to create 1D plan.");
    }

    // Defined here so the compiler sees the complete Impl type when generating
    // the destructor and move operations — required for unique_ptr<Impl> to work.
    FFT1D::~FFT1D() = default;
    FFT1D::FFT1D(FFT1D&&) noexcept = default;
    FFT1D& FFT1D::operator=(FFT1D&&) noexcept = default;

    // Forwards and inverse transforms
    void FFT1D::forward(const Eigen::VectorXcd& in, Eigen::VectorXcd& out) const {
        if (in.size() != impl_->n || out.size() != impl_->n) throw std::invalid_argument("Buffer size mismatch.");
        execute_safe(impl_->forward_plan.get(), impl_->alignment, in, out);
    }
    
    void FFT1D::inverse(const Eigen::VectorXcd& in, Eigen::VectorXcd& out) const {
        if (in.size() != impl_->n || out.size() != impl_->n) throw std::invalid_argument("Buffer size mismatch.");
        execute_safe(impl_->inverse_plan.get(), impl_->alignment, in, out);
        if (normalize_) out /= (double) impl_->n;
    }

    void FFT1D::loadWisdom(const std::string& path) { loadWisdomImpl(path); }
    void FFT1D::saveWisdom(const std::string& path) { saveWisdomImpl(path); }

    // -----------------------------------------------------------------------
    // FFT2D
    // -----------------------------------------------------------------------

    struct FFT2D::Impl {
        PlanPtr forward_plan;
        PlanPtr inverse_plan;
        Eigen::Index rows;
        Eigen::Index cols;
        int alignment;
    };

    FFT2D::FFT2D(Eigen::Index rows, Eigen::Index cols, PlanRigor rigor, bool normalize)
    : impl_(std::make_unique<Impl>()), normalize_(normalize) {
        // Allocate aligned scratch buffers for planning — same reasoning as FFT1D.
        FFTWBuffer2D in_buf(rows, cols);
        FFTWBuffer2D out_buf(rows, cols);
        unsigned int flags = toFFTWFlag(rigor);

        std::lock_guard<std::mutex> lock(s_planner_mutex);
        impl_->forward_plan = PlanPtr(fftw_plan_dft_2d(
            static_cast<int>(rows), static_cast<int>(cols),
            in_buf.data(), out_buf.data(), FFTW_FORWARD, flags));
        impl_->inverse_plan = PlanPtr(fftw_plan_dft_2d(
            static_cast<int>(rows), static_cast<int>(cols),
            in_buf.data(), out_buf.data(), FFTW_BACKWARD, flags));
        impl_->rows = rows;
        impl_->cols = cols;
        impl_->alignment = fftw_alignment_of(reinterpret_cast<double*>(in_buf.data()));

        if (!impl_->forward_plan || !impl_->inverse_plan)
            throw std::runtime_error("FFTW failed to create 2D plan.");
    }

    FFT2D::~FFT2D() = default;
    FFT2D::FFT2D(FFT2D&&) noexcept = default;
    FFT2D& FFT2D::operator=(FFT2D&&) noexcept = default;

    // Row-major overloads — zero-copy when aligned, scratch fallback when not.
    void FFT2D::forward(const RowMatrixXcd& in, RowMatrixXcd& out) const {
        if (in.rows() != impl_->rows || in.cols() != impl_->cols ||
            out.rows() != impl_->rows || out.cols() != impl_->cols)
            throw std::invalid_argument("Matrix dimensions do not match the 2D plan.");
        execute_safe_2d(impl_->forward_plan.get(), impl_->alignment, in, out, impl_->rows, impl_->cols);
    }

    void FFT2D::inverse(const RowMatrixXcd& in, RowMatrixXcd& out) const {
        if (in.rows() != impl_->rows || in.cols() != impl_->cols ||
            out.rows() != impl_->rows || out.cols() != impl_->cols)
            throw std::invalid_argument("Matrix dimensions do not match the 2D plan.");
        execute_safe_2d(impl_->inverse_plan.get(), impl_->alignment, in, out, impl_->rows, impl_->cols);
        if (normalize_) out /= static_cast<double>(impl_->rows * impl_->cols);
    }

    // Column-major overloads — always copies due to layout conversion.
    void FFT2D::forward(const Eigen::MatrixXcd& in, Eigen::MatrixXcd& out) const {
        if (in.rows() != impl_->rows || in.cols() != impl_->cols ||
            out.rows() != impl_->rows || out.cols() != impl_->cols)
            throw std::invalid_argument("Matrix dimensions do not match the 2D plan.");
        FFTWBuffer2D tmp_in(impl_->rows, impl_->cols);
        FFTWBuffer2D tmp_out(impl_->rows, impl_->cols);
        tmp_in.as_eigen() = in;  // col-major → row-major aligned
        fftw_execute_dft(impl_->forward_plan.get(), tmp_in.data(), tmp_out.data());
        out = tmp_out.as_eigen(); // row-major aligned → col-major
    }

    void FFT2D::inverse(const Eigen::MatrixXcd& in, Eigen::MatrixXcd& out) const {
        if (in.rows() != impl_->rows || in.cols() != impl_->cols ||
            out.rows() != impl_->rows || out.cols() != impl_->cols)
            throw std::invalid_argument("Matrix dimensions do not match the 2D plan.");
        FFTWBuffer2D tmp_in(impl_->rows, impl_->cols);
        FFTWBuffer2D tmp_out(impl_->rows, impl_->cols);
        tmp_in.as_eigen() = in;
        fftw_execute_dft(impl_->inverse_plan.get(), tmp_in.data(), tmp_out.data());
        out = tmp_out.as_eigen();
        if (normalize_) out /= static_cast<double>(impl_->rows * impl_->cols);
    }

    void FFT2D::loadWisdom(const std::string& path) { loadWisdomImpl(path); }
    void FFT2D::saveWisdom(const std::string& path) { saveWisdomImpl(path); }

    // -----------------------------------------------------------------------
    // FFT3D
    // -----------------------------------------------------------------------

    struct FFT3D::Impl {
        PlanPtr forward_plan;
        PlanPtr inverse_plan;
        Eigen::Index depth;
        Eigen::Index rows;
        Eigen::Index cols;
        int alignment;
    };

    FFT3D::FFT3D(Eigen::Index depth, Eigen::Index rows, Eigen::Index cols,
                 PlanRigor rigor, bool normalize)
    : impl_(std::make_unique<Impl>()), normalize_(normalize) {
        FFTWBuffer3D in_buf(depth, rows, cols);
        FFTWBuffer3D out_buf(depth, rows, cols);
        unsigned int flags = toFFTWFlag(rigor);

        std::lock_guard<std::mutex> lock(s_planner_mutex);
        impl_->forward_plan = PlanPtr(fftw_plan_dft_3d(
            static_cast<int>(depth), static_cast<int>(rows), static_cast<int>(cols),
            in_buf.data(), out_buf.data(), FFTW_FORWARD, flags));
        impl_->inverse_plan = PlanPtr(fftw_plan_dft_3d(
            static_cast<int>(depth), static_cast<int>(rows), static_cast<int>(cols),
            in_buf.data(), out_buf.data(), FFTW_BACKWARD, flags));
        impl_->depth = depth;
        impl_->rows  = rows;
        impl_->cols  = cols;
        impl_->alignment = fftw_alignment_of(reinterpret_cast<double*>(in_buf.data()));

        if (!impl_->forward_plan || !impl_->inverse_plan)
            throw std::runtime_error("FFTW failed to create 3D plan.");
    }

    FFT3D::~FFT3D() = default;
    FFT3D::FFT3D(FFT3D&&) noexcept = default;
    FFT3D& FFT3D::operator=(FFT3D&&) noexcept = default;

    // FFTWBuffer3D always uses fftw_alloc_complex so alignment matches the plan.
    // The check + scratch fallback mirrors 1D/2D for consistency in case a buffer
    // is ever constructed from non-FFTW memory in future.
    static void execute_safe_3d(fftw_plan plan, int plan_alignment,
        const FFTWBuffer3D& in, FFTWBuffer3D& out) {
            auto* in_ptr  = const_cast<fftw_complex*>(in.data());
            auto* out_ptr = out.data();

            bool aligned = fftw_alignment_of(reinterpret_cast<double*>(in_ptr))  == plan_alignment
                        && fftw_alignment_of(reinterpret_cast<double*>(out_ptr)) == plan_alignment;

            if (aligned) {
                fftw_execute_dft(plan, in_ptr, out_ptr);
            } else {
                FFTWBuffer3D tmp_in(in.depth(), in.rows(), in.cols());
                FFTWBuffer3D tmp_out(in.depth(), in.rows(), in.cols());
                std::memcpy(tmp_in.data(), in.data(), in.size() * sizeof(fftw_complex));
                fftw_execute_dft(plan, tmp_in.data(), tmp_out.data());
                std::memcpy(out.data(), tmp_out.data(), out.size() * sizeof(fftw_complex));
            }
    }

    void FFT3D::forward(const FFTWBuffer3D& in, FFTWBuffer3D& out) const {
        if (in.depth() != impl_->depth || in.rows() != impl_->rows || in.cols() != impl_->cols ||
            out.depth() != impl_->depth || out.rows() != impl_->rows || out.cols() != impl_->cols)
            throw std::invalid_argument("Buffer dimensions do not match the 3D plan.");
        execute_safe_3d(impl_->forward_plan.get(), impl_->alignment, in, out);
    }

    void FFT3D::inverse(const FFTWBuffer3D& in, FFTWBuffer3D& out) const {
        if (in.depth() != impl_->depth || in.rows() != impl_->rows || in.cols() != impl_->cols ||
            out.depth() != impl_->depth || out.rows() != impl_->rows || out.cols() != impl_->cols)
            throw std::invalid_argument("Buffer dimensions do not match the 3D plan.");
        execute_safe_3d(impl_->inverse_plan.get(), impl_->alignment, in, out);
        if (normalize_) {
            auto* p = reinterpret_cast<std::complex<double>*>(out.data());
            double N = static_cast<double>(impl_->depth * impl_->rows * impl_->cols);
            for (Eigen::Index i = 0; i < out.size(); ++i)
                p[i] /= N;
        }
    }

    void FFT3D::loadWisdom(const std::string& path) { loadWisdomImpl(path); }
    void FFT3D::saveWisdom(const std::string& path) { saveWisdomImpl(path); }
}