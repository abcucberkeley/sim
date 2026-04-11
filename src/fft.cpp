#include "sirius/fft.hpp"
#include "sirius/fft_buffers.hpp"
#include <Eigen/Core>
#include <fftw3.h>
#include <mutex>
#include <memory>
#include <stdexcept>

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

    } // anonymous namespace

    struct FFT1D::Impl {
        PlanPtr forward_plan;
        PlanPtr inverse_plan;
        Eigen::Index n;
        int alignment; // fftw_alignment_of at plan creating time
    };

    FFT1D::FFT1D(Eigen::Index n, PlanRigor rigor) : impl_(std::make_unique<Impl>()) {
        // Always allocate temp buffers — passing nullptr is only valid with FFTW_ESTIMATE,
        // but conditioning on rigor is fragile.
        FFTWBuffer1D in_buf(n);
        FFTWBuffer1D out_buf(n);
        unsigned int flags = toFFTWFlag(rigor);

        std::lock_guard<std::mutex> lock(s_planner_mutex);
        impl_->forward_plan = PlanPtr(fftw_plan_dft_1d(n, in_buf.data(), out_buf.data(), FFTW_FORWARD,  flags));
        impl_->inverse_plan = PlanPtr(fftw_plan_dft_1d(n, in_buf.data(), out_buf.data(), FFTW_BACKWARD, flags));
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

    void FFT1D::forward(const Eigen::VectorXcd& in, Eigen::VectorXcd& out) const {
        if (in.size() != impl_->n || out.size() != impl_->n) throw std::invalid_argument("Buffer size mismatch.");
        execute_safe(impl_->forward_plan.get(), impl_->alignment, in, out);
    }

    void FFT1D::inverse(const Eigen::VectorXcd& in, Eigen::VectorXcd& out) const {
        if (in.size() != impl_->n || out.size() != impl_->n) throw std::invalid_argument("Buffer size mismatch.");
        execute_safe(impl_->inverse_plan.get(), impl_->alignment, in, out);
    }
}