#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sirius/fft.hpp>

#include <complex>
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using sirius::FFT;
using sirius::PlanRigor;

namespace {

    using Cplx = std::complex<double>;

    // Writable C-contig complex128 view. We don't pin the rank at the binding
    // level — the same FFT instance accepts whichever shape matches the dims
    // it was planned for (flat 1-D, {rows,cols}, {depth,rows,cols}, or any
    // reshape of equivalent total size).
    using CdArray = nb::ndarray<Cplx, nb::c_contig, nb::device::cpu>;

    std::size_t totalElements(const CdArray& a) {
        std::size_t n = 1;
        for (std::size_t i = 0; i < a.ndim(); ++i) n *= a.shape(i);
        return n;
    }

    void checkSize(const CdArray& a, std::size_t expected, const char* name) {
        const std::size_t got = totalElements(a);
        if (got != expected)
            throw std::invalid_argument(
                std::string(name) + " has " + std::to_string(got) +
                " complex elements, expected " + std::to_string(expected));
    }

    // Fresh numpy array with the same shape as `like`. Buffer is heap-owned
    // by a capsule that frees it on Python GC.
    nb::ndarray<nb::numpy, Cplx> emptyLike(const CdArray& like) {
        const std::size_t ndim = like.ndim();
        std::vector<std::size_t> shape(ndim);
        std::size_t total = 1;
        for (std::size_t i = 0; i < ndim; ++i) {
            shape[i] = like.shape(i);
            total *= shape[i];
        }
        auto* data = new Cplx[total];
        nb::capsule owner(data, [](void* p) noexcept {
            delete[] static_cast<Cplx*>(p);
        });
        return nb::ndarray<nb::numpy, Cplx>(data, ndim, shape.data(), owner);
    }

    // Python-facing wrapper. Owns the planned FFT plus the size metadata the
    // C++ class deliberately doesn't expose, so we can reject buffer-size
    // mismatches up-front instead of trusting the caller to match dims*howmany.
    class PyFFT {
    public:
        PyFFT(std::vector<int> dims, int howmany, PlanRigor rigor)
            : total_(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>{}))
            , full_(total_ * howmany)
            , fft_(std::move(dims), howmany, rigor) {}

        nb::ndarray<nb::numpy, Cplx> fft(CdArray in) const {
            checkSize(in, static_cast<std::size_t>(full_), "in");
            auto out = emptyLike(in);
            fft_.fft(in.data(), out.data());
            return out;
        }

        void fft_into(CdArray in, CdArray out) const {
            checkSize(in,  static_cast<std::size_t>(full_), "in");
            checkSize(out, static_cast<std::size_t>(full_), "out");
            fft_.fft(in.data(), out.data());
        }

        nb::ndarray<nb::numpy, Cplx> ifft(CdArray in, bool normalize) const {
            checkSize(in, static_cast<std::size_t>(full_), "in");
            auto out = emptyLike(in);
            fft_.ifft(in.data(), out.data());
            if (normalize) scaleByInvN(out.data());
            return out;
        }

        void ifft_into(CdArray in, CdArray out, bool normalize) const {
            checkSize(in,  static_cast<std::size_t>(full_), "in");
            checkSize(out, static_cast<std::size_t>(full_), "out");
            fft_.ifft(in.data(), out.data());
            if (normalize) scaleByInvN(out.data());
        }

    private:
        // Divide every element by total_ (product of dims) — matches the
        // behavior of sirius::FFT::ifft<Rank>(..., normalize=true).
        void scaleByInvN(Cplx* p) const {
            const double s = 1.0 / static_cast<double>(total_);
            const auto n = static_cast<std::size_t>(full_);
            for (std::size_t i = 0; i < n; ++i) p[i] *= s;
        }

        int total_;
        int full_;
        FFT fft_;
    };

} // namespace

void bind_fft(nb::module_& m) {
    nb::enum_<PlanRigor>(m, "PlanRigor",
            "Planning rigor for FFTW. Trades one-time planning cost for "
            "runtime speed.")
        .value("Estimate",   PlanRigor::Estimate)
        .value("Measure",    PlanRigor::Measure)
        .value("Patient",    PlanRigor::Patient)
        .value("Exhaustive", PlanRigor::Exhaustive)
        .export_values();

    nb::class_<PyFFT>(m, "FFT",
            "Planned FFT over complex128 arrays. Supports 1D, 2D, and 3D "
            "transforms and batched execution via `howmany`. Both `fft` and "
            "`ifft` come in allocating and in-place variants:\n\n"
            "    f = FFT([8, 8])\n"
            "    y = f.fft(x)              # new array, shape matches x\n"
            "    f.fft(x, out=y)           # no allocation\n"
            "    x_back = f.ifft(y, normalize=True)\n")
        .def(nb::init<std::vector<int>, int, PlanRigor>(),
             nb::arg("dims"),
             nb::arg("howmany") = 1,
             nb::arg("rigor") = PlanRigor::Measure,
             "Plan an FFT for `dims` ([n], [rows, cols], or [depth, rows, cols]) "
             "and `howmany` batched transforms. Construction runs FFTW's planner, "
             "which can take a moment at higher rigor levels.")

        .def("fft", &PyFFT::fft,
             nb::arg("in"),
             "Forward transform. Returns a new numpy array with the same shape as `in`.")
        .def("fft", &PyFFT::fft_into,
             nb::arg("in"), nb::arg("out"),
             "Forward transform into a preallocated output array. `out` must be "
             "complex128, C-contiguous, and have the same total element count as `in`.")

        .def("ifft", &PyFFT::ifft,
             nb::arg("in"), nb::arg("normalize") = false,
             "Inverse transform. When `normalize=True`, the result is divided by the "
             "product of `dims`, so `ifft(fft(x), normalize=True)` recovers `x`.")
        .def("ifft", &PyFFT::ifft_into,
             nb::arg("in"), nb::arg("out"), nb::arg("normalize") = false,
             "Inverse transform into a preallocated output array of matching size.")

        .def_static("load_wisdom", &FFT::loadWisdom, nb::arg("path"),
                    "Import FFTW wisdom from a file. A missing file is silently ignored.")
        .def_static("save_wisdom", &FFT::saveWisdom, nb::arg("path"),
                    "Export accumulated FFTW wisdom to a file.");
}