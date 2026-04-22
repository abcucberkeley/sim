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
    using CdArray = nb::array<Cplx, nb::c_contig, nb::device::cpu>;
} // anonymous namespace