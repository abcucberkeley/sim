#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_fft(nb::module_&);
void bind_tiff_io(nb::module_&);

NB_MODULE(_sirius_ext, m){
    m.doc() = "SIRIUS - Structured Illumination Reconstruction and Image Utility Suite";

    bind_fft(m);
    bind_tiff_io(m);
}