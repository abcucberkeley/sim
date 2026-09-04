#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_device(nb::module_&);
void bind_buffer(nb::module_&);
void bind_fft(nb::module_&);
void bind_tiff_io(nb::module_&);
void bind_sim(nb::module_&);
void bind_registration(nb::module_&);

NB_MODULE(_sirius_ext, m){
    m.doc() = "SIRIUS - Structured Illumination Reconstruction and Image Utility Suite";

    bind_device(m);
    bind_buffer(m);
    bind_fft(m);
    bind_tiff_io(m);
    bind_sim(m);
    bind_registration(m);
}
