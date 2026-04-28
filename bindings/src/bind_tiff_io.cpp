#include <nanobind/nanobind.h>
#include <nanobind/eigen/tensor.h>
#include <nanobind/stl/string.h>

#include <sirius/tiff_io.hpp>

namespace nb = nanobind;

using sirius::TiffCompression;
using sirius::readTiffStackAny;
using sirius::writeTiff;
using sirius::writeTiffStack;

namespace {
    template <typename ImgT>
    using nparray = nb::ndarray<nb::numpy, typename ImgT::Scalar, nb::ndim<ImgT::NumDimensions>, nb::c_contig>;
    
    template <typename ImgT>
    nparray<ImgT> eigenTensorToNumpy(ImgT img) {
        using T = typename ImgT::Scalar;
        constexpr size_t Rank = ImgT::NumDimensions;

        auto* ptr = new ImgT(std::move(img));

        nb::capsule owner(ptr, [](void* p) noexcept {
            delete static_cast<ImgT*>(p);
        });

        std::array<size_t, Rank> shape;
        for (size_t i = 0; i < Rank; ++i)
            shape[i] = static_cast<size_t>(ptr->dimension(i));

        return nparray<ImgT>(
            ptr->data(), Rank, shape.data(), owner
        );
    }

    nb::object readTiffStackNumpy(const std::string& path) {
        return std::visit(
            [](auto img) -> nb::object {
                return nb::cast(eigenTensorToNumpy(std::move(img)));
            },
            readTiffStackAny(path)
        );
    }
} // anonymous namespace

void bind_tiff_io(nb::module_& m) {
    m.def("read_tiff", &readTiffStackNumpy, nb::arg("path"));

    nb::enum_<TiffCompression>(m, "TiffCompression",
        "Write tiff compression options.")
        .value("NoCompression", TiffCompression::None)
        .value("Lzw",  TiffCompression::Lzw)
        .value("Deflate", TiffCompression::Deflate);
    
    m.def("write_tiff", &writeTiff<int8_t>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
    m.def("write_tiff", &writeTiff<uint8_t>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
    m.def("write_tiff", &writeTiff<int16_t>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
    m.def("write_tiff", &writeTiff<uint16_t>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
    m.def("write_tiff", &writeTiff<int32_t>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
    m.def("write_tiff", &writeTiff<uint32_t>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
    m.def("write_tiff", &writeTiff<float>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
    m.def("write_tiff", &writeTiff<double>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);

    m.def("write_tiff", &writeTiffStack<int8_t>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
    m.def("write_tiff", &writeTiffStack<uint8_t>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
    m.def("write_tiff", &writeTiffStack<int16_t>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
    m.def("write_tiff", &writeTiffStack<uint16_t>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
    m.def("write_tiff", &writeTiffStack<int32_t>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
    m.def("write_tiff", &writeTiffStack<uint32_t>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
    m.def("write_tiff", &writeTiffStack<float>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
    m.def("write_tiff", &writeTiffStack<double>, nb::arg("path"), nb::arg("image"), nb::arg("comp") = TiffCompression::None);
}
