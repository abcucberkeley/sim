#ifndef SIRIUS_TIFF_IO_HPP
#define SIRIUS_TIFF_IO_HPP

#include <stdexcept>
#include <string>
#include <variant>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace sirius {

    // Row major so the inner most dim is contig maching tiff scan layyout
    template <typename T>
    using Image = Eigen::Tensor<T, 2, Eigen::RowMajor>;

    template <typename T>
    using ImageStack = Eigen::Tensor<T, 3, Eigen::RowMajor>;

    // Row-major matrix type returned by asMatrix/slice views. Exposes the
    // Eigen matrix API (isApprox, operator==, comma-init, block ops, etc.)
    template <typename T>
    using ImageMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Need to be able to dispatch the correct reader based on tiff data
    using AnyImageStack = std::variant<ImageStack<uint8_t>, ImageStack<int8_t>, ImageStack<uint16_t>, ImageStack<int16_t>,
                                        ImageStack<uint32_t>, ImageStack<int32_t>, ImageStack<float>, ImageStack<double>>;

    // Compression options for writing
    enum class TiffCompression {
        None,
        Lzw,
        Deflate // Often referred to as ZIP
    };

    // Zero-copy 2D dense-matrix view over an Image<T>.
    template <typename T>
    inline Eigen::Map<ImageMatrix<T>> asMatrix(Image<T>& img) {
        return {img.data(), img.dimension(0), img.dimension(1)};
    }
    template <typename T>
    inline Eigen::Map<const ImageMatrix<T>> asMatrix(const Image<T>& img) {
        return {img.data(), img.dimension(0), img.dimension(1)};
    }

    // Zero-copy 2D dense-matrix view over page z of an ImageStack<T>.
    template <typename T>
    inline Eigen::Map<ImageMatrix<T>> slice(ImageStack<T>& stack, Eigen::Index z) {
        const auto rows = stack.dimension(1);
        const auto cols = stack.dimension(2);
        return {stack.data() + z * rows * cols, rows, cols};
    }
    template <typename T>
    inline Eigen::Map<const ImageMatrix<T>> slice(const ImageStack<T>& stack, Eigen::Index z) {
        const auto rows = stack.dimension(1);
        const auto cols = stack.dimension(2);
        return {stack.data() + z * rows * cols, rows, cols};
    }

    template <typename T>
    Image<T> readTiff(const std::string& path);

    template <typename T>
    ImageStack<T> readTiffStack(const std::string& path);

    // Call the correct function based on tiff data type
    // Usage:
    //     std::visit([](auto& img) {}, readTiffStackAny("file.tiff"));
    // Note: it is more efficient to call the correct function
    // if the underlying data is already known or data will be recast downstream
    AnyImageStack readTiffStackAny(const std::string& path);

    template <typename T>
    void writeTiff(const std::string& path, const Image<T>& image,
                   TiffCompression comp = TiffCompression::None);

    template <typename T>
    void writeTiffStack(const std::string& path, const ImageStack<T>& stack,
                        TiffCompression comp = TiffCompression::None);

} // namespace sirius

#endif // SIRIUS_TIFF_IO_HPP