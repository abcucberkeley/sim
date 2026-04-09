#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdio>

#include "sirius/tiff_io.hpp"

using namespace sirius;

TEST_CASE("ImageStack memory layout and slicing", "[ImageStack]") {
    const Eigen::Index depth = 3;
    const Eigen::Index rows = 4;
    const Eigen::Index cols = 5;

    ImageStack<uint16_t> stack(depth, rows, cols);
    for (Eigen::Index z = 0; z < depth; z++) {
        for (Eigen::Index r = 0; r < rows; r++) {
            for (Eigen::Index c = 0; c < cols; c++) {
                stack(z, r, c) = static_cast<uint16_t>(z * 100 + r * 10 + c);
            }
        }
    }

    SECTION("Dimensions are correct") {
        REQUIRE(stack.depth() == depth);
        REQUIRE(stack.rows() == rows);
        REQUIRE(stack.cols() == cols);
        REQUIRE(stack.size() == depth * rows * cols);
        REQUIRE_FALSE(stack.empty());
    }

    SECTION("Slicing is correct") {
        auto slice1 = stack.slice(1);
        
        REQUIRE(slice1.rows() == rows);
        REQUIRE(slice1.cols() == cols);
        
        REQUIRE(slice1(0, 0) == 100);
        REQUIRE(slice1(3, 4) == 134);

        slice1(0, 0) = 999;
        REQUIRE(stack(1, 0, 0) == 999);
    }

}

TEST_CASE("TIFF Round-trip supports all compression modes", "[tiff][io]") {
    // GENERATE yields one value at a time. This TEST_CASE will execute 3 times total.
    auto compression = GENERATE(
        TiffCompression::None, 
        TiffCompression::Lzw, 
        TiffCompression::Deflate
    );

    // INFO prints only if the test fails
    INFO("Testing compression mode: " << static_cast<int>(compression));

    const std::string path = "test_compression_roundtrip.tiff";
    Image<float> original(32, 32);
    original.setRandom();

    // Read/write
    writeTiff(path, original, compression);
    auto loaded = readTiff<float>(path);

    REQUIRE(loaded.rows() == original.rows());
    REQUIRE(loaded.cols() == original.cols());
    
    REQUIRE(loaded.isApprox(original)); 

    std::remove(path.c_str());
}

TEST_CASE("Error handling is correct", "[tiff][io][error_handling]") {
    
    SECTION("Throws runtime_error on nonexistent file") {
        REQUIRE_THROWS_AS(readTiff<uint8_t>("does_not_exist_12345.tiff"), std::runtime_error);
    }

    SECTION("Writing empty stack") {
        ImageStack<float> empty_stack;
        
        REQUIRE_THROWS_WITH(
            writeTiffStack("empty.tiff", empty_stack), 
            "Cannot write empty stack"
        );
    }
}