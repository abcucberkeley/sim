// zarr v2 / v3 / N5 stores through TensorStore: write with every codec,
// inspect, read regions with type conversion, OME-NGFF pyramids and group
// metadata. Every case SKIPs when the build has no TensorStore.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "sirius/zarr_io.hpp"

#include "temp_path.hpp"

using namespace sirius;
namespace fs = std::filesystem;

namespace {

    // A store directory that is removed on scope exit.
    struct TempStore {
        fs::path path;
        explicit TempStore(const char* suffix) : path(test::uniqueTempPath("zarr", suffix)) {}
        ~TempStore() {
            std::error_code ec;
            fs::remove_all(path, ec);
        }
        std::string str() const { return path.string(); }
    };

    void requireTensorStore() {
        if (!zarrSupported()) SKIP("built without TensorStore");
    }

    // (t, c, z, y, x) = (2, 2, 5, 13, 17): small, prime-ish, distinct per voxel.
    const std::vector<Index> kShape{2, 2, 5, 13, 17};
    std::vector<float> testData() {
        std::vector<float> v(2 * 2 * 5 * 13 * 17);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i % 1009) * 0.5f;
        return v;
    }
    Index at(Index t, Index c, Index z, Index y, Index x) { return (((t * 2 + c) * 5 + z) * 13 + y) * 17 + x; }

} // namespace

TEST_CASE("zarr: unsupported build reports it", "[zarr]") {
    if (zarrSupported()) SKIP("TensorStore is available");
    REQUIRE_THROWS_AS(inspectZarr("/nonexistent"), std::runtime_error);
    REQUIRE_FALSE(isZarrStore("/nonexistent"));
}

TEST_CASE("zarr: write, inspect and read back with every driver and codec", "[zarr][io]") {
    requireTensorStore();
    const int version = GENERATE(2, 3, 0);
    const std::string codec = GENERATE(std::string("none"), std::string("blosc-zstd"), std::string("blosc-lz4"),
                                       std::string("zstd"), std::string("gzip"));
    INFO("version=" << version << " codec=" << codec);
    TempStore s(version == 0 ? ".n5" : ".zarr");
    const std::vector<float> data = testData();

    ZarrWriteOptions o;
    o.zarrVersion = version;
    o.codec = codec;
    o.level = 3;
    o.chunks = {1, 1, 2, 8, 8};
    o.omeNgff = false;
    double last = 0.0;
    writeZarr<float>(s.str(), data.data(), kShape, o, [&](double p) { last = p; });
    REQUIRE(last == 1.0);
    REQUIRE(isZarrStore(s.str()));

    const ZarrArrayInfo info = inspectZarr(s.str());
    REQUIRE(info.driver == (version == 0 ? "n5" : version == 2 ? "zarr" : "zarr3"));
    REQUIRE(info.shape == kShape);
    REQUIRE(info.chunks == std::vector<Index>{1, 1, 2, 8, 8});
    REQUIRE(info.pixelType == PixelType::Float32);
    REQUIRE_FALSE(info.isGroup);
    REQUIRE(info.bytesOnDisk > 0);
    if (codec == "none") REQUIRE((info.codec == "none" || info.codec == "raw"));
    else if (codec == "blosc-zstd") REQUIRE(info.codec == "blosc(zstd,3)");
    else if (codec == "gzip") REQUIRE(info.codec.rfind("gzip", 0) == 0);

    ZarrArray a(s.str());
    const Buffer<float> full = a.read<float>({0, 0, 0, 0, 0}, {0, 0, 0, 0, 0});
    REQUIRE(full.shape() == Shape{2 * 2 * 5, 13, 17});   // rank 5 collapses to (planes, y, x)
    for (std::size_t i = 0; i < data.size(); ++i)
        if (full.data()[i] != data[i]) FAIL("mismatch at " << i);

    // a region with conversion to double
    const Buffer<double> region = a.read<double>({1, 0, 2, 3, 5}, {1, 2, 2, 6, 9});
    REQUIRE(region.shape() == Shape{4, 6, 9});
    for (Index c = 0; c < 2; ++c)
        for (Index z = 0; z < 2; ++z)
            for (Index y = 0; y < 6; ++y)
                for (Index x = 0; x < 9; ++x) {
                    const double expected = data[static_cast<std::size_t>(at(1, c, 2 + z, 3 + y, 5 + x))];
                    const double got = region.data()[((c * 2 + z) * 6 + y) * 9 + x];
                    if (got != expected) FAIL("region mismatch at c" << c << " z" << z << " y" << y << " x" << x);
                }
}

TEST_CASE("zarr: N5 stores list dimensions in reversed order on disk", "[zarr][io][n5]") {
    requireTensorStore();
    TempStore s(".n5");
    const std::vector<float> data = testData();
    ZarrWriteOptions o;
    o.zarrVersion = 0;
    o.omeNgff = false;
    o.chunks = {1, 1, 5, 13, 17};
    writeZarr<float>(s.str(), data.data(), kShape, o);
    std::ifstream in(s.path / "attributes.json");
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    REQUIRE(text.find("[17,13,5,2,2]") != std::string::npos);   // x fastest, as N5 tools expect
    REQUIRE(inspectZarr(s.str()).shape == kShape);               // but C order for us
}

TEST_CASE("zarr: every pixel type round-trips and reads convert", "[zarr][io]") {
    requireTensorStore();
    const int version = GENERATE(2, 3);
    TempStore s(".zarr");
    const std::vector<Index> shape{3, 4, 5};
    auto check = [&](auto tag) {
        using T = decltype(tag);
        std::vector<T> v(60);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<T>(i % 100);
        ZarrWriteOptions o;
        o.zarrVersion = version;
        o.omeNgff = false;
        o.codec = "zstd";
        writeZarr<T>(s.str(), v.data(), shape, o);
        const ZarrArrayInfo info = inspectZarr(s.str());
        REQUIRE(info.pixelType == pixelTypeOf<T>());
        const Buffer<T> back = readZarr<T>(s.str(), {0, 0, 0}, {3, 4, 5});
        for (std::size_t i = 0; i < v.size(); ++i)
            if (back.data()[i] != v[i]) FAIL("mismatch at " << i);
        const Buffer<float> asFloat = readZarr<float>(s.str(), {1, 0, 0}, {2, 4, 5});
        REQUIRE(asFloat.data()[0] == static_cast<float>(v[20]));
    };
    check(std::uint8_t{});
    check(std::int8_t{});
    check(std::uint16_t{});
    check(std::int16_t{});
    check(std::uint32_t{});
    check(std::int32_t{});
    check(float{});
    check(double{});
}

TEST_CASE("zarr: OME-NGFF group with a pyramid, axes, scale and channels", "[zarr][io][ngff]") {
    requireTensorStore();
    const int version = GENERATE(2, 3, 0);
    INFO("version=" << version);
    TempStore s(version == 0 ? ".n5" : ".zarr");
    const std::vector<float> data = testData();
    ZarrWriteOptions o;
    o.zarrVersion = version;
    o.chunks = {1, 1, 5, 8, 8};
    o.axes = {"t", "c", "z", "y", "x"};
    o.scale = {1.0, 1.0, 0.3, 0.1, 0.1};
    o.channelNames = {"DAPI", "GFP"};
    o.channelColors = {"#7c9cff", "#63e08a"};
    o.pyramidLevels = 3;
    o.downsample = 2;
    writeZarr<float>(s.str(), data.data(), kShape, o);

    const ZarrArrayInfo info = inspectZarr(s.str());
    REQUIRE(info.isGroup);
    REQUIRE(info.levelPath == "0");
    REQUIRE(info.axes == std::vector<std::string>{"t", "c", "z", "y", "x"});
    REQUIRE(info.axisTypes[0] == "time");
    REQUIRE(info.axisTypes[1] == "channel");
    REQUIRE(info.axisTypes[2] == "space");
    REQUIRE(info.scale == std::vector<double>{1.0, 1.0, 0.3, 0.1, 0.1});
    REQUIRE(info.channelNames == std::vector<std::string>{"DAPI", "GFP"});
    REQUIRE(info.channelColors == std::vector<std::string>{"#7c9cff", "#63e08a"});
    REQUIRE(info.multiscalePaths == std::vector<std::string>{"0", "1", "2"});
    REQUIRE(info.shape == kShape);

    // level 1 halves y and x only (z stays), level 2 halves again
    ZarrArray l1(s.str(), "1");
    REQUIRE(l1.info().shape == std::vector<Index>{2, 2, 5, 7, 9});
    ZarrArray l2(s.str(), "2");
    REQUIRE(l2.info().shape == std::vector<Index>{2, 2, 5, 4, 5});
    // level 1 voxel (0,0,0,0,0) is the mean of the 2x2 box at the origin
    const Buffer<float> v = l1.read<float>({0, 0, 0, 0, 0}, {1, 1, 1, 1, 1});
    const float expected = (data[static_cast<std::size_t>(at(0, 0, 0, 0, 0))] + data[static_cast<std::size_t>(at(0, 0, 0, 0, 1))] +
                            data[static_cast<std::size_t>(at(0, 0, 0, 1, 0))] + data[static_cast<std::size_t>(at(0, 0, 0, 1, 1))]) / 4.0f;
    REQUIRE_THAT(v.data()[0], Catch::Matchers::WithinRel(expected, 1e-6f));

    // the group metadata files exist where the spec puts them
    if (version == 3) REQUIRE(fs::exists(s.path / "zarr.json"));
    else if (version == 2) {
        REQUIRE(fs::exists(s.path / ".zgroup"));
        REQUIRE(fs::exists(s.path / ".zattrs"));
    } else REQUIRE(fs::exists(s.path / "attributes.json"));
}

TEST_CASE("zarr: zarr v3 sharding groups chunks", "[zarr][io][shard]") {
    requireTensorStore();
    TempStore s(".zarr");
    const std::vector<float> data = testData();
    ZarrWriteOptions o;
    o.zarrVersion = 3;
    o.chunks = {1, 1, 1, 4, 4};
    o.shard = true;
    o.shardFactor = 4;
    o.omeNgff = false;
    writeZarr<float>(s.str(), data.data(), kShape, o);
    const ZarrArrayInfo info = inspectZarr(s.str());
    REQUIRE(info.codec.find("sharded") != std::string::npos);
    REQUIRE(info.chunks == std::vector<Index>{1, 1, 1, 4, 4});
    const Buffer<float> back = readZarr<float>(s.str(), {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0});
    for (std::size_t i = 0; i < data.size(); ++i)
        if (back.data()[i] != data[i]) FAIL("mismatch at " << i);
    // far fewer files than chunks
    std::size_t files = 0;
    for (const auto& e : fs::recursive_directory_iterator(s.path))
        if (e.is_regular_file()) ++files;
    REQUIRE(files < 2 * 2 * 5 * 4 * 5);
}

TEST_CASE("zarr: errors are clear", "[zarr][io][error]") {
    requireTensorStore();
    TempStore s(".zarr");
    REQUIRE_THROWS_AS(inspectZarr(s.str()), std::runtime_error);
    REQUIRE_FALSE(isZarrStore(s.str()));

    const std::vector<float> data = testData();
    ZarrWriteOptions o;
    o.omeNgff = false;
    writeZarr<float>(s.str(), data.data(), kShape, o);
    ZarrArray a(s.str());
    REQUIRE_THROWS_AS(a.read<float>({0, 0, 0, 0, 0}, {3, 0, 0, 0, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(a.read<float>({0, 0}, {1, 1}), std::invalid_argument);
    REQUIRE_THROWS_AS(inspectZarr(s.str() + "/missing"), std::runtime_error);

    // refuses to clobber a directory that is not a store
    TempStore d(".dir");
    fs::create_directories(d.path);
    std::ofstream(d.path / "keep.txt") << "x";
    REQUIRE_THROWS_AS(writeZarr<float>(d.str(), data.data(), kShape, o), std::runtime_error);
    REQUIRE(fs::exists(d.path / "keep.txt"));
}
