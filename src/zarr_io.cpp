// zarr v2 / zarr v3 / N5 stores through TensorStore.
//
// Store discovery (which metadata file is present, OME-NGFF multiscales,
// axes, channel names) is plain filesystem + JSON and is compiled always, so
// isZarrStore() works in every build; the pixel paths need TensorStore and
// throw a clear error without it.

#include "sirius/zarr_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <system_error>

#include <nlohmann/json.hpp>

#ifdef SIRIUS_HAS_TENSORSTORE
#include "tensorstore/array.h"
#include "tensorstore/cast.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/context.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#endif

namespace sirius {

    namespace fs = std::filesystem;

// Most of the helpers below serve the TensorStore path; without it only the
// store discovery is used, so silence the unused-function warnings there.
#if !defined(SIRIUS_HAS_TENSORSTORE) && defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
    using json = nlohmann::json;

    // --- metadata discovery (no TensorStore needed) ---------------------------

    namespace {

        json readJsonFile(const fs::path& p) {
            std::ifstream in(p);
            if (!in) throw std::runtime_error("cannot read " + p.string());
            json j;
            try {
                in >> j;
            } catch (const std::exception& e) {
                throw std::runtime_error("malformed JSON in " + p.string() + ": " + e.what());
            }
            return j;
        }

        void writeJsonFile(const fs::path& p, const json& j) {
            std::ofstream out(p);
            if (!out) throw std::runtime_error("cannot write " + p.string());
            out << j.dump(2) << "\n";
        }

        // What a directory is, as far as its metadata files say.
        struct StoreLayout {
            std::string driver;                 // "zarr", "zarr3", "n5"
            bool isArray = false;
            bool isGroup = false;
            json attributes;                    // group / array attributes (zarr: .zattrs or zarr.json attributes)
        };

        std::optional<StoreLayout> discover(const fs::path& dir) {
            std::error_code ec;
            if (!fs::is_directory(dir, ec)) return std::nullopt;
            if (fs::exists(dir / "zarr.json", ec)) {
                StoreLayout l;
                l.driver = "zarr3";
                const json j = readJsonFile(dir / "zarr.json");
                const std::string node = j.value("node_type", "array");
                l.isArray = node == "array";
                l.isGroup = node == "group";
                l.attributes = j.value("attributes", json::object());
                return l;
            }
            if (fs::exists(dir / ".zarray", ec) || fs::exists(dir / ".zgroup", ec)) {
                StoreLayout l;
                l.driver = "zarr";
                l.isArray = fs::exists(dir / ".zarray", ec);
                l.isGroup = !l.isArray;
                if (fs::exists(dir / ".zattrs", ec)) l.attributes = readJsonFile(dir / ".zattrs");
                return l;
            }
            if (fs::exists(dir / "attributes.json", ec)) {
                StoreLayout l;
                l.driver = "n5";
                const json j = readJsonFile(dir / "attributes.json");
                l.isArray = j.contains("dimensions");
                l.isGroup = !l.isArray;
                l.attributes = j;
                return l;
            }
            // A bare directory of numbered datasets ("0", "1", ...) without
            // group metadata: treat as a group when "0" is an array.
            if (fs::is_directory(dir / "0", ec)) {
                if (auto sub = discover(dir / "0"); sub && sub->isArray) {
                    StoreLayout l;
                    l.driver = sub->driver;
                    l.isGroup = true;
                    return l;
                }
            }
            return std::nullopt;
        }

        // OME-NGFF multiscales: 0.4 puts them at the top level of the
        // attributes, 0.5 under "ome".
        const json* multiscalesOf(const json& attributes) {
            if (attributes.contains("ome") && attributes["ome"].is_object() && attributes["ome"].contains("multiscales"))
                return &attributes["ome"]["multiscales"];
            if (attributes.contains("multiscales")) return &attributes["multiscales"];
            return nullptr;
        }

        void fillNgff(ZarrArrayInfo& info, const json& attributes) {
            const json* ms = multiscalesOf(attributes);
            if (ms && ms->is_array() && !ms->empty()) {
                const json& m = ms->front();
                if (m.contains("axes") && m["axes"].is_array()) {
                    for (const json& a : m["axes"]) {
                        if (a.is_string()) {
                            info.axes.push_back(a.get<std::string>());
                            info.axisTypes.push_back("");
                        } else if (a.is_object()) {
                            info.axes.push_back(a.value("name", ""));
                            info.axisTypes.push_back(a.value("type", ""));
                        }
                    }
                }
                if (m.contains("datasets") && m["datasets"].is_array()) {
                    for (const json& d : m["datasets"]) {
                        if (!d.is_object() || !d.contains("path")) continue;
                        info.multiscalePaths.push_back(d["path"].get<std::string>());
                        if (info.scale.empty() && d.contains("coordinateTransformations"))
                            for (const json& t : d["coordinateTransformations"])
                                if (t.is_object() && t.value("type", "") == "scale" && t.contains("scale"))
                                    info.scale = t["scale"].get<std::vector<double>>();
                    }
                }
            }
            const json* omero = nullptr;
            if (attributes.contains("omero")) omero = &attributes["omero"];
            else if (attributes.contains("ome") && attributes["ome"].is_object() && attributes["ome"].contains("omero"))
                omero = &attributes["ome"]["omero"];
            if (omero && omero->is_object() && omero->contains("channels") && (*omero)["channels"].is_array()) {
                for (const json& c : (*omero)["channels"]) {
                    if (!c.is_object()) continue;
                    info.channelNames.push_back(c.value("label", ""));
                    std::string color = c.value("color", "");
                    if (!color.empty() && color[0] != '#') color = "#" + color;
                    info.channelColors.push_back(color);
                }
            }
        }

        std::uint64_t directoryBytes(const fs::path& dir) {
            std::uint64_t total = 0;
            std::error_code ec;
            for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
                if (it->is_regular_file(ec)) total += static_cast<std::uint64_t>(it->file_size(ec));
            }
            return total;
        }

        // Resolve `path` (+ optional level) to the directory of the array to open.
        struct Resolved {
            fs::path root;          // the store the user named
            fs::path arrayDir;      // the array to open
            StoreLayout rootLayout;
            std::string levelPath;
            bool isGroup = false;
        };

        Resolved resolveArray(const std::string& path, const std::string& levelPath) {
            Resolved r;
            r.root = fs::path(path);
            auto layout = discover(r.root);
            if (!layout) throw std::runtime_error("not a zarr / N5 store (no zarr.json, .zarray, .zgroup or attributes.json): " + path);
            r.rootLayout = *layout;
            if (layout->isArray && levelPath.empty()) {
                r.arrayDir = r.root;
                return r;
            }
            r.isGroup = true;
            std::string level = levelPath;
            if (level.empty()) {
                ZarrArrayInfo tmp;
                fillNgff(tmp, layout->attributes);
                level = tmp.multiscalePaths.empty() ? "0" : tmp.multiscalePaths.front();
            }
            r.levelPath = level;
            r.arrayDir = r.root / level;
            auto sub = discover(r.arrayDir);
            if (!sub || !sub->isArray)
                throw std::runtime_error("zarr / N5 group has no array at '" + level + "': " + path);
            if (r.rootLayout.driver.empty()) r.rootLayout.driver = sub->driver;
            return r;
        }

        PixelType pixelTypeFromName(const std::string& name) {
            if (name == "uint8") return PixelType::UInt8;
            if (name == "int8") return PixelType::Int8;
            if (name == "uint16") return PixelType::UInt16;
            if (name == "int16") return PixelType::Int16;
            if (name == "uint32" || name == "uint64") return PixelType::UInt32;
            if (name == "int32" || name == "int64") return PixelType::Int32;
            if (name == "float32" || name == "float16" || name == "bfloat16") return PixelType::Float32;
            if (name == "float64") return PixelType::Float64;
            throw std::runtime_error("unsupported zarr data type '" + name + "'");
        }

        const char* dtypeName(PixelType t) noexcept {
            return toString(t);   // "uint8" ... "float64": the names TensorStore uses
        }

        std::string describeCodec(const json& metadata, const std::string& driver) {
            auto level = [](const json& c, const char* key) -> std::string {
                if (c.contains(key) && c[key].is_number()) return "," + std::to_string(c[key].get<int>());
                return "";
            };
            if (driver == "zarr") {
                if (!metadata.contains("compressor") || metadata["compressor"].is_null()) return "none";
                const json& c = metadata["compressor"];
                const std::string id = c.value("id", "?");
                if (id == "blosc") return "blosc(" + c.value("cname", "lz4") + level(c, "clevel") + ")";
                // numcodecs' "zlib" is what everyone else calls gzip
                const std::string name = id == "zlib" ? "gzip" : id;
                return name + "(" + (c.contains("level") ? std::to_string(c["level"].get<int>()) : std::string()) + ")";
            }
            if (driver == "zarr3") {
                std::string out;
                std::function<void(const json&)> walk = [&](const json& codecs) {
                    if (!codecs.is_array()) return;
                    for (const json& c : codecs) {
                        const std::string name = c.value("name", "");
                        const json cfg = c.value("configuration", json::object());
                        if (name == "bytes" || name == "transpose") continue;
                        if (name == "sharding_indexed") {
                            walk(cfg.value("codecs", json::array()));
                            out += (out.empty() ? "" : "+") + std::string("sharded");
                            continue;
                        }
                        std::string piece = name;
                        if (name == "blosc") piece += "(" + cfg.value("cname", "lz4") + level(cfg, "clevel") + ")";
                        else if (cfg.contains("level")) piece += "(" + std::to_string(cfg["level"].get<int>()) + ")";
                        out += (out.empty() ? "" : "+") + piece;
                    }
                };
                walk(metadata.value("codecs", json::array()));
                return out.empty() ? "none" : out;
            }
            if (driver == "n5") {
                if (!metadata.contains("compression")) return "raw";
                const json& c = metadata["compression"];
                const std::string type = c.value("type", "raw");
                if (type == "blosc") return "blosc(" + c.value("cname", "lz4") + level(c, "clevel") + ")";
                if (type == "gzip") return "gzip" + (c.contains("level") ? "(" + std::to_string(c["level"].get<int>()) + ")" : std::string());
                return type;
            }
            return "?";
        }

        std::vector<std::string> defaultAxes(int rank) {
            static const char* names[] = {"t", "c", "z", "y", "x"};
            std::vector<std::string> out;
            for (int i = 5 - rank; i < 5; ++i) out.emplace_back(names[i]);
            return out;
        }

        json ngffAxes(const std::vector<std::string>& axes) {
            json out = json::array();
            for (const std::string& a : axes) {
                json ax = {{"name", a}};
                if (a == "t") { ax["type"] = "time"; ax["unit"] = "second"; }
                else if (a == "c") ax["type"] = "channel";
                else { ax["type"] = "space"; ax["unit"] = "micrometer"; }
                out.push_back(ax);
            }
            return out;
        }

    } // namespace

#if !defined(SIRIUS_HAS_TENSORSTORE) && defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

    bool isZarrStore(const std::string& path) noexcept {
        try {
            return discover(fs::path(path)).has_value();
        } catch (...) {
            return false;
        }
    }

#ifndef SIRIUS_HAS_TENSORSTORE

    bool zarrSupported() noexcept { return false; }

    namespace {
        [[noreturn]] void noTensorStore() { throw std::runtime_error("built without TensorStore (SIRIUS_ENABLE_TENSORSTORE=OFF)"); }
    }

    ZarrArrayInfo inspectZarr(const std::string&) { noTensorStore(); }

    struct ZarrArray::Impl {};
    ZarrArray::ZarrArray(const std::string&, const std::string&) { noTensorStore(); }
    ZarrArray::~ZarrArray() = default;
    ZarrArray::ZarrArray(ZarrArray&&) noexcept = default;
    ZarrArray& ZarrArray::operator=(ZarrArray&&) noexcept = default;
    const ZarrArrayInfo& ZarrArray::info() const noexcept {
        static const ZarrArrayInfo empty;
        return empty;
    }
    template <typename T>
    void ZarrArray::read(const std::vector<Index>&, const std::vector<Index>&, T*) const { noTensorStore(); }
    template <typename T>
    Buffer<T> ZarrArray::read(const std::vector<Index>&, const std::vector<Index>&) const { noTensorStore(); }
    template <typename T>
    Buffer<T> readZarr(const std::string&, const std::vector<Index>&, const std::vector<Index>&, const std::string&) {
        noTensorStore();
    }
    template <typename T>
    void writeZarr(const std::string&, const T*, const std::vector<Index>&, const ZarrWriteOptions&,
                   const std::function<void(double)>&) {
        noTensorStore();
    }

#else

    bool zarrSupported() noexcept { return true; }

    namespace {

        namespace ts = tensorstore;

        [[noreturn]] void throwStatus(const absl::Status& s, const std::string& what) {
            throw std::runtime_error(what + ": " + s.ToString());
        }

        template <typename T>
        T valueOrThrow(ts::Result<T> r, const std::string& what) {
            if (!r.ok()) throwStatus(r.status(), what);
            return std::move(*r);
        }

        json fileSpec(const std::string& driver, const fs::path& dir) {
            return {{"driver", driver}, {"kvstore", {{"driver", "file"}, {"path", dir.string() + "/"}}}};
        }

        ts::Context makeContext() {
            return valueOrThrow(ts::Context::FromJson({{"cache_pool", {{"total_bytes_limit", 256 * 1024 * 1024}}}}),
                                "TensorStore context");
        }

        // C-order byte strides of `shape` for elements of `elemBytes`.
        std::vector<Index> cStrides(const std::vector<Index>& shape, std::size_t elemBytes) {
            std::vector<Index> s(shape.size());
            Index acc = static_cast<Index>(elemBytes);
            for (std::size_t i = shape.size(); i-- > 0;) {
                s[i] = acc;
                acc *= std::max<Index>(shape[i], 1);
            }
            return s;
        }

        // A TensorStore array view over caller memory laid out in C order
        // over `shape`; for N5 the store's dimensions are reversed, so the
        // view is the transposed one (reversed shape and strides).
        template <typename T>
        ts::SharedArray<T> viewOver(T* data, const std::vector<Index>& shape, bool reversed) {
            std::vector<Index> s = shape, b = cStrides(shape, sizeof(T));
            if (reversed) {
                std::reverse(s.begin(), s.end());
                std::reverse(b.begin(), b.end());
            }
            ts::StridedLayout<> layout(static_cast<ts::DimensionIndex>(s.size()), s.data(), b.data());
            std::shared_ptr<T> owner(std::shared_ptr<void>(), data);   // non-owning
            return ts::SharedArray<T>(std::move(owner), std::move(layout));
        }

        ZarrArrayInfo describe(const Resolved& r, const ts::TensorStore<>& store) {
            ZarrArrayInfo info;
            info.path = r.root.string();
            info.levelPath = r.levelPath;
            info.driver = r.rootLayout.driver.empty() ? std::string("zarr") : r.rootLayout.driver;
            info.isGroup = r.isGroup;
            const bool n5 = info.driver == "n5";
            const auto shape = store.domain().shape();
            info.shape.assign(shape.begin(), shape.end());
            const ts::ChunkLayout layout = valueOrThrow(store.chunk_layout(), "chunk layout of " + info.path);
            const auto rc = layout.read_chunk_shape();
            info.chunks.assign(rc.begin(), rc.end());
            if (n5) {
                std::reverse(info.shape.begin(), info.shape.end());
                std::reverse(info.chunks.begin(), info.chunks.end());
            }
            info.pixelType = pixelTypeFromName(std::string(store.dtype().name()));
            const ts::Spec spec = valueOrThrow(store.spec(), "spec of " + info.path);
            const json sj = valueOrThrow(spec.ToJson(), "spec json of " + info.path);
            info.codec = describeCodec(sj.value("metadata", json::object()), info.driver);
            if (r.isGroup) fillNgff(info, r.rootLayout.attributes);
            else {
                // a bare array may still carry OME-style attributes
                fillNgff(info, r.rootLayout.attributes);
            }
            if (n5 && !r.rootLayout.attributes.is_null() && !r.rootLayout.attributes.contains("multiscales") &&
                r.rootLayout.attributes.contains("axes")) {
                // some N5 writers put an "axes" list (F order) on the array
                std::vector<std::string> ax = r.rootLayout.attributes["axes"].get<std::vector<std::string>>();
                std::reverse(ax.begin(), ax.end());
                info.axes = ax;
            }
            info.bytesOnDisk = directoryBytes(r.root);
            return info;
        }

        ts::TensorStore<> openForRead(const Resolved& r, const ts::Context& ctx) {
            const std::string driver = r.rootLayout.driver;
            return valueOrThrow(ts::Open(fileSpec(driver, r.arrayDir), ctx, ts::OpenMode::open, ts::ReadWriteMode::read).result(),
                                "opening " + r.arrayDir.string());
        }

    } // namespace

    // --- ZarrArray ------------------------------------------------------------

    struct ZarrArray::Impl {
        ts::Context context;
        ts::TensorStore<> store;
        ZarrArrayInfo info;
        bool reversed = false;   // N5
    };

    ZarrArray::ZarrArray(const std::string& path, const std::string& levelPath) : impl_(std::make_unique<Impl>()) {
        const Resolved r = resolveArray(path, levelPath);
        impl_->context = makeContext();
        impl_->store = openForRead(r, impl_->context);
        impl_->info = describe(r, impl_->store);
        impl_->reversed = impl_->info.driver == "n5";
    }

    ZarrArray::~ZarrArray() = default;
    ZarrArray::ZarrArray(ZarrArray&&) noexcept = default;
    ZarrArray& ZarrArray::operator=(ZarrArray&&) noexcept = default;

    const ZarrArrayInfo& ZarrArray::info() const noexcept { return impl_->info; }

    template <typename T>
    void ZarrArray::read(const std::vector<Index>& origin, const std::vector<Index>& shapeIn, T* out) const {
        const ZarrArrayInfo& info = impl_->info;
        const std::size_t rank = info.shape.size();
        if (origin.size() != rank || shapeIn.size() != rank)
            throw std::invalid_argument("ZarrArray::read: origin/shape rank " + std::to_string(origin.size()) + "/" +
                                        std::to_string(shapeIn.size()) + " does not match the array rank " +
                                        std::to_string(rank));
        std::vector<Index> shape = shapeIn;
        for (std::size_t i = 0; i < rank; ++i) {
            if (shape[i] == 0) shape[i] = info.shape[i] - origin[i];
            if (origin[i] < 0 || shape[i] <= 0 || origin[i] + shape[i] > info.shape[i])
                throw std::out_of_range("ZarrArray::read: [" + std::to_string(origin[i]) + ", " +
                                        std::to_string(origin[i] + shape[i]) + ") exceeds axis " + std::to_string(i) +
                                        " of extent " + std::to_string(info.shape[i]));
        }
        std::vector<Index> o = origin, s = shape;
        if (impl_->reversed) {
            std::reverse(o.begin(), o.end());
            std::reverse(s.begin(), s.end());
        }
        auto sliced = impl_->store | ts::AllDims().SizedInterval(o, s) | ts::AllDims().TranslateTo(0);
        auto casted = ts::Cast(valueOrThrow(std::move(sliced), "slicing " + info.path), ts::dtype_v<T>);
        auto target = viewOver<T>(out, shape, impl_->reversed);
        const absl::Status st = ts::Read(valueOrThrow(std::move(casted), "casting " + info.path), target).result().status();
        if (!st.ok()) throwStatus(st, "reading " + info.path);
    }

    template <typename T>
    Buffer<T> ZarrArray::read(const std::vector<Index>& origin, const std::vector<Index>& shapeIn) const {
        std::vector<Index> shape = shapeIn;
        for (std::size_t i = 0; i < shape.size() && i < impl_->info.shape.size(); ++i)
            if (shape[i] == 0) shape[i] = impl_->info.shape[i] - (i < origin.size() ? origin[i] : 0);
        // Buffer shapes stop at rank 4: a higher-rank read collapses its
        // leading axes into one (planes, y, x).
        Shape bshape;
        if (shape.size() <= static_cast<std::size_t>(Shape::kMaxRank)) {
            bshape = Shape(shape.begin(), shape.end());
        } else {
            Index planes = 1;
            for (std::size_t i = 0; i + 2 < shape.size(); ++i) planes *= shape[i];
            bshape = Shape{planes, shape[shape.size() - 2], shape[shape.size() - 1]};
        }
        Buffer<T> out(bshape);
        read<T>(origin, shape, out.data());
        return out;
    }

    ZarrArrayInfo inspectZarr(const std::string& path) {
        ZarrArray a(path);
        return a.info();
    }

    template <typename T>
    Buffer<T> readZarr(const std::string& path, const std::vector<Index>& origin, const std::vector<Index>& shape,
                       const std::string& levelPath) {
        ZarrArray a(path, levelPath);
        return a.read<T>(origin, shape);
    }

    // --- writing ----------------------------------------------------------------

    namespace {

        json compressorZarr2(const std::string& codec, int level) {
            if (codec == "none") return nullptr;
            if (codec == "blosc-zstd") return {{"id", "blosc"}, {"cname", "zstd"}, {"clevel", level}, {"shuffle", 1}, {"blocksize", 0}};
            if (codec == "blosc-lz4") return {{"id", "blosc"}, {"cname", "lz4"}, {"clevel", level}, {"shuffle", 1}, {"blocksize", 0}};
            if (codec == "zstd") return {{"id", "zstd"}, {"level", level}};
            if (codec == "gzip") return {{"id", "zlib"}, {"level", std::clamp(level, 1, 9)}};
            throw std::invalid_argument("unknown zarr codec '" + codec + "'");
        }

        json codecsZarr3(const std::string& codec, int level, std::size_t typeSize) {
            json codecs = json::array({{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}});
            if (codec == "none") return codecs;
            if (codec == "blosc-zstd" || codec == "blosc-lz4")
                codecs.push_back({{"name", "blosc"},
                                  {"configuration", {{"cname", codec == "blosc-zstd" ? "zstd" : "lz4"}, {"clevel", level},
                                                     {"shuffle", "shuffle"}, {"typesize", typeSize}, {"blocksize", 0}}}});
            else if (codec == "zstd")
                codecs.push_back({{"name", "zstd"}, {"configuration", {{"level", level}, {"checksum", false}}}});
            else if (codec == "gzip")
                codecs.push_back({{"name", "gzip"}, {"configuration", {{"level", std::clamp(level, 1, 9)}}}});
            else
                throw std::invalid_argument("unknown zarr codec '" + codec + "'");
            return codecs;
        }

        json compressionN5(const std::string& codec, int level) {
            if (codec == "none") return {{"type", "raw"}};
            if (codec == "blosc-zstd") return {{"type", "blosc"}, {"cname", "zstd"}, {"clevel", level}, {"shuffle", 1}, {"blocksize", 0}};
            if (codec == "blosc-lz4") return {{"type", "blosc"}, {"cname", "lz4"}, {"clevel", level}, {"shuffle", 1}, {"blocksize", 0}};
            if (codec == "zstd") return {{"type", "zstd"}, {"level", level}};
            if (codec == "gzip") return {{"type", "gzip"}, {"level", std::clamp(level, 1, 9)}};
            throw std::invalid_argument("unknown N5 codec '" + codec + "'");
        }

        // Default chunking: full planes up to 512 x 512, 16 planes, one of
        // everything else.
        std::vector<Index> defaultChunks(const std::vector<Index>& shape) {
            const std::size_t r = shape.size();
            std::vector<Index> c(r, 1);
            if (r >= 1) c[r - 1] = std::min<Index>(shape[r - 1], 512);
            if (r >= 2) c[r - 2] = std::min<Index>(shape[r - 2], 512);
            if (r >= 3) c[r - 3] = std::min<Index>(shape[r - 3], 16);
            return c;
        }

        // Box-average `in` (C order over `shape`) by integer `factors` per axis.
        template <typename T>
        std::vector<T> downsampleBox(const T* in, const std::vector<Index>& shape, const std::vector<int>& factors,
                                     std::vector<Index>& outShape) {
            const std::size_t r = shape.size();
            outShape.resize(r);
            for (std::size_t i = 0; i < r; ++i) outShape[i] = (shape[i] + factors[i] - 1) / factors[i];
            const Index n = std::accumulate(shape.begin(), shape.end(), Index{1}, std::multiplies<>());
            const Index m = std::accumulate(outShape.begin(), outShape.end(), Index{1}, std::multiplies<>());
            std::vector<double> acc(static_cast<std::size_t>(m), 0.0);
            std::vector<std::uint32_t> cnt(static_cast<std::size_t>(m), 0);
            const std::vector<Index> outStride = cStrides(outShape, 1);
            std::vector<Index> idx(r, 0);
            for (Index i = 0; i < n; ++i) {
                Index o = 0;
                for (std::size_t k = 0; k < r; ++k) o += (idx[k] / factors[k]) * outStride[k];
                acc[static_cast<std::size_t>(o)] += static_cast<double>(in[i]);
                ++cnt[static_cast<std::size_t>(o)];
                for (std::size_t k = r; k-- > 0;) {
                    if (++idx[k] < shape[k]) break;
                    idx[k] = 0;
                }
            }
            std::vector<T> out(static_cast<std::size_t>(m));
            for (Index i = 0; i < m; ++i) {
                const double v = cnt[static_cast<std::size_t>(i)] ? acc[static_cast<std::size_t>(i)] / cnt[static_cast<std::size_t>(i)] : 0.0;
                if constexpr (std::is_integral_v<T>) out[static_cast<std::size_t>(i)] = static_cast<T>(std::llround(v));
                else out[static_cast<std::size_t>(i)] = static_cast<T>(v);
            }
            return out;
        }

        template <typename T>
        void writeArray(const std::string& driver, const fs::path& dir, const T* data, const std::vector<Index>& shape,
                        const std::vector<Index>& chunksIn, const ZarrWriteOptions& o, const ts::Context& ctx) {
            const bool n5 = driver == "n5";
            std::vector<Index> shp = shape, chunks = chunksIn;
            for (std::size_t i = 0; i < chunks.size(); ++i) chunks[i] = std::clamp<Index>(chunks[i], 1, std::max<Index>(shp[i], 1));
            if (n5) {
                std::reverse(shp.begin(), shp.end());
                std::reverse(chunks.begin(), chunks.end());
            }
            json spec = fileSpec(driver, dir);
            spec["create"] = true;
            spec["delete_existing"] = true;
            json schema = {{"domain", {{"shape", shp}}}, {"dtype", dtypeName(pixelTypeOf<T>())}};
            if (driver == "zarr3" && o.shard) {
                // a shard is a whole number of chunks; it may exceed the array
                std::vector<Index> shard = chunks;
                for (std::size_t i = 0; i < shard.size(); ++i) shard[i] = chunks[i] * std::max(o.shardFactor, 1);
                schema["chunk_layout"] = {{"read_chunk", {{"shape", chunks}}}, {"write_chunk", {{"shape", shard}}}};
                schema["codec"] = {{"driver", "zarr3"},
                                   {"codecs", json::array({{{"name", "sharding_indexed"},
                                                            {"configuration", {{"chunk_shape", chunks},
                                                                               {"codecs", codecsZarr3(o.codec, o.level, sizeof(T))}}}}})}};
            } else {
                schema["chunk_layout"] = {{"chunk", {{"shape", chunks}}}};
                if (driver == "zarr") schema["codec"] = {{"driver", "zarr"}, {"compressor", compressorZarr2(o.codec, o.level)}};
                else if (driver == "zarr3") schema["codec"] = {{"driver", "zarr3"}, {"codecs", codecsZarr3(o.codec, o.level, sizeof(T))}};
                else schema["codec"] = {{"driver", "n5"}, {"compression", compressionN5(o.codec, o.level)}};
            }
            spec["schema"] = schema;
            auto store = valueOrThrow(ts::Open(spec, ctx, ts::OpenMode::create | ts::OpenMode::delete_existing,
                                               ts::ReadWriteMode::read_write).result(),
                                      "creating " + dir.string());
            auto casted = valueOrThrow(ts::Cast(store, ts::dtype_v<T>), "casting " + dir.string());
            auto source = viewOver<const T>(data, shape, n5);
            const absl::Status st = ts::Write(source, casted).commit_future.result().status();
            if (!st.ok()) throwStatus(st, "writing " + dir.string());
        }

        void removeExistingStore(const fs::path& path) {
            std::error_code ec;
            if (!fs::exists(path, ec)) return;
            if (!fs::is_directory(path, ec))
                throw std::runtime_error("cannot write a zarr store over an existing file: " + path.string());
            const bool empty = fs::is_empty(path, ec);
            if (!empty && !isZarrStore(path.string()))
                throw std::runtime_error("refusing to delete " + path.string() + ": it exists and is not a zarr / N5 store");
            fs::remove_all(path, ec);
            if (ec) throw std::runtime_error("cannot remove " + path.string() + ": " + ec.message());
        }

    } // namespace

    template <typename T>
    void writeZarr(const std::string& pathIn, const T* data, const std::vector<Index>& shape, const ZarrWriteOptions& o,
                   const std::function<void(double)>& progress) {
        if (shape.empty() || shape.size() > 5) throw std::invalid_argument("writeZarr: rank must be 1..5");
        for (Index n : shape)
            if (n <= 0) throw std::invalid_argument("writeZarr: every extent must be >= 1");
        const std::string driver = o.zarrVersion == 0 ? "n5" : (o.zarrVersion == 2 ? "zarr" : "zarr3");
        if (o.zarrVersion != 0 && o.zarrVersion != 2 && o.zarrVersion != 3)
            throw std::invalid_argument("writeZarr: zarrVersion must be 2, 3 or 0 (N5)");
        const fs::path path(pathIn);
        const int levels = std::max(o.pyramidLevels, 1);
        const bool group = o.omeNgff || levels > 1;
        const std::size_t rank = shape.size();
        std::vector<std::string> axes = o.axes.empty() ? defaultAxes(static_cast<int>(rank)) : o.axes;
        if (axes.size() != rank) throw std::invalid_argument("writeZarr: axes must name every dimension");
        std::vector<double> scale = o.scale.empty() ? std::vector<double>(rank, 1.0) : o.scale;
        if (scale.size() != rank) throw std::invalid_argument("writeZarr: scale must have one entry per dimension");
        std::vector<Index> chunks = o.chunks.empty() ? defaultChunks(shape) : o.chunks;
        if (chunks.size() != rank) throw std::invalid_argument("writeZarr: chunks must have one entry per dimension");

        if (o.deleteExisting) removeExistingStore(path);
        std::error_code ec;
        fs::create_directories(group ? path : path.parent_path(), ec);
        const ts::Context ctx = makeContext();

        // which axes shrink per level
        std::vector<int> factors(rank, 1);
        const int f = std::max(o.downsample, 2);
        for (std::size_t i = 0; i < rank; ++i) {
            const std::string& a = axes[i];
            if (a == "y" || a == "x" || (a == "z" && o.downsampleZ)) factors[i] = f;
        }
        if (o.axes.empty() && rank >= 2) {   // unnamed: last two axes
            std::fill(factors.begin(), factors.end(), 1);
            factors[rank - 1] = factors[rank - 2] = f;
            if (o.downsampleZ && rank >= 3) factors[rank - 3] = f;
        }

        json datasets = json::array();
        std::vector<T> level, next;
        std::vector<Index> levelShape = shape;
        const T* src = data;
        std::vector<double> levelScale = scale;
        for (int k = 0; k < levels; ++k) {
            if (k > 0) {
                std::vector<Index> outShape;
                next = downsampleBox<T>(src, levelShape, factors, outShape);
                std::swap(level, next);
                src = level.data();
                levelShape = outShape;
                for (std::size_t i = 0; i < rank; ++i) levelScale[i] *= factors[i];
            }
            const fs::path dir = group ? path / std::to_string(k) : path;
            writeArray<T>(driver, dir, src, levelShape, chunks, o, ctx);
            datasets.push_back({{"path", std::to_string(k)},
                                {"coordinateTransformations", json::array({{{"type", "scale"}, {"scale", levelScale}}})}});
            if (progress) progress(static_cast<double>(k + 1) / static_cast<double>(levels));
            if (k + 1 < levels) {
                bool shrinkable = false;
                for (std::size_t i = 0; i < rank; ++i) shrinkable = shrinkable || (factors[i] > 1 && levelShape[i] > 1);
                if (!shrinkable) break;
            }
        }

        if (group) {
            json multiscale = {{"name", path.stem().string()},
                               {"axes", ngffAxes(axes)},
                               {"datasets", datasets},
                               {"type", "mean"},
                               {"metadata", {{"method", "box mean"}, {"factor", f}}}};
            json omero;
            if (!o.channelNames.empty()) {
                json channels = json::array();
                for (std::size_t i = 0; i < o.channelNames.size(); ++i) {
                    std::string color = i < o.channelColors.size() ? o.channelColors[i] : std::string("ffffff");
                    if (!color.empty() && color[0] == '#') color.erase(0, 1);
                    channels.push_back({{"label", o.channelNames[i]}, {"color", color}, {"active", true}});
                }
                omero = {{"channels", channels}};
            }
            if (driver == "zarr3") {
                json ome = {{"version", "0.5"}, {"multiscales", json::array({multiscale})}};
                if (!omero.is_null()) ome["omero"] = omero;
                writeJsonFile(path / "zarr.json",
                              {{"zarr_format", 3}, {"node_type", "group"}, {"attributes", {{"ome", ome}}}});
            } else if (driver == "zarr") {
                multiscale["version"] = "0.4";
                json attrs = {{"multiscales", json::array({multiscale})}};
                if (!omero.is_null()) attrs["omero"] = omero;
                writeJsonFile(path / ".zgroup", {{"zarr_format", 2}});
                writeJsonFile(path / ".zattrs", attrs);
            } else {
                multiscale["version"] = "0.4";
                json attrs = {{"multiscales", json::array({multiscale})}};
                if (!omero.is_null()) attrs["omero"] = omero;
                writeJsonFile(path / "attributes.json", attrs);
            }
        }
    }

#endif // SIRIUS_HAS_TENSORSTORE

    // --- instantiations ----------------------------------------------------------

#define SIRIUS_ZARR_INSTANTIATE(T)                                                                                   \
    template void ZarrArray::read<T>(const std::vector<Index>&, const std::vector<Index>&, T*) const;                \
    template Buffer<T> ZarrArray::read<T>(const std::vector<Index>&, const std::vector<Index>&) const;               \
    template Buffer<T> readZarr<T>(const std::string&, const std::vector<Index>&, const std::vector<Index>&,          \
                                   const std::string&);                                                              \
    template void writeZarr<T>(const std::string&, const T*, const std::vector<Index>&, const ZarrWriteOptions&,     \
                               const std::function<void(double)>&);

    SIRIUS_ZARR_INSTANTIATE(std::uint8_t)
    SIRIUS_ZARR_INSTANTIATE(std::int8_t)
    SIRIUS_ZARR_INSTANTIATE(std::uint16_t)
    SIRIUS_ZARR_INSTANTIATE(std::int16_t)
    SIRIUS_ZARR_INSTANTIATE(std::uint32_t)
    SIRIUS_ZARR_INSTANTIATE(std::int32_t)
    SIRIUS_ZARR_INSTANTIATE(float)
    SIRIUS_ZARR_INSTANTIATE(double)
#undef SIRIUS_ZARR_INSTANTIATE

} // namespace sirius
