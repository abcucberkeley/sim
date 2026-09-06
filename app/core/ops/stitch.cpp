// Stitch tiles: masked pairwise registration of every overlapping tile pair,
// a global fit of the tile origins and a blended fusion (sirius/stitching.hpp)
// over a list of tile TIFFs with nominal stage positions.
#include "core/ops/builtin.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <set>

#include <sirius/stitching.hpp>

namespace sirius::app {

    namespace {

        BlendMode blendOf(const std::string& s) {
            if (s == "Average") return BlendMode::Average;
            if (s == "Maximum") return BlendMode::Maximum;
            if (s == "Overwrite") return BlendMode::Overwrite;
            return BlendMode::Feather;
        }

        class StitchOperation final : public Operation {
        public:
            StitchOperation() {
                info_.kind = "stitch";
                info_.name = "Stitch tiles";
                info_.group = "Combine";
                info_.kindLabel = "COMBINE";
                info_.diagnostics = DiagnosticsKind::Alignment;
                info_.defaultCache = CachePolicy::Disk;
                info_.helpPage = "stitch";
                ParamSpec tiles;
                tiles.key = "tiles";
                tiles.label = "Tiles";
                tiles.type = ParamType::StringList;
                tiles.defaultValue = std::vector<std::string>{};
                tiles.help = "One multi-page TIFF per tile";
                info_.params = {
                    tiles,
                    doubleListParam("positions", "Positions", {})
                        .withUnit("voxels")
                        .withHelp("Nominal origins, z y x per tile (flattened); empty = a grid from the overlap"),
                    intParam("grid_cols", "Grid columns", 0).range(0, 1000).withHelp("0 = square grid (positions empty)"),
                    doubleParam("overlap_fraction", "Overlap", 0.10).range(0.0, 0.9, 0.01, 2)
                        .withHelp("Nominal overlap between neighbours (positions empty)"),
                    doubleListParam("search_radius", "Search radius", {4.0, 32.0, 32.0}).withUnit("voxels")
                        .withHelp("How far a tile may move from its nominal origin (z, y, x)"),
                    doubleParam("min_correlation", "Min. correlation", 0.3).range(0.0, 1.0, 0.05, 2),
                    choiceParam("blend", "Blend", {"Feather", "Average", "Maximum", "Overwrite"}, "Feather"),
                    boolParam("mask_background", "Mask background", true)
                        .withHelp("Ignore voxels at or below the background level when correlating"),
                    doubleParam("background_level", "Background level", 0.0).range(-1e9, 1e9, 1.0, 2),
                };
            }

            const OpInfo& info() const noexcept override { return info_; }

            std::string summary(const ParamSet& p, const DatasetMeta&) const override {
                const std::vector<std::string> tiles = p.getStringList("tiles");
                if (tiles.empty()) return "no tiles";
                char ov[32];
                std::snprintf(ov, sizeof ov, "overlap %.0f %%", 100.0 * p.getDouble("overlap_fraction", 0.1));
                std::string blend = p.getString("blend", "Feather");
                std::transform(blend.begin(), blend.end(), blend.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                return joinSummary({std::to_string(tiles.size()) + " tiles", ov, blend});
            }

            Validation validate(const ParamSet& p, const DatasetMeta& in) const override {
                Validation v;
                (void)in;
                const std::vector<std::string> tiles = p.getStringList("tiles");
                if (tiles.size() < 2) v.errors.push_back("Stitching needs at least two tile files.");
                for (const std::string& t : tiles)
                    if (!std::filesystem::exists(t)) v.errors.push_back("Tile not found: " + t);
                const std::vector<double> pos = p.getDoubleList("positions");
                if (!pos.empty() && pos.size() != 3 * tiles.size())
                    v.errors.push_back("Positions must hold z y x for every tile (" + std::to_string(3 * tiles.size()) +
                                       " numbers, got " + std::to_string(pos.size()) + ").");
                return v;
            }

            // Nominal positions: given, or a grid from the overlap and the first tile's size.
            std::vector<std::array<double, 3>> positionsOf(const ParamSet& p, const std::vector<std::string>& tiles,
                                                           Index& rows, Index& cols) const {
                std::vector<std::array<double, 3>> out;
                const std::vector<double> pos = p.getDoubleList("positions");
                const std::size_t n = tiles.size();
                if (pos.size() == 3 * n) {
                    for (std::size_t i = 0; i < n; ++i) out.push_back({pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]});
                    std::set<long long> ys, xs;
                    for (const auto& q : out) {
                        ys.insert(std::llround(q[1]));
                        xs.insert(std::llround(q[2]));
                    }
                    rows = static_cast<Index>(ys.size());
                    cols = static_cast<Index>(xs.size());
                    return out;
                }
                const TiffInfo info = inspectTiff(tiles.front());
                cols = p.getInt("grid_cols", 0);
                if (cols <= 0) cols = static_cast<Index>(std::ceil(std::sqrt(static_cast<double>(n))));
                rows = static_cast<Index>((n + static_cast<std::size_t>(cols) - 1) / static_cast<std::size_t>(cols));
                const double ov = p.getDouble("overlap_fraction", 0.1);
                const double sx = info.width() * (1.0 - ov), sy = info.height() * (1.0 - ov);
                for (std::size_t i = 0; i < n; ++i) {
                    const Index r = static_cast<Index>(i) / cols, c = static_cast<Index>(i) % cols;
                    out.push_back({0.0, r * sy, c * sx});
                }
                return out;
            }

            DatasetMeta outputMeta(const ParamSet& p, const DatasetMeta& in) const override {
                DatasetMeta out = in;
                out.dims = Dims5{1, 1, 1, 1, 1};
                out.channels.clear();
                out.rgb = false;
                out.sim = SimLayout{};
                out.sourceType = PixelType::Float32;
                out.acquisition = "Stitched mosaic";
                const std::vector<std::string> tiles = p.getStringList("tiles");
                try {
                    if (!tiles.empty()) {
                        Index rows = 0, cols = 0;
                        const auto pos = positionsOf(p, tiles, rows, cols);
                        const TiffInfo info = inspectTiff(tiles.front());
                        double zmax = 0, ymax = 0, xmax = 0, zmin = 1e300, ymin = 1e300, xmin = 1e300;
                        for (const auto& q : pos) {
                            zmin = std::min(zmin, q[0]); ymin = std::min(ymin, q[1]); xmin = std::min(xmin, q[2]);
                            zmax = std::max(zmax, q[0]); ymax = std::max(ymax, q[1]); xmax = std::max(xmax, q[2]);
                        }
                        out.dims.z = static_cast<Index>(std::lround(zmax - zmin)) + static_cast<Index>(info.pageCount());
                        out.dims.y = static_cast<Index>(std::lround(ymax - ymin)) + info.height();
                        out.dims.x = static_cast<Index>(std::lround(xmax - xmin)) + info.width();
                    }
                } catch (const std::exception&) {
                    // unreadable tiles: validate() reports it
                }
                out.normalizeChannels();
                return out;
            }

            StepOutput run(const StepInput& input, const ParamSet& p, const StepContext& ctx) const override {
                const Validation v = validate(p, input.meta);
                if (!v.ok()) throw std::runtime_error(v.firstError());
                const std::vector<std::string> paths = p.getStringList("tiles");
                Index rows = 0, cols = 0;
                const auto positions = positionsOf(p, paths, rows, cols);
                std::vector<StitchTile> tiles;
                for (std::size_t i = 0; i < paths.size(); ++i) tiles.push_back({paths[i], positions[i]});
                StitchOptions options;
                const std::vector<double> sr = p.getDoubleList("search_radius");
                if (sr.size() == 3)
                    options.searchRadius = {static_cast<Index>(sr[0]), static_cast<Index>(sr[1]), static_cast<Index>(sr[2])};
                options.minCorrelation = p.getDouble("min_correlation", 0.3);
                options.blend = blendOf(p.getString("blend", "Feather"));
                options.maskBackground = p.getBool("mask_background", true);
                options.backgroundLevel = p.getDouble("background_level", 0.0);
                options.skipBackground = options.maskBackground;
                options.fusionBackgroundLevel = options.backgroundLevel;
                options.minOverlapFraction = 0.02;
                ctx.report(0.05, "registering " + std::to_string(tiles.size()) + " tiles");
                StitchLayout layout;
                Buffer<float> mosaic = stitchTiffTiles<float>(tiles, options, &layout);
                ctx.throwIfCancelled();
                ctx.report(0.9, "fused");

                StepOutput out;
                out.meta = outputMeta(p, input.meta);
                const Shape s = mosaic.shape().asStack();
                out.meta.dims = Dims5{1, 1, s[0], s[1], s[2]};
                mosaic.reshape(s);
                out.array = std::make_shared<Array5>(Array5::fromBuffer(std::move(mosaic), out.meta.dims));
                out.ranOn = Backend::Cpu;

                // diagnostics
                Diagnostics d;
                d.kind = DiagnosticsKind::Alignment;
                d.summary = summary(p, input.meta);
                AlignmentInfo a;
                a.gridRows = rows;
                a.gridCols = cols;
                for (std::size_t i = 0; i < tiles.size(); ++i) a.tileNames.push_back("t" + std::to_string(i + 1));
                a.highlightedTile = 0;
                double sum = 0.0, mx = 0.0, ncc = 0.0;
                int n = 0;
                for (const TileMatch& m : layout.matches) {
                    if (!m.accepted) continue;
                    double dd = 0.0;
                    for (int k = 0; k < 3; ++k) {
                        const double e = m.displacement[static_cast<std::size_t>(k)] - m.nominalDisplacement[static_cast<std::size_t>(k)];
                        dd += e * e;
                    }
                    dd = std::sqrt(dd);
                    sum += dd;
                    mx = std::max(mx, dd);
                    ncc += m.correlation;
                    ++n;
                }
                a.shiftStats.push_back({"Pairs", std::to_string(n) + " / " + std::to_string(layout.matches.size())});
                a.shiftStats.push_back({"Mean |Δ|", n ? formatNumber(sum / n, 1) + " px" : "—"});
                a.shiftStats.push_back({"Max |Δ|", n ? formatNumber(mx, 1) + " px" : "—"});
                a.shiftStats.push_back({"NCC", n ? formatNumber(ncc / n, 2) : "—"});
                d.alignment = std::move(a);
                d.facts = d.alignment->shiftStats;
                DiagnosticTab tab{"Alignment", {}};
                const Dims5& od = out.meta.dims;
                tab.images.push_back(d.addImage(thumbnail(out.array->plane(0, 0, od.z / 2), od.y, od.x, 512, "Mosaic", od.toString())));
                d.tabs.push_back(std::move(tab));
                DiagnosticTable table;
                table.caption = "Pairwise shifts";
                table.header = {"Pair", "Δz", "Δy", "Δx", "NCC"};
                for (const TileMatch& m : layout.matches) {
                    if (!m.accepted) continue;
                    table.rows.push_back({"t" + std::to_string(m.fixed + 1) + " → t" + std::to_string(m.moving + 1),
                                          formatNumber(m.displacement[0] - m.nominalDisplacement[0], 1),
                                          formatNumber(m.displacement[1] - m.nominalDisplacement[1], 1),
                                          formatNumber(m.displacement[2] - m.nominalDisplacement[2], 1),
                                          formatNumber(m.correlation, 2)});
                }
                d.table = std::move(table);
                out.diagnostics = std::move(d);
                out.note = std::to_string(tiles.size()) + " tiles · " + std::to_string(n) + " pairs · CPU";
                ctx.report(1.0, "");
                return out;
            }

        private:
            OpInfo info_;
        };

    } // namespace

    std::unique_ptr<Operation> makeStitchOperation() { return std::make_unique<StitchOperation>(); }

} // namespace sirius::app
