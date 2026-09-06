#include "core/labels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace sirius::app {

    namespace {
        constexpr float kInf = std::numeric_limits<float>::infinity();

        // The design's label palette, cycled by id.
        constexpr std::array<std::array<float, 3>, 7> kPalette{{
            {0xff / 255.f, 0xb3 / 255.f, 0x47 / 255.f},
            {0x7c / 255.f, 0x9c / 255.f, 0xff / 255.f},
            {0xe8 / 255.f, 0x71 / 255.f, 0xd9 / 255.f},
            {0x63 / 255.f, 0xe0 / 255.f, 0x8a / 255.f},
            {0xff / 255.f, 0x5c / 255.f, 0x7a / 255.f},
            {0x6e / 255.f, 0xe7 / 255.f, 0xf2 / 255.f},
            {0xf2 / 255.f, 0xe3 / 255.f, 0x5c / 255.f},
        }};

        void requireExtent(Index z, Index y, Index x, const char* what) {
            if (z < 1 || y < 1 || x < 1)
                throw std::invalid_argument(std::string(what) + ": extents must be >= 1, got " + std::to_string(z) +
                                            " x " + std::to_string(y) + " x " + std::to_string(x));
        }

        // Union-find over provisional component labels (path halving).
        struct UnionFind {
            std::vector<std::uint32_t> parent;
            std::uint32_t find(std::uint32_t a) noexcept {
                while (parent[a] != a) {
                    parent[a] = parent[parent[a]];
                    a = parent[a];
                }
                return a;
            }
            void unite(std::uint32_t a, std::uint32_t b) noexcept {
                a = find(a);
                b = find(b);
                if (a == b) return;
                if (a < b) parent[b] = a;
                else parent[a] = b;
            }
            std::uint32_t make() {
                parent.push_back(static_cast<std::uint32_t>(parent.size()));
                return static_cast<std::uint32_t>(parent.size() - 1);
            }
        };

        // 1D squared Euclidean distance transform (Felzenszwalb & Huttenlocher
        // 2012) of f (INF where no source yet) along a strided line.
        void edt1d(const float* f, Index n, Index stride, float* out, std::vector<Index>& v, std::vector<double>& z,
                   std::vector<double>& g) {
            v.resize(static_cast<std::size_t>(n));
            z.resize(static_cast<std::size_t>(n) + 1);
            g.resize(static_cast<std::size_t>(n));
            for (Index i = 0; i < n; ++i) g[static_cast<std::size_t>(i)] = f[i * stride];
            Index k = 0;
            v[0] = 0;
            z[0] = -std::numeric_limits<double>::infinity();
            z[1] = std::numeric_limits<double>::infinity();
            for (Index q = 1; q < n; ++q) {
                const double gq = g[static_cast<std::size_t>(q)];
                if (std::isinf(gq)) continue;   // no parabola for an unreachable sample
                double s;
                for (;;) {
                    const Index p = v[static_cast<std::size_t>(k)];
                    const double gp = g[static_cast<std::size_t>(p)];
                    if (std::isinf(gp)) {
                        // the running lower envelope holds only infinities: restart from q
                        s = -std::numeric_limits<double>::infinity();
                        k = -1;
                        break;
                    }
                    s = ((gq + static_cast<double>(q) * q) - (gp + static_cast<double>(p) * p)) /
                        (2.0 * static_cast<double>(q - p));
                    if (s > z[static_cast<std::size_t>(k)]) break;
                    --k;
                    if (k < 0) { s = -std::numeric_limits<double>::infinity(); break; }
                }
                ++k;
                v[static_cast<std::size_t>(k)] = q;
                z[static_cast<std::size_t>(k)] = s;
                z[static_cast<std::size_t>(k) + 1] = std::numeric_limits<double>::infinity();
            }
            k = 0;
            for (Index q = 0; q < n; ++q) {
                while (z[static_cast<std::size_t>(k) + 1] < static_cast<double>(q)) ++k;
                const Index p = v[static_cast<std::size_t>(k)];
                const double gp = g[static_cast<std::size_t>(p)];
                const double d = static_cast<double>(q - p);
                out[q * stride] = std::isinf(gp) ? kInf : static_cast<float>(d * d + gp);
            }
        }

        struct Voxel {
            Index z, y, x;
        };
    } // namespace

    // --- LabelStats ---------------------------------------------------------------

    std::string LabelStats::flagText() const { return flags.empty() ? std::string() : flags.front(); }

    // --- LabelVolume --------------------------------------------------------------

    LabelVolume::LabelVolume(Index t, Index z, Index y, Index x) : t_(t), z_(z), y_(y), x_(x) {
        if (t < 1) throw std::invalid_argument("LabelVolume: t must be >= 1");
        requireExtent(z, y, x, "LabelVolume");
        data_ = Buffer<std::uint32_t>(Shape{t, z, y, x});
        std::fill(data_.data(), data_.data() + data_.size(), 0u);
    }

    std::uint32_t* LabelVolume::volume(Index t) noexcept { return data_.data() + t * volumeSize(); }
    const std::uint32_t* LabelVolume::volume(Index t) const noexcept { return data_.data() + t * volumeSize(); }
    std::uint32_t* LabelVolume::plane(Index t, Index z) noexcept { return volume(t) + z * y_ * x_; }
    const std::uint32_t* LabelVolume::plane(Index t, Index z) const noexcept { return volume(t) + z * y_ * x_; }
    std::uint32_t LabelVolume::at(Index t, Index z, Index y, Index x) const noexcept { return plane(t, z)[y * x_ + x]; }

    std::uint32_t LabelVolume::maxLabel() const noexcept { return maxLabel_; }

    void LabelVolume::recomputeStats(Index t, const float* probabilities) {
        if (t < 0 || t >= t_) throw std::out_of_range("LabelVolume::recomputeStats: t out of range");
        // annotations survive a recompute: keep class / reviewed of known ids
        std::unordered_map<std::uint32_t, LabelStats> previous;
        for (const LabelStats& s : stats_) previous.emplace(s.id, s);

        const std::uint32_t* v = volume(t);
        const Index n = volumeSize();
        std::uint32_t maxId = 0;
        for (Index i = 0; i < n; ++i) maxId = std::max(maxId, v[i]);
        maxLabel_ = std::max(maxLabel_, maxId);

        struct Acc {
            Index voxels = 0;
            double prob = 0.0;
            Index z0 = std::numeric_limits<Index>::max(), z1 = -1, y0 = std::numeric_limits<Index>::max(), y1 = -1,
                  x0 = std::numeric_limits<Index>::max(), x1 = -1;
        };
        std::vector<Acc> acc(static_cast<std::size_t>(maxId) + 1);
        for (Index z = 0; z < z_; ++z)
            for (Index y = 0; y < y_; ++y) {
                const std::uint32_t* row = v + (z * y_ + y) * x_;
                const float* prow = probabilities ? probabilities + (z * y_ + y) * x_ : nullptr;
                for (Index x = 0; x < x_; ++x) {
                    const std::uint32_t id = row[x];
                    if (!id) continue;
                    Acc& a = acc[id];
                    ++a.voxels;
                    if (prow) a.prob += prow[x];
                    a.z0 = std::min(a.z0, z); a.z1 = std::max(a.z1, z);
                    a.y0 = std::min(a.y0, y); a.y1 = std::max(a.y1, y);
                    a.x0 = std::min(a.x0, x); a.x1 = std::max(a.x1, x);
                }
            }

        stats_.clear();
        for (std::uint32_t id = 1; id <= maxId; ++id) {
            const Acc& a = acc[id];
            if (!a.voxels) continue;
            LabelStats s;
            auto it = previous.find(id);
            if (it != previous.end()) {
                s.cls = it->second.cls;
                s.reviewed = it->second.reviewed;
            }
            s.id = id;
            s.voxels = a.voxels;
            s.confidence = probabilities ? a.prob / static_cast<double>(a.voxels) : 1.0;
            s.bbox = {a.z0, a.z1 + 1, a.y0, a.y1 + 1, a.x0, a.x1 + 1};
            // a single plane has no z border to touch
            s.touchesBorder = a.y0 == 0 || a.y1 == y_ - 1 || a.x0 == 0 || a.x1 == x_ - 1 ||
                              (z_ > 1 && (a.z0 == 0 || a.z1 == z_ - 1));
            stats_.push_back(std::move(s));
        }
    }

    const LabelStats* LabelVolume::statsOf(std::uint32_t id) const noexcept {
        for (const LabelStats& s : stats_)
            if (s.id == id) return &s;
        return nullptr;
    }

    void LabelVolume::applyFlags(const LabelFlagRules& rules) {
        if (stats_.empty()) return;
        std::vector<Index> sizes;
        sizes.reserve(stats_.size());
        for (const LabelStats& s : stats_) sizes.push_back(s.voxels);
        std::nth_element(sizes.begin(), sizes.begin() + static_cast<std::ptrdiff_t>(sizes.size() / 2), sizes.end());
        const Index median = sizes[sizes.size() / 2];
        const Index minVoxels = rules.minVoxels > 0 ? rules.minVoxels : std::max<Index>(1, median / 8);
        for (LabelStats& s : stats_) {
            s.flags.clear();
            if (s.confidence < rules.lowConfidence) s.flags.push_back("low conf");
            if (s.voxels < minVoxels) s.flags.push_back("small");
            if (rules.flagBorder && s.touchesBorder) s.flags.push_back("touching border");
            if (rules.sizeOutlierFactor > 0.0 && static_cast<double>(s.voxels) > rules.sizeOutlierFactor * static_cast<double>(median))
                s.flags.push_back("merged?");
        }
    }

    Index LabelVolume::reviewedCount() const noexcept {
        return static_cast<Index>(std::count_if(stats_.begin(), stats_.end(), [](const LabelStats& s) { return s.reviewed; }));
    }

    Index LabelVolume::flaggedCount(const std::string& flag) const noexcept {
        Index n = 0;
        for (const LabelStats& s : stats_)
            if (std::find(s.flags.begin(), s.flags.end(), flag) != s.flags.end()) ++n;
        return n;
    }

    // --- edits ----------------------------------------------------------------------

    LabelDiff LabelVolume::paint(Index t, Index cz, Index cy, Index cx, double radius, Index zRadius,
                                 std::uint32_t label, std::uint32_t onlyLabel) {
        LabelDiff diff;
        diff.t = t;
        if (t < 0 || t >= t_) throw std::out_of_range("LabelVolume::paint: t out of range");
        std::uint32_t* v = volume(t);
        const double r = std::max(radius, 0.0);
        const Index ri = static_cast<Index>(std::ceil(r));
        zRadius = std::max<Index>(zRadius, 0);
        const double rz = static_cast<double>(zRadius);
        for (Index z = std::max<Index>(cz - zRadius, 0); z <= std::min<Index>(cz + zRadius, z_ - 1); ++z) {
            const double fz = rz > 0.0 ? static_cast<double>(z - cz) / rz : 0.0;
            for (Index y = std::max<Index>(cy - ri, 0); y <= std::min<Index>(cy + ri, y_ - 1); ++y)
                for (Index x = std::max<Index>(cx - ri, 0); x <= std::min<Index>(cx + ri, x_ - 1); ++x) {
                    const double dy = static_cast<double>(y - cy), dx = static_cast<double>(x - cx);
                    // an ellipsoid: (dx² + dy²) / r² + (dz / zRadius)² <= 1, a single
                    // voxel when the radius is below half a voxel
                    const double lateral = r >= 0.5 ? (dx * dx + dy * dy) / (r * r) : (dx == 0.0 && dy == 0.0 ? 0.0 : 2.0);
                    if (lateral + fz * fz > 1.0 + 1e-9) continue;
                    const Index i = (z * y_ + y) * x_ + x;
                    const std::uint32_t cur = v[i];
                    if (cur == label) continue;
                    if (onlyLabel && cur != onlyLabel) continue;
                    diff.indices.push_back(i);
                    diff.before.push_back(cur);
                    diff.after.push_back(label);
                    v[i] = label;
                }
        }
        maxLabel_ = std::max(maxLabel_, label);
        return diff;
    }

    LabelDiff LabelVolume::fill(Index t, Index z, Index y, Index x, std::uint32_t label) {
        LabelDiff diff;
        diff.t = t;
        if (t < 0 || t >= t_) throw std::out_of_range("LabelVolume::fill: t out of range");
        if (z < 0 || z >= z_ || y < 0 || y >= y_ || x < 0 || x >= x_)
            throw std::out_of_range("LabelVolume::fill: seed outside the volume");
        std::uint32_t* v = volume(t);
        const Index seed = (z * y_ + y) * x_ + x;
        const std::uint32_t from = v[seed];
        if (from == label) return diff;
        // the changed value doubles as the visited mark
        std::vector<Index> stack{seed};
        v[seed] = label;
        diff.indices.push_back(seed);
        const Index plane = y_ * x_;
        while (!stack.empty()) {
            const Index i = stack.back();
            stack.pop_back();
            const Index iz = i / plane, iy = (i / x_) % y_, ix = i % x_;
            const Index nb[6] = {iz > 0 ? i - plane : -1, iz + 1 < z_ ? i + plane : -1,
                                 iy > 0 ? i - x_ : -1, iy + 1 < y_ ? i + x_ : -1,
                                 ix > 0 ? i - 1 : -1, ix + 1 < x_ ? i + 1 : -1};
            for (Index j : nb) {
                if (j < 0 || v[j] != from) continue;
                v[j] = label;
                diff.indices.push_back(j);
                stack.push_back(j);
            }
        }
        diff.before.assign(diff.indices.size(), from);
        diff.after.assign(diff.indices.size(), label);
        maxLabel_ = std::max(maxLabel_, label);
        return diff;
    }

    LabelDiff LabelVolume::merge(Index t, const std::vector<std::uint32_t>& ids) {
        LabelDiff diff;
        diff.t = t;
        if (t < 0 || t >= t_) throw std::out_of_range("LabelVolume::merge: t out of range");
        std::vector<std::uint32_t> sources;
        for (std::uint32_t id : ids)
            if (id) sources.push_back(id);
        if (sources.size() < 2) return diff;
        std::sort(sources.begin(), sources.end());
        sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
        const std::uint32_t target = sources.front();
        std::uint32_t* v = volume(t);
        const Index n = volumeSize();
        for (Index i = 0; i < n; ++i) {
            const std::uint32_t cur = v[i];
            if (cur == target || cur == 0) continue;
            if (!std::binary_search(sources.begin(), sources.end(), cur)) continue;
            diff.indices.push_back(i);
            diff.before.push_back(cur);
            diff.after.push_back(target);
            v[i] = target;
        }
        return diff;
    }

    LabelDiff LabelVolume::remove(Index t, std::uint32_t id) {
        LabelDiff diff;
        diff.t = t;
        if (t < 0 || t >= t_) throw std::out_of_range("LabelVolume::remove: t out of range");
        if (!id) return diff;
        std::uint32_t* v = volume(t);
        const Index n = volumeSize();
        for (Index i = 0; i < n; ++i) {
            if (v[i] != id) continue;
            diff.indices.push_back(i);
            diff.before.push_back(id);
            diff.after.push_back(0);
            v[i] = 0;
        }
        return diff;
    }

    LabelDiff LabelVolume::split(Index t, std::uint32_t id, std::array<Index, 3> seedA, std::array<Index, 3> seedB) {
        LabelDiff diff;
        diff.t = t;
        if (t < 0 || t >= t_) throw std::out_of_range("LabelVolume::split: t out of range");
        if (!id) throw std::invalid_argument("LabelVolume::split: cannot split the background");
        std::uint32_t* v = volume(t);
        auto inside = [&](const std::array<Index, 3>& s) {
            return s[0] >= 0 && s[0] < z_ && s[1] >= 0 && s[1] < y_ && s[2] >= 0 && s[2] < x_ &&
                   v[(s[0] * y_ + s[1]) * x_ + s[2]] == id;
        };
        if (!inside(seedA) || !inside(seedB))
            throw std::invalid_argument("LabelVolume::split: both seeds must lie inside label " + std::to_string(id));

        // bounding box of the label, padded by one background voxel so the
        // distance transform sees the object's boundary on every side
        Index z0 = z_, z1 = -1, y0 = y_, y1 = -1, x0 = x_, x1 = -1;
        for (Index z = 0; z < z_; ++z)
            for (Index y = 0; y < y_; ++y) {
                const std::uint32_t* row = v + (z * y_ + y) * x_;
                for (Index x = 0; x < x_; ++x)
                    if (row[x] == id) {
                        z0 = std::min(z0, z); z1 = std::max(z1, z);
                        y0 = std::min(y0, y); y1 = std::max(y1, y);
                        x0 = std::min(x0, x); x1 = std::max(x1, x);
                    }
            }
        if (z1 < 0) return diff;
        const Index bz = z1 - z0 + 3, by = y1 - y0 + 3, bx = x1 - x0 + 3;   // one voxel of padding each side
        const Index oz = z0 - 1, oy = y0 - 1, ox = x0 - 1;
        const Index bn = bz * by * bx;
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(bn), 0);
        for (Index z = z0; z <= z1; ++z)
            for (Index y = y0; y <= y1; ++y)
                for (Index x = x0; x <= x1; ++x)
                    if (v[(z * y_ + y) * x_ + x] == id)
                        mask[static_cast<std::size_t>(((z - oz) * by + (y - oy)) * bx + (x - ox))] = 1;

        std::vector<float> dist(static_cast<std::size_t>(bn));
        distanceTransform(mask.data(), bz, by, bx, dist.data());
        // ridges of the watershed are the thin necks: flood from deep inside
        for (float& d : dist) d = -d;

        const std::uint32_t newId = maxLabel_ + 1;
        std::vector<std::uint32_t> labels(static_cast<std::size_t>(bn), 0);
        labels[static_cast<std::size_t>(((seedA[0] - oz) * by + (seedA[1] - oy)) * bx + (seedA[2] - ox))] = id;
        labels[static_cast<std::size_t>(((seedB[0] - oz) * by + (seedB[1] - oy)) * bx + (seedB[2] - ox))] = newId;
        watershed(dist.data(), mask.data(), bz, by, bx, labels.data());

        for (Index z = z0; z <= z1; ++z)
            for (Index y = y0; y <= y1; ++y)
                for (Index x = x0; x <= x1; ++x) {
                    const std::uint32_t l = labels[static_cast<std::size_t>(((z - oz) * by + (y - oy)) * bx + (x - ox))];
                    if (l != newId) continue;
                    const Index i = (z * y_ + y) * x_ + x;
                    diff.indices.push_back(i);
                    diff.before.push_back(id);
                    diff.after.push_back(newId);
                    v[i] = newId;
                }
        if (!diff.empty()) maxLabel_ = newId;
        return diff;
    }

    void LabelVolume::apply(const LabelDiff& diff, bool forward) {
        if (diff.t < 0 || diff.t >= t_) throw std::out_of_range("LabelVolume::apply: t out of range");
        if (diff.before.size() != diff.indices.size() || diff.after.size() != diff.indices.size())
            throw std::invalid_argument("LabelVolume::apply: malformed diff");
        std::uint32_t* v = volume(diff.t);
        const std::vector<std::uint32_t>& values = forward ? diff.after : diff.before;
        const Index n = volumeSize();
        for (std::size_t k = 0; k < diff.indices.size(); ++k) {
            const Index i = diff.indices[k];
            if (i < 0 || i >= n) throw std::out_of_range("LabelVolume::apply: index outside the volume");
            v[i] = values[k];
            maxLabel_ = std::max(maxLabel_, values[k]);
        }
    }

    std::shared_ptr<LabelVolume> LabelVolume::clone() const {
        auto c = std::make_shared<LabelVolume>();
        c->t_ = t_;
        c->z_ = z_;
        c->y_ = y_;
        c->x_ = x_;
        if (!data_.empty()) c->data_ = data_.clone();
        c->stats_ = stats_;
        c->maxLabel_ = maxLabel_;
        return c;
    }

    // --- algorithms -----------------------------------------------------------------

    std::uint32_t connectedComponents(const std::uint8_t* mask, Index z, Index y, Index x, std::uint32_t* out) {
        requireExtent(z, y, x, "connectedComponents");
        // Two passes: a raster scan joins each foreground voxel to its already
        // visited neighbours (-z, -y, -x) through a union-find of provisional
        // labels, then the roots are renumbered densely. Sequential, but one
        // cache-friendly sweep, which is what makes 512³ affordable.
        UnionFind uf;
        uf.make();   // 0 = background
        const Index plane = y * x;
        for (Index iz = 0; iz < z; ++iz)
            for (Index iy = 0; iy < y; ++iy) {
                const Index base = (iz * y + iy) * x;
                for (Index ix = 0; ix < x; ++ix) {
                    const Index i = base + ix;
                    if (!mask[i]) {
                        out[i] = 0;
                        continue;
                    }
                    std::uint32_t label = 0;
                    const std::uint32_t nz = iz > 0 ? out[i - plane] : 0;
                    const std::uint32_t ny = iy > 0 ? out[i - x] : 0;
                    const std::uint32_t nx = ix > 0 ? out[i - 1] : 0;
                    if (nx) label = nx;
                    if (ny) { if (label) uf.unite(label, ny); else label = ny; }
                    if (nz) { if (label) uf.unite(label, nz); else label = nz; }
                    if (!label) label = uf.make();
                    out[i] = label;
                }
            }
        // dense renumbering of the roots
        std::vector<std::uint32_t> dense(uf.parent.size(), 0);
        std::uint32_t count = 0;
        for (std::uint32_t l = 1; l < uf.parent.size(); ++l) {
            const std::uint32_t r = uf.find(l);
            if (r == l) dense[l] = ++count;
        }
        const Index n = z * plane;
        for (Index i = 0; i < n; ++i)
            if (out[i]) out[i] = dense[uf.find(out[i])];
        return count;
    }

    void watershed(const float* landscape, const std::uint8_t* mask, Index z, Index y, Index x,
                   std::uint32_t* labels) {
        requireExtent(z, y, x, "watershed");
        // Meyer's flooding: seeds enter a priority queue keyed by height, the
        // lowest voxel expands into its unlabelled masked neighbours, which
        // enter the queue at their own height. Insertion order breaks ties so
        // plateaus grow evenly from every side.
        using Item = std::tuple<float, std::uint64_t, Index>;
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
        std::uint64_t seq = 0;
        const Index plane = y * x, n = z * plane;
        for (Index i = 0; i < n; ++i) {
            if (!mask[i]) {
                labels[i] = 0;
                continue;
            }
            if (labels[i]) queue.emplace(landscape[i], seq++, i);
        }
        while (!queue.empty()) {
            const Index i = std::get<2>(queue.top());
            queue.pop();
            const std::uint32_t label = labels[i];
            const Index iz = i / plane, iy = (i / x) % y, ix = i % x;
            const Index nb[6] = {iz > 0 ? i - plane : -1, iz + 1 < z ? i + plane : -1,
                                 iy > 0 ? i - x : -1, iy + 1 < y ? i + x : -1,
                                 ix > 0 ? i - 1 : -1, ix + 1 < x ? i + 1 : -1};
            for (Index j : nb) {
                if (j < 0 || !mask[j] || labels[j]) continue;
                labels[j] = label;
                queue.emplace(landscape[j], seq++, j);
            }
        }
    }

    void distanceTransform(const std::uint8_t* mask, Index z, Index y, Index x, float* out) {
        requireExtent(z, y, x, "distanceTransform");
        const Index plane = y * x, n = z * plane;
        for (Index i = 0; i < n; ++i) out[i] = mask[i] ? kInf : 0.0f;
        // separable exact squared EDT: one 1D pass per axis, each in place
        #pragma omp parallel
        {
            std::vector<Index> v;
            std::vector<double> zz, g;
            #pragma omp for collapse(2) schedule(static)
            for (Index iz = 0; iz < z; ++iz)
                for (Index iy = 0; iy < y; ++iy) {
                    float* line = out + (iz * y + iy) * x;
                    edt1d(line, x, 1, line, v, zz, g);
                }
            #pragma omp for collapse(2) schedule(static)
            for (Index iz = 0; iz < z; ++iz)
                for (Index ix = 0; ix < x; ++ix) {
                    float* line = out + iz * plane + ix;
                    edt1d(line, y, x, line, v, zz, g);
                }
            if (z > 1) {
                #pragma omp for collapse(2) schedule(static)
                for (Index iy = 0; iy < y; ++iy)
                    for (Index ix = 0; ix < x; ++ix) {
                        float* line = out + iy * x + ix;
                        edt1d(line, z, plane, line, v, zz, g);
                    }
            }
        }
        // no background anywhere: every voxel is as far as the volume is wide
        const float far = static_cast<float>(std::max({z, y, x}));
        #pragma omp parallel for schedule(static)
        for (Index i = 0; i < n; ++i) out[i] = std::isinf(out[i]) ? far : std::sqrt(out[i]);
    }

    std::uint32_t distanceSeeds(const std::uint8_t* mask, Index z, Index y, Index x, double minDistance,
                                std::uint32_t* out) {
        requireExtent(z, y, x, "distanceSeeds");
        const Index plane = y * x, n = z * plane;
        std::vector<float> dist(static_cast<std::size_t>(n));
        distanceTransform(mask, z, y, x, dist.data());
        std::fill(out, out + n, 0u);
        // candidates: masked voxels at least as far from the background as
        // every 26-neighbour
        std::vector<Index> candidates;
        for (Index iz = 0; iz < z; ++iz)
            for (Index iy = 0; iy < y; ++iy)
                for (Index ix = 0; ix < x; ++ix) {
                    const Index i = (iz * y + iy) * x + ix;
                    const float d = dist[static_cast<std::size_t>(i)];
                    if (!mask[i] || !(d > 0.0f)) continue;
                    bool maximal = true;
                    for (Index dz = -1; dz <= 1 && maximal; ++dz)
                        for (Index dy = -1; dy <= 1 && maximal; ++dy)
                            for (Index dx = -1; dx <= 1; ++dx) {
                                const Index jz = iz + dz, jy = iy + dy, jx = ix + dx;
                                if (jz < 0 || jz >= z || jy < 0 || jy >= y || jx < 0 || jx >= x) continue;
                                if (dist[static_cast<std::size_t>((jz * y + jy) * x + jx)] > d) { maximal = false; break; }
                            }
                    if (maximal) candidates.push_back(i);
                }
        // deepest first; a candidate closer than minDistance to an accepted
        // seed is the same object's plateau, not a new one
        std::stable_sort(candidates.begin(), candidates.end(),
                         [&](Index a, Index b) { return dist[static_cast<std::size_t>(a)] > dist[static_cast<std::size_t>(b)]; });
        const double minD2 = std::max(minDistance, 1.0) * std::max(minDistance, 1.0);
        std::vector<Voxel> accepted;
        for (Index i : candidates) {
            const Voxel c{i / plane, (i / x) % y, i % x};
            bool ok = true;
            for (const Voxel& a : accepted) {
                const double dz = static_cast<double>(a.z - c.z), dy = static_cast<double>(a.y - c.y), dx = static_cast<double>(a.x - c.x);
                if (dz * dz + dy * dy + dx * dx < minD2) { ok = false; break; }
            }
            if (!ok) continue;
            accepted.push_back(c);
            out[i] = static_cast<std::uint32_t>(accepted.size());
        }
        return static_cast<std::uint32_t>(accepted.size());
    }

    std::uint32_t removeSmall(std::uint32_t* labels, Index n, Index minVoxels) {
        std::uint32_t maxId = 0;
        for (Index i = 0; i < n; ++i) maxId = std::max(maxId, labels[i]);
        std::vector<Index> counts(static_cast<std::size_t>(maxId) + 1, 0);
        for (Index i = 0; i < n; ++i) ++counts[labels[i]];
        std::vector<std::uint32_t> remap(counts.size(), 0);
        std::uint32_t next = 0;
        for (std::uint32_t id = 1; id <= maxId; ++id)
            if (counts[id] > 0 && counts[id] >= minVoxels) remap[id] = ++next;
        for (Index i = 0; i < n; ++i) labels[i] = remap[labels[i]];
        return next;
    }

    std::array<float, 3> labelColor(std::uint32_t id) noexcept {
        if (!id) return {0.f, 0.f, 0.f};
        return kPalette[(id - 1) % kPalette.size()];
    }

} // namespace sirius::app
