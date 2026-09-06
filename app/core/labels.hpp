#ifndef SIRIUS_APP_LABELS_HPP
#define SIRIUS_APP_LABELS_HPP

// Instance label volumes: (t, z, y, x) uint32 with 0 = background, produced by
// the segmentation steps and edited in the viewer's paint mode. Statistics
// per label feed the review table (voxels, confidence, flags) and the review
// queue; every edit returns the voxels it changed so the history can undo it.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <sirius/buffer.hpp>

#include "core/array.hpp"

namespace sirius::app {

    struct LabelStats {
        std::uint32_t id = 0;
        std::string cls = "object";
        Index voxels = 0;
        double confidence = 1.0;                // mean foreground probability, 1 when unknown
        std::array<Index, 6> bbox{};            // z0, z1, y0, y1, x0, x1 (half open)
        bool touchesBorder = false;
        std::vector<std::string> flags;         // "low conf", "small", "touching border", "merged?"
        bool reviewed = false;

        std::string flagText() const;           // first flag or ""
    };

    // Voxel diff of one edit: linear indices into one (z, y, x) volume of
    // time point t, with the values before and after.
    struct LabelDiff {
        Index t = 0;
        std::vector<Index> indices;
        std::vector<std::uint32_t> before;
        std::vector<std::uint32_t> after;

        bool empty() const noexcept { return indices.empty(); }
    };

    struct LabelFlagRules {
        double lowConfidence = 0.6;
        Index minVoxels = 0;                    // 0 = median / 8
        bool flagBorder = true;
        double sizeOutlierFactor = 4.0;         // > factor x median volume -> "merged?"
    };

    class LabelVolume {
    public:
        LabelVolume() = default;
        LabelVolume(Index t, Index z, Index y, Index x);   // zero filled

        Index t() const noexcept { return t_; }
        Index z() const noexcept { return z_; }
        Index y() const noexcept { return y_; }
        Index x() const noexcept { return x_; }
        Index volumeSize() const noexcept { return z_ * y_ * x_; }
        bool empty() const noexcept { return data_.empty(); }

        std::uint32_t* volume(Index t) noexcept;                 // (z, y, x)
        const std::uint32_t* volume(Index t) const noexcept;
        std::uint32_t* plane(Index t, Index z) noexcept;
        const std::uint32_t* plane(Index t, Index z) const noexcept;
        std::uint32_t at(Index t, Index z, Index y, Index x) const noexcept;
        BufferView<const std::uint32_t> view() const noexcept { return data_.view(); }

        std::uint32_t maxLabel() const noexcept;

        // --- statistics ---------------------------------------------------
        // Recomputed from the voxels; `probabilities` (same (z, y, x) as one
        // volume, optional) gives per-label mean confidence.
        void recomputeStats(Index t, const float* probabilities = nullptr);
        const std::vector<LabelStats>& stats() const noexcept { return stats_; }
        std::vector<LabelStats>& stats() noexcept { return stats_; }
        const LabelStats* statsOf(std::uint32_t id) const noexcept;
        void applyFlags(const LabelFlagRules& rules);
        Index reviewedCount() const noexcept;
        Index flaggedCount(const std::string& flag) const noexcept;

        // --- edits (all return the voxels they changed) -------------------
        // Paint a ball (radius in x/y voxels, zRadius planes) of `label`;
        // `erase` paints 0. `onlyLabel` restricts the change to voxels of that label.
        LabelDiff paint(Index t, Index cz, Index cy, Index cx, double radius, Index zRadius,
                        std::uint32_t label, std::uint32_t onlyLabel = 0);
        // Fill the connected region of the label under (z, y, x) with `label` (flood fill, 6-connected).
        LabelDiff fill(Index t, Index z, Index y, Index x, std::uint32_t label);
        LabelDiff merge(Index t, const std::vector<std::uint32_t>& ids);   // all into the smallest id
        LabelDiff remove(Index t, std::uint32_t id);
        // Split `id` into two by a watershed from the two seeds (distance transform);
        // the new part gets maxLabel() + 1.
        LabelDiff split(Index t, std::uint32_t id, std::array<Index, 3> seedA, std::array<Index, 3> seedB);
        // Re-apply / revert a diff.
        void apply(const LabelDiff& diff, bool forward);

        std::shared_ptr<LabelVolume> clone() const;

    private:
        Index t_ = 0, z_ = 0, y_ = 0, x_ = 0;
        Buffer<std::uint32_t> data_;
        std::vector<LabelStats> stats_;
        std::uint32_t maxLabel_ = 0;
    };

    using LabelsPtr = std::shared_ptr<const LabelVolume>;

    // --- algorithms --------------------------------------------------------

    // 6-connected components of mask > 0 on a (z, y, x) volume; returns the
    // label count. `out` receives 1..n.
    std::uint32_t connectedComponents(const std::uint8_t* mask, Index z, Index y, Index x, std::uint32_t* out);

    // Marker-based watershed on `landscape` (higher = ridge) starting from
    // `seeds` (non-zero label voxels), restricted to mask > 0. Priority-flood.
    void watershed(const float* landscape, const std::uint8_t* mask, Index z, Index y, Index x,
                   std::uint32_t* labels /* in: seeds, out: result */);

    // Seeds for a watershed of a foreground probability map: local maxima of
    // the distance transform of fg > threshold, at least `minDistance` apart.
    std::uint32_t distanceSeeds(const std::uint8_t* mask, Index z, Index y, Index x, double minDistance,
                                std::uint32_t* out);

    // Euclidean distance transform of mask > 0 (distance to the nearest 0), 3D.
    void distanceTransform(const std::uint8_t* mask, Index z, Index y, Index x, float* out);

    // Drop components smaller than minVoxels, relabel 1..n densely.
    std::uint32_t removeSmall(std::uint32_t* labels, Index n, Index minVoxels);

    // Distinct colour for a label id (the design's 7-colour label palette, cycled).
    std::array<float, 3> labelColor(std::uint32_t id) noexcept;

} // namespace sirius::app

#endif // SIRIUS_APP_LABELS_HPP
