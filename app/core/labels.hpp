#ifndef SIRIUS_APP_LABELS_HPP
#define SIRIUS_APP_LABELS_HPP

// Instance label volumes: (t, z, y, x) uint32 with 0 = background, produced by
// the segmentation steps and edited in the viewer's paint mode. Statistics
// per label feed the review table (voxels, confidence, flags) and the review
// queue; every edit returns the voxels it changed so the history can undo it.
//
// Ownership rule: a LabelVolume owns its statistics and shares its voxels
// copy-on-write. share() makes a second volume over the same voxels (the
// executor carries an input's labels through a step that does not produce
// its own this way), and the first mutating call on either volume -- the
// non-const volume() / plane() accessors, the edits, apply() -- gives that
// volume a private copy while the voxels are still shared. An edit through
// one step's output therefore never reaches another step's cached output,
// and an unmodified carry-through costs nothing. Sharing is decided by the
// thread doing the mutation (shared_ptr::use_count); one volume must not be
// written from two threads at once, which the workbench guarantees by
// refusing label edits while a run is active.

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
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
        LabelVolume();
        LabelVolume(Index t, Index z, Index y, Index x);   // zero filled

        Index t() const noexcept { return t_; }
        Index z() const noexcept { return z_; }
        Index y() const noexcept { return y_; }
        Index x() const noexcept { return x_; }
        Index volumeSize() const noexcept { return z_ * y_ * x_; }
        bool empty() const noexcept { return data_->empty(); }
        // True while another volume shares these voxels (share()).
        bool sharesVoxels() const noexcept { return data_.use_count() > 1; }

        // The mutable accessors detach a shared copy first (see the header note).
        std::uint32_t* volume(Index t);                          // (z, y, x)
        const std::uint32_t* volume(Index t) const noexcept;
        std::uint32_t* plane(Index t, Index z);
        const std::uint32_t* plane(Index t, Index z) const noexcept;
        std::uint32_t at(Index t, Index z, Index y, Index x) const noexcept;
        BufferView<const std::uint32_t> view() const noexcept { return data_->view(); }

        // Highest id handed out so far: monotonic, so ids never collide with
        // labels that an undo may bring back. After a dense relabel
        // (cleanup) call resetMaxLabel() to make it the highest id present.
        std::uint32_t maxLabel() const noexcept;
        void resetMaxLabel() noexcept;

        // --- statistics ---------------------------------------------------
        // Recomputed from the voxels; `probabilities` (same (z, y, x) as one
        // volume, optional) gives per-label mean confidence. The statistics
        // describe one time point, the last one computed (statsT()).
        void recomputeStats(Index t, const float* probabilities = nullptr);
        // Brings the statistics up to date after an edit, touching only the
        // labels the diff changed (each is rescanned within its bounding
        // box): what the workbench calls at the end of a brush stroke and
        // after every other edit, undo and redo. Falls back to a full
        // recompute when the diff is for another time point than the
        // statistics, or when none were computed yet. Confidence and class
        // of a known label are kept; new labels start at 1.0 / "object".
        // The flags are refreshed with the rules of the last applyFlags().
        void updateStats(const LabelDiff& diff);
        Index statsT() const noexcept { return statsT_; }   // -1: none computed
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

        // A deep copy: its own voxels from the start.
        std::shared_ptr<LabelVolume> clone() const;
        // A volume over the same voxels (and a copy of the statistics) that
        // takes a private copy on its first write; see the header note.
        std::shared_ptr<LabelVolume> share() const;

    private:
        void detach();                       // own the voxels before writing
        LabelStats* mutableStatsOf(std::uint32_t id) noexcept;

        Index t_ = 0, z_ = 0, y_ = 0, x_ = 0;
        std::shared_ptr<Buffer<std::uint32_t>> data_;   // never null
        std::vector<LabelStats> stats_;
        Index statsT_ = -1;
        std::optional<LabelFlagRules> flagRules_;       // the last applyFlags()
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
