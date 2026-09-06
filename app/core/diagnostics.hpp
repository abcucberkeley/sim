#ifndef SIRIUS_APP_DIAGNOSTICS_HPP
#define SIRIUS_APP_DIAGNOSTICS_HPP

// What a step reports about itself beyond its output array, shown in the
// diagnostics dock. The model is data-driven -- tabs of image panels, a
// table, curves, histograms, key/value facts -- so a new operation gets a
// diagnostics view by filling this struct, and only the segmentation
// cleanup and the volume rendering have dedicated widgets.

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <sirius/buffer.hpp>

namespace sirius::app {

    enum class DiagnosticsKind { Generic, Sim, Deconvolve, Contrast, Segment, Alignment, Volume };

    struct DiagnosticMark {
        enum class Kind { Cross, Circle, Ring };
        Kind kind = Kind::Cross;
        double x = 0.0, y = 0.0;      // image pixel coordinates
        double radius = 0.0;          // circle / ring radius (pixels)
        bool accent = true;           // accent colour, else viewer text colour
        std::string text;
    };

    // One image cell: a (rows, cols) float plane rendered on a black ground
    // with an auto (percentile) window, optionally on a log scale.
    struct DiagnosticImage {
        std::string title;            // "Raw FFT · phase 1"
        std::string meta;             // "0°"
        Index rows = 0, cols = 0;
        std::vector<float> values;
        bool logScale = false;
        std::vector<DiagnosticMark> marks;
    };

    struct DiagnosticTab {
        std::string name;             // "Raw spectrum"
        std::vector<int> images;      // indices into Diagnostics::images
    };

    struct DiagnosticTable {
        std::string caption;          // "Estimated parameters"
        std::vector<std::string> header;
        std::vector<std::vector<std::string>> rows;
        std::vector<std::pair<int, int>> accentCells;   // (row, col) drawn in accent + bold
    };

    struct DiagnosticCurve {
        std::string title;            // "Convergence · relative change per iteration"
        std::vector<double> x, y;
        std::optional<double> stopX;  // dashed vertical line
        std::string leftLabel, midLabel, rightLabel;
        bool logY = false;
    };

    struct DiagnosticHistogram {
        std::string channel;          // "α-actinin"
        std::array<float, 3> color{1.f, 1.f, 1.f};
        std::vector<double> bins;     // counts
        double binLo = 0.0, binHi = 1.0;   // value range covered by the bins
        double lo = 0.0, hi = 1.0;    // window applied (values)
        double gamma = 1.0;
    };

    struct DiagnosticFact {
        std::string key, value;
    };

    struct AlignmentInfo {
        Index gridRows = 0, gridCols = 0;
        std::vector<std::string> tileNames;   // row-major over the grid
        int highlightedTile = -1;
        std::vector<DiagnosticFact> shiftStats;   // "Mean |Δ|" "2.3 px", ...
    };

    struct Diagnostics {
        DiagnosticsKind kind = DiagnosticsKind::Generic;
        std::string summary;                  // one line ("mean over t · c2 z48 y4096 x4096")
        std::string footer;                   // "Wiener 0.001 · OTF measured · ..."
        std::vector<DiagnosticImage> images;
        std::vector<DiagnosticTab> tabs;      // empty -> a single "Preview" tab of all images
        std::optional<DiagnosticTable> table;
        std::vector<DiagnosticCurve> curves;
        std::vector<DiagnosticHistogram> histograms;
        std::vector<DiagnosticFact> facts;    // "Method / Ray casting", "Est. time / ~9 s"
        std::optional<AlignmentInfo> alignment;
        std::vector<std::string> warnings;    // advisory notes shown in the parameters dock

        bool empty() const noexcept {
            return images.empty() && !table && curves.empty() && histograms.empty() && facts.empty() &&
                   !alignment && summary.empty();
        }
        int addImage(DiagnosticImage img) {
            images.push_back(std::move(img));
            return static_cast<int>(images.size()) - 1;
        }
    };

    // Centered log-power spectrum of a (rows, cols) plane as a DiagnosticImage
    // (the shared helper every spectrum panel uses).
    DiagnosticImage spectrumImage(const float* plane, Index rows, Index cols, std::string title,
                                  std::string meta = {});

    // Down-sample a (rows, cols) plane to at most `maxSide` on its longer side
    // (box filter) so diagnostics stay cheap to keep and draw.
    DiagnosticImage thumbnail(const float* plane, Index rows, Index cols, Index maxSide, std::string title,
                              std::string meta = {});

    // 30-bin histogram of `n` values.
    DiagnosticHistogram histogramOf(const float* values, Index n, int bins = 30);

} // namespace sirius::app

#endif // SIRIUS_APP_DIAGNOSTICS_HPP
