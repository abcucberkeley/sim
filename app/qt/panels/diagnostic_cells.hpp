#ifndef SIRIUS_APP_DIAGNOSTIC_CELLS_HPP
#define SIRIUS_APP_DIAGNOSTIC_CELLS_HPP

// The widgets the diagnostics dock is assembled from. None of them knows
// the workbench: they render a Diagnostics struct (or one of its parts), so
// the same cells serve every operation kind and can be exercised offscreen.
// Every cell is a bg-coloured box with a 10 px uppercase caption (title
// left, meta right) on a 2 px divider-coloured grid.

#include <optional>
#include <vector>

#include <QImage>
#include <QString>
#include <QTableWidget>
#include <QWidget>

#include "core/diagnostics.hpp"

class QGridLayout;
class QLabel;

namespace sirius::app {

    // Float plane -> 8-bit grayscale with a robust (percentile) window; log
    // scale honoured by the caller through DiagnosticImage::logScale.
    QImage renderDiagnosticImage(const DiagnosticImage& image);

    // Caption row + content, the cell of the grid.
    class DiagnosticCell : public QWidget {
        Q_OBJECT
    public:
        DiagnosticCell(const QString& title, const QString& meta, QWidget* content, QWidget* parent = nullptr);
        void setCaption(const QString& title, const QString& meta);
        QWidget* content() const noexcept { return content_; }

    private:
        QLabel* title_ = nullptr;
        QLabel* meta_ = nullptr;
        QWidget* content_ = nullptr;
    };

    // One image on a black ground, aspect preserved, marks in accent.
    class ImageCellView : public QWidget {
        Q_OBJECT
    public:
        explicit ImageCellView(QWidget* parent = nullptr);
        void setImage(const DiagnosticImage& image);
        void clear(const QString& placeholder = {});
        QSize sizeHint() const override { return {200, 120}; }

    protected:
        void paintEvent(QPaintEvent*) override;

    private:
        QImage image_;
        std::vector<DiagnosticMark> marks_;
        QString placeholder_;
    };

    // Polyline over a baseline, optional dashed stop line, three labels.
    class CurveView : public QWidget {
        Q_OBJECT
    public:
        explicit CurveView(QWidget* parent = nullptr);
        void setCurve(const DiagnosticCurve& curve);
        void clear();
        QSize sizeHint() const override { return {240, 120}; }

    protected:
        void paintEvent(QPaintEvent*) override;

    private:
        std::optional<DiagnosticCurve> curve_;
    };

    // Bars flush to the bottom; bins outside [lo, hi] in neutral-400.
    class HistogramView : public QWidget {
        Q_OBJECT
    public:
        explicit HistogramView(QWidget* parent = nullptr);
        void setHistogram(const DiagnosticHistogram& h);
        void clear();
        QSize sizeHint() const override { return {200, 120}; }

    protected:
        void paintEvent(QPaintEvent*) override;

    private:
        std::optional<DiagnosticHistogram> hist_;
    };

    // Key / value rows separated by hairlines, an optional lead paragraph
    // (the step summary) and a trailing muted line.
    class FactsView : public QWidget {
        Q_OBJECT
    public:
        explicit FactsView(QWidget* parent = nullptr);
        void setFacts(const std::vector<DiagnosticFact>& facts, const QString& lead = {}, const QString& trailer = {});
        QSize sizeHint() const override;

    protected:
        void paintEvent(QPaintEvent*) override;

    private:
        std::vector<DiagnosticFact> facts_;
        QString lead_, trailer_;
    };

    // Tile grid of an AlignmentInfo, highlighted tile accent-filled.
    class TileMapView : public QWidget {
        Q_OBJECT
    public:
        explicit TileMapView(QWidget* parent = nullptr);
        void setAlignment(const AlignmentInfo& info);
        QSize sizeHint() const override { return {200, 120}; }

    protected:
        void paintEvent(QPaintEvent*) override;

    private:
        AlignmentInfo info_;
    };

    // Styled QTableWidget for a DiagnosticTable (accent cells in accent 800).
    class DiagnosticTableView : public QTableWidget {
        Q_OBJECT
    public:
        explicit DiagnosticTableView(QWidget* parent = nullptr);
        void setTable(const DiagnosticTable& table);
    };

    // The per-kind grid built from a Diagnostics value: which cells, in
    // which columns, for the active tab. Placeholders describe what a run
    // would fill in when the diagnostics are empty.
    class DiagnosticsBody : public QWidget {
        Q_OBJECT
    public:
        explicit DiagnosticsBody(QWidget* parent = nullptr);

        struct Context {
            QString stepSummary;      // one line about the selected step
            QString inputShape;       // "c2 t40 z48 y2048 x2048"
            QString outputShape;
            QString estimate;         // "Est. ~9 s · 6.2 GB peak GPU"
        };

        void setDiagnostics(const Diagnostics& d, DiagnosticsKind kind, int tab, const Context& ctx);
        // Tab names for a kind (Diagnostics::tabs when present).
        static QStringList tabNames(const Diagnostics& d, DiagnosticsKind kind);

    private:
        void clearGrid();
        DiagnosticCell* addCell(const QString& title, const QString& meta, QWidget* content, int column, int stretch,
                                int fixedWidth = 0);
        QGridLayout* grid_ = nullptr;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_DIAGNOSTIC_CELLS_HPP
