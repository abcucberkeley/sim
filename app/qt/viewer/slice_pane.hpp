#ifndef SIRIUS_APP_VIEWER_SLICE_PANE_HPP
#define SIRIUS_APP_VIEWER_SLICE_PANE_HPP

// One pane of the ortho / compare layouts: draws a rendered slice image on
// the viewer ground with a per-axis view transform (screen pixels per voxel
// and the screen position of voxel (0, 0)), and the overlays of the design
// -- corner label, scale bar, tool hint, crosshair, brush outline, measure
// and ROI marks. The pane knows nothing about tools: it reports mouse
// events in voxel coordinates and the ViewerWidget decides what they mean.

#include <QImage>
#include <QPoint>
#include <QPointF>
#include <QRectF>
#include <QString>
#include <QVector>
#include <QWidget>

#include <sirius/buffer.hpp>

namespace sirius::app {

    class SlicePane : public QWidget {
        Q_OBJECT
    public:
        enum class Kind { XY, YZ, XZ, MIP, Compare };

        // Screen pixels per voxel along the columns / rows, and where voxel
        // (0, 0) sits on screen.
        struct View {
            double zx = 1.0, zy = 1.0;
            double ox = 0.0, oy = 0.0;
        };

        explicit SlicePane(Kind kind, QWidget* parent = nullptr);
        Kind kind() const noexcept { return kind_; }

        // The rendered image covers part of a (cols, rows) voxel grid with
        // `factor` voxels per image pixel, starting at voxel `origin` (the
        // viewer renders the visible region plus a margin, not the whole
        // plane). The image is implicitly shared (cheap).
        void setContent(const QImage& img, int factor, Index cols, Index rows, const QPoint& origin = QPoint(0, 0));
        QPoint origin() const noexcept { return origin_; }
        // The grid alone (fitView needs it before the first content arrives).
        void setGrid(Index cols, Index rows) { cols_ = cols; rows_ = rows; }
        void clearContent();
        bool hasContent() const noexcept { return !image_.isNull(); }
        Index cols() const noexcept { return cols_; }
        Index rows() const noexcept { return rows_; }

        void setView(const View& v);
        const View& view() const noexcept { return view_; }
        // View that fits a grid whose voxels are (ax, ay) units in size.
        View fitView(double ax, double ay) const;
        QPointF toVoxel(const QPointF& screen) const;
        QPointF toScreen(const QPointF& voxel) const;
        bool inside(const QPointF& voxel) const;

        // --- overlays ---------------------------------------------------------
        void setTitle(const QString& title);              // "XY  z 24 / 47 ..."
        void setHint(const QString& hint);                // tool hint, bottom-left
        void setScaleBar(double umPerVoxel);              // 0 hides it
        void setCrosshair(const QPointF& voxel, bool visible, bool locked);
        void setBrushCursor(bool on, double radiusVoxels);
        // Annotations in voxel coordinates of this pane's plane: measurements
        // (1..3 points: mark, distance, angle) and ROI boxes. Pending ones
        // (still being drawn) are painted lighter.
        struct Annotation {
            enum class Kind { Measure, Roi };
            Kind kind = Kind::Measure;
            QVector<QPointF> points;
            QRectF rect;
            QString text;
            bool pending = false;
        };
        void setAnnotations(const QVector<Annotation>& annotations);
        void setMessage(const QString& text);             // centred notice ("volume too large")
        void setSmooth(bool smooth) { smooth_ = smooth; update(); }
        QPointF lastMouse() const noexcept { return mouse_; }

    signals:
        void hovered(QPointF voxel);
        void exited();
        void pressed(QPointF voxel, Qt::MouseButton button, Qt::KeyboardModifiers mods);
        void dragged(QPointF voxel, QPointF screenDelta, Qt::MouseButton button, Qt::KeyboardModifiers mods);
        void released(QPointF voxel, Qt::MouseButton button, Qt::KeyboardModifiers mods, bool moved);
        void doubleClicked(QPointF voxel, Qt::KeyboardModifiers mods);
        void wheeled(QPointF screen, double steps);
        void resized();
        void contextMenuRequested(QPoint screen, QPointF voxel);   // right click

    protected:
        void paintEvent(QPaintEvent*) override;
        void mousePressEvent(QMouseEvent*) override;
        void mouseMoveEvent(QMouseEvent*) override;
        void mouseReleaseEvent(QMouseEvent*) override;
        void mouseDoubleClickEvent(QMouseEvent*) override;
        void wheelEvent(QWheelEvent*) override;
        void leaveEvent(QEvent*) override;
        void resizeEvent(QResizeEvent*) override;

    private:
        Kind kind_;
        QImage image_;
        int factor_ = 1;
        QPoint origin_;   // voxel of the image's top-left corner
        Index cols_ = 0, rows_ = 0;
        View view_;
        QString title_, hint_, message_;
        double umPerVoxel_ = 0.0;
        QPointF cross_;
        bool crossVisible_ = false, crossLocked_ = false;
        bool brush_ = false;
        double brushRadius_ = 0.0;
        QVector<Annotation> annotations_;
        bool smooth_ = false;
        QPointF mouse_{-1, -1};
        bool mouseIn_ = false;
        Qt::MouseButton button_ = Qt::NoButton;
        QPointF pressPos_, lastDrag_;
        bool moved_ = false;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_VIEWER_SLICE_PANE_HPP
