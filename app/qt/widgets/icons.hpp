#ifndef SIRIUS_APP_WIDGETS_ICONS_HPP
#define SIRIUS_APP_WIDGETS_ICONS_HPP

// The icon set of docs/design/README.md ("Icons: Lucide (thin, 1.5-2 px
// stroke) - the prototype uses glyph stand-ins"). Archivo carries none of
// those stand-in glyphs, so what the screen showed came from whatever
// fallback font the platform picked. Every icon here is instead drawn with
// QPainter from a table of paths on Lucide's 24 x 24 grid: flat strokes, no
// gradients, no radius, one colour.

#include <QPixmap>
#include <QRectF>
#include <QString>

class QPainter;
class QColor;

namespace sirius::app::widgets {

    enum class Icon {
        None,
        // viewer tools
        Navigate,
        Probe,
        Measure,
        Roi,
        Brush,
        // label cleanup tools
        Erase,
        Fill,
        Pick,
        Merge,
        Split,
        Lasso,
        // zoom / view
        Plus,
        Minus,
        ZoomIn,
        ZoomOut,
        Fit,
        Eye,
        Pin,
        // transport and dock chrome
        Play,
        Pause,
        Maximize,
        Float,
        Dock,
        ChevronUp,
        ChevronDown,
        ChevronRight,
        // panels
        Trash,
        Sparkle,
        Pencil,
        Info,
        Check,
        Close,
        Enter,
        Recompute,
        More,
        Help,
    };

    // Paints `icon` centred in `box`, scaled from the 24 x 24 design grid,
    // with a `strokePx` pen (measured at the painted size) in `colour`.
    void drawIcon(QPainter& p, const QRectF& box, Icon icon, const QColor& colour, qreal strokePx = 1.5);
    // The same drawing as a transparent pixmap of `px` x `px` logical pixels,
    // for QLabel::setPixmap and QIcon; cached per (icon, size, colour, dpr).
    // `dpr` 0 (the default) takes the application's device pixel ratio, so an
    // icon is sharp on a 1.25x / 1.5x / 2x screen without every caller
    // remembering to ask.
    QPixmap iconPixmap(Icon icon, int px, const QColor& colour, qreal dpr = 0.0);

} // namespace sirius::app::widgets

#endif // SIRIUS_APP_WIDGETS_ICONS_HPP
