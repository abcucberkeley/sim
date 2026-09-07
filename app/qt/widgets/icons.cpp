#include "qt/widgets/icons.hpp"

#include <initializer_list>

#include <QColor>
#include <QHash>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QGuiApplication>
#include <QPixmapCache>
#include <QPointF>
#include <QPolygonF>

namespace sirius::app::widgets {

    namespace {

        // One icon: what is stroked and what is filled, both on the 24 x 24
        // grid. Built once per icon and cached; drawIcon only scales it.
        struct IconPaths {
            QPainterPath stroke;
            QPainterPath fill;
        };

        IconPaths buildIcon(Icon icon) {
            IconPaths out;
            auto line = [&out](qreal x1, qreal y1, qreal x2, qreal y2) {
                out.stroke.moveTo(x1, y1);
                out.stroke.lineTo(x2, y2);
            };
            auto poly = [&out](std::initializer_list<QPointF> pts, bool close = false) {
                bool first = true;
                for (const QPointF& p : pts) {
                    if (first) out.stroke.moveTo(p);
                    else out.stroke.lineTo(p);
                    first = false;
                }
                if (close) out.stroke.closeSubpath();
            };
            auto box = [&out](qreal x, qreal y, qreal w, qreal h) { out.stroke.addRect(x, y, w, h); };
            auto ring = [&out](qreal cx, qreal cy, qreal r) { out.stroke.addEllipse(QPointF(cx, cy), r, r); };
            auto dot = [&out](qreal cx, qreal cy, qreal r) { out.fill.addEllipse(QPointF(cx, cy), r, r); };
            auto slab = [&out](qreal x, qreal y, qreal w, qreal h) { out.fill.addRect(x, y, w, h); };
            auto wedge = [&out](std::initializer_list<QPointF> pts) {
                QPolygonF p;
                for (const QPointF& q : pts) p << q;
                out.fill.addPolygon(p);
                out.fill.closeSubpath();
            };

            switch (icon) {
                case Icon::None: break;
                case Icon::Navigate:   // four-way move
                    line(12, 3, 12, 21);
                    line(3, 12, 21, 12);
                    poly({{9, 6}, {12, 3}, {15, 6}});
                    poly({{9, 18}, {12, 21}, {15, 18}});
                    poly({{6, 9}, {3, 12}, {6, 15}});
                    poly({{18, 9}, {21, 12}, {18, 15}});
                    break;
                case Icon::Probe:   // crosshair
                    line(12, 2, 12, 22);
                    line(2, 12, 22, 12);
                    ring(12, 12, 4.5);
                    break;
                case Icon::Measure:   // double-headed ruler arrow
                    line(3, 12, 21, 12);
                    poly({{7, 8}, {3, 12}, {7, 16}});
                    poly({{17, 8}, {21, 12}, {17, 16}});
                    break;
                case Icon::Roi:   // selection box with corner handles
                    box(4, 4, 16, 16);
                    slab(2.6, 2.6, 2.8, 2.8);
                    slab(18.6, 2.6, 2.8, 2.8);
                    slab(2.6, 18.6, 2.8, 2.8);
                    slab(18.6, 18.6, 2.8, 2.8);
                    break;
                case Icon::Brush:
                    dot(8, 16, 3.8);
                    line(10.8, 13.2, 20, 4);
                    break;
                case Icon::Erase:
                    poly({{3, 13}, {12, 4}, {19, 11}, {10, 20}}, true);
                    line(9, 20.5, 21, 20.5);
                    break;
                case Icon::Fill:   // filled region inside a frame
                    box(3.5, 3.5, 17, 17);
                    slab(8, 8, 8, 8);
                    break;
                case Icon::Pick:   // target
                    ring(12, 12, 8);
                    dot(12, 12, 3);
                    break;
                case Icon::Merge:   // two overlapping labels
                    ring(9, 12, 6);
                    ring(15, 12, 6);
                    break;
                case Icon::Split:   // one label forking into two
                    line(12, 21, 12, 12);
                    line(12, 12, 5, 5);
                    line(12, 12, 19, 5);
                    poly({{5, 9}, {5, 5}, {9, 5}});
                    poly({{15, 5}, {19, 5}, {19, 9}});
                    break;
                case Icon::Lasso:
                    out.stroke.addEllipse(QPointF(12, 10), 8.5, 6.0);
                    line(9, 15.5, 8, 19.5);
                    dot(7.5, 20.5, 1.6);
                    break;
                case Icon::Plus:
                    line(12, 5, 12, 19);
                    line(5, 12, 19, 12);
                    break;
                case Icon::Minus: line(5, 12, 19, 12); break;
                case Icon::ZoomIn:
                    ring(10.5, 10.5, 7);
                    line(15.5, 15.5, 21, 21);
                    line(10.5, 7, 10.5, 14);
                    line(7, 10.5, 14, 10.5);
                    break;
                case Icon::ZoomOut:
                    ring(10.5, 10.5, 7);
                    line(15.5, 15.5, 21, 21);
                    line(7, 10.5, 14, 10.5);
                    break;
                case Icon::Fit:   // arrows to opposite corners
                    poly({{14, 3}, {21, 3}, {21, 10}});
                    line(21, 3, 14, 10);
                    poly({{10, 21}, {3, 21}, {3, 14}});
                    line(3, 21, 10, 14);
                    break;
                case Icon::Eye:
                    out.stroke.moveTo(2, 12);
                    out.stroke.cubicTo(5, 6.5, 8.5, 5, 12, 5);
                    out.stroke.cubicTo(15.5, 5, 19, 6.5, 22, 12);
                    out.stroke.cubicTo(19, 17.5, 15.5, 19, 12, 19);
                    out.stroke.cubicTo(8.5, 19, 5, 17.5, 2, 12);
                    dot(12, 12, 2.8);
                    break;
                case Icon::Pin:   // the pinned Load step's hexagon
                    wedge({{12, 2}, {21, 7}, {21, 17}, {12, 22}, {3, 17}, {3, 7}});
                    break;
                case Icon::Play: wedge({{7, 4}, {20, 12}, {7, 20}}); break;
                case Icon::Pause:
                    slab(7, 4, 3.6, 16);
                    slab(13.4, 4, 3.6, 16);
                    break;
                case Icon::Maximize:   // corner brackets
                    poly({{3, 9}, {3, 3}, {9, 3}});
                    poly({{15, 3}, {21, 3}, {21, 9}});
                    poly({{21, 15}, {21, 21}, {15, 21}});
                    poly({{9, 21}, {3, 21}, {3, 15}});
                    break;
                case Icon::Float:   // a window lifted off another
                    box(2.5, 2.5, 13, 13);
                    box(8.5, 8.5, 13, 13);
                    break;
                case Icon::Dock:   // panel pinned to the bottom edge
                    box(3, 4, 18, 16);
                    slab(4.5, 14.5, 15, 4);
                    break;
                case Icon::ChevronUp: poly({{6, 15}, {12, 9}, {18, 15}}); break;
                case Icon::ChevronDown: poly({{6, 9}, {12, 15}, {18, 9}}); break;
                case Icon::ChevronRight: poly({{9, 6}, {15, 12}, {9, 18}}); break;
                case Icon::Trash:
                    line(4, 7.5, 20, 7.5);
                    line(9.5, 4.5, 14.5, 4.5);
                    box(6.5, 7.5, 11, 12);
                    line(10, 11, 10, 16);
                    line(14, 11, 14, 16);
                    break;
                case Icon::Sparkle:   // four-point star with concave arms
                    out.fill.moveTo(12, 2);
                    out.fill.quadTo(12, 12, 22, 12);
                    out.fill.quadTo(12, 12, 12, 22);
                    out.fill.quadTo(12, 12, 2, 12);
                    out.fill.quadTo(12, 12, 12, 2);
                    break;
                case Icon::Pencil:
                    poly({{3, 21}, {4.5, 15.8}, {16, 4.3}, {19.7, 8}, {8.2, 19.5}}, true);
                    line(14.4, 5.9, 18.1, 9.6);
                    break;
                case Icon::Info:
                    ring(12, 12, 9);
                    line(12, 11, 12, 17);
                    dot(12, 7.4, 1.2);
                    break;
                case Icon::Check: poly({{4, 12.5}, {9.5, 18}, {20, 6}}); break;
                case Icon::Close:
                    line(5, 5, 19, 19);
                    line(19, 5, 5, 19);
                    break;
                case Icon::Enter:   // the return key of the assistant's input
                    poly({{20, 5}, {20, 15}, {5, 15}});
                    poly({{10, 10}, {5, 15}, {10, 20}});
                    break;
                case Icon::Recompute:   // the Recompute cache policy
                    out.stroke.arcMoveTo(QRectF(4, 4, 16, 16), 65);
                    out.stroke.arcTo(QRectF(4, 4, 16, 16), 65, -295);
                    poly({{14.5, 3.5}, {19.5, 6.4}, {14.5, 9.3}});
                    break;
                case Icon::More:   // the overflow "…"
                    dot(5.5, 12, 1.6);
                    dot(12, 12, 1.6);
                    dot(18.5, 12, 1.6);
                    break;
                case Icon::Help:
                    ring(12, 12, 9);
                    out.stroke.moveTo(8.6, 9.4);
                    out.stroke.cubicTo(9.2, 6.4, 14.8, 6.2, 15.2, 9.4);
                    out.stroke.cubicTo(15.5, 11.8, 12, 12.2, 12, 15);
                    dot(12, 18, 1.2);
                    break;
            }
            return out;
        }

        const IconPaths& iconPaths(Icon icon) {
            static QHash<int, IconPaths> cache;
            const int key = static_cast<int>(icon);
            auto it = cache.find(key);
            if (it == cache.end()) it = cache.insert(key, buildIcon(icon));
            return it.value();
        }

    } // namespace

    void drawIcon(QPainter& p, const QRectF& box, Icon icon, const QColor& colour, qreal strokePx) {
        if (icon == Icon::None || box.isEmpty()) return;
        const IconPaths& paths = iconPaths(icon);
        const qreal side = qMin(box.width(), box.height());
        const qreal scale = side / 24.0;
        p.save();
        p.setRenderHint(QPainter::Antialiasing, true);
        p.translate(box.center().x() - side / 2.0, box.center().y() - side / 2.0);
        p.scale(scale, scale);
        if (!paths.fill.isEmpty()) {
            p.setPen(Qt::NoPen);
            p.setBrush(colour);
            p.drawPath(paths.fill);
        }
        if (!paths.stroke.isEmpty()) {
            QPen pen(colour, strokePx / scale);
            pen.setCapStyle(Qt::RoundCap);
            pen.setJoinStyle(Qt::RoundJoin);
            p.setPen(pen);
            p.setBrush(Qt::NoBrush);
            p.drawPath(paths.stroke);
        }
        p.restore();
    }

    QPixmap iconPixmap(Icon icon, int px, const QColor& colour, qreal dpr) {
        if (dpr <= 0.0) dpr = qApp ? qApp->devicePixelRatio() : 1.0;
        const QString key = QStringLiteral("sirius_icon_%1_%2_%3_%4")
                                .arg(static_cast<int>(icon))
                                .arg(px)
                                .arg(colour.rgba())
                                .arg(qRound(dpr * 100.0));
        QPixmap pm;
        if (QPixmapCache::find(key, &pm)) return pm;
        pm = QPixmap(qRound(px * dpr), qRound(px * dpr));
        pm.setDevicePixelRatio(dpr);
        pm.fill(Qt::transparent);
        QPainter p(&pm);
        drawIcon(p, QRectF(0, 0, px, px), icon, colour);
        p.end();
        QPixmapCache::insert(key, pm);
        return pm;
    }

} // namespace sirius::app::widgets
