#ifndef SIRIUS_APP_THEME_HPP
#define SIRIUS_APP_THEME_HPP

// The Modernist design tokens (docs/design/README.md) as the single source
// of colours, fonts and metrics for the Qt layer. Widgets read these
// constants when they paint; everything stock-Qt is styled through the QSS
// that applyTheme() installs. Flat: 0 px radius, 2 px rules between
// regions, 1 px between rows, shadows only on floating panels.

#include <QColor>
#include <QFont>
#include <QString>

class QApplication;
class QWidget;

namespace sirius::app::theme {

    // --- colours -----------------------------------------------------------
    inline const QColor kBg{0xf3, 0xf2, 0xf2};
    inline const QColor kSurface{0xea, 0xe9, 0xe9};
    inline const QColor kText{0x20, 0x1e, 0x1d};
    inline const QColor kDivider{0xa6, 0xa5, 0xa4};          // text @ 40 % on bg
    inline const QColor kAccent{0xec, 0x30, 0x13};
    inline const QColor kAccent600{0xdd, 0x2b, 0x0f};
    inline const QColor kAccent700{0xae, 0x18, 0x00};
    inline const QColor kNeutral200{0xea, 0xe7, 0xe7};
    inline const QColor kNeutral300{0xd7, 0xd3, 0xd3};
    inline const QColor kNeutral400{0xba, 0xb6, 0xb6};
    inline const QColor kNeutral500{0x9b, 0x97, 0x97};
    inline const QColor kNeutral600{0x7d, 0x79, 0x79};
    inline const QColor kNeutral700{0x60, 0x5d, 0x5d};
    inline const QColor kNeutral800{0x44, 0x41, 0x41};
    inline const QColor kNeutral900{0x2d, 0x2b, 0x2b};
    inline const QColor kViewerGround{0x0a, 0x09, 0x09};
    inline const QColor kViewerText{0xf3, 0xf2, 0xf2};

    // --- type ----------------------------------------------------------------
    inline const QString kFontFamily = QStringLiteral("Archivo");
    constexpr int kBodyPx = 13;
    constexpr int kSmallPx = 11;
    constexpr int kCaptionPx = 10;       // uppercase, 0.1 em tracking
    constexpr int kH4Px = 20;
    constexpr int kH3Px = 24;
    constexpr int kBrandPx = 15;
    constexpr int kMonoPx = 15;

    QFont font(int px, int weight = QFont::Normal);      // Archivo with fallbacks
    QFont heading(int px);                               // weight 800
    QFont caption();                                     // 10 px uppercase tracking
    QFont mono(int px = kMonoPx);

    // --- metrics -------------------------------------------------------------
    constexpr int kTitleBarH = 38;
    constexpr int kViewerToolbarH = 40;
    constexpr int kStatusBarH = 26;
    constexpr int kOpsDockW = 290;
    constexpr int kParamsDockW = 320;
    constexpr int kAssistantW = 330;
    constexpr int kToolStripW = 36;
    constexpr int kDiagnosticsH = 250;
    constexpr int kDiagnosticsHeaderH = 34;
    constexpr int kRule = 2;
    constexpr int kHairline = 1;

    // The complete application stylesheet (QSS) generated from the tokens.
    QString styleSheet();
    // Loads the bundled Archivo faces, sets the default font and palette and
    // installs the stylesheet.
    void applyTheme(QApplication& app);

    // CSS-style hex ("#ec3013") for building rich text.
    QString hex(const QColor& c);

} // namespace sirius::app::theme

#endif // SIRIUS_APP_THEME_HPP
