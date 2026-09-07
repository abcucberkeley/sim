#ifndef SIRIUS_APP_VIEWER_DIMS_STRIP_HPP
#define SIRIUS_APP_VIEWER_DIMS_STRIP_HPP

// The strip below the viewer: Z with its µm readout, slider and "n / max";
// T with the 20 x 20 play / pause button, the seconds readout, slider and
// "n / max". The T row hides itself when the data has one time point.

#include <QWidget>

#include <sirius/buffer.hpp>

class QLabel;
class QSlider;

namespace sirius::app {

    namespace widgets {
        class GlyphButton;
    }

    class DimsStrip : public QWidget {
        Q_OBJECT
    public:
        explicit DimsStrip(QWidget* parent = nullptr);

        // Extents and physical scales; t <= 1 hides the T row.
        void setExtents(Index nz, Index nt, double dzUm, double frameIntervalS);
        void setPosition(Index z, Index t);
        void setPlaying(bool on);
        bool playing() const noexcept { return playing_; }

    signals:
        void zRequested(Index z);
        void tRequested(Index t);
        void playToggled(bool on);

    private:
        void refreshLabels();

        QSlider* zSlider_ = nullptr;
        QSlider* tSlider_ = nullptr;
        QLabel* zUm_ = nullptr;
        QLabel* zPos_ = nullptr;
        QLabel* tSec_ = nullptr;
        QLabel* tPos_ = nullptr;
        QWidget* tRow_[3] = {nullptr, nullptr, nullptr};
        widgets::GlyphButton* play_ = nullptr;
        Index nz_ = 1, nt_ = 1, z_ = 0, t_ = 0;
        double dz_ = 0.0, dt_ = 0.0;
        bool playing_ = false;
        bool updating_ = false;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_VIEWER_DIMS_STRIP_HPP
