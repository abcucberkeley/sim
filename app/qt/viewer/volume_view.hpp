#ifndef SIRIUS_APP_VIEWER_VOLUME_VIEW_HPP
#define SIRIUS_APP_VIEWER_VOLUME_VIEW_HPP

// The 3D layout: a ray-cast rendering of the current time point in a
// QOpenGLWidget (OpenGL 3.3 core or ES 3.0). Every visible channel is
// uploaded as an 8-bit 3D texture of its windowed intensities, down-sampled
// to at most 256 voxels per axis, and composited front to back through a
// linear opacity ramp (or as a maximum-intensity projection) in the
// channel's colour. The bounding box, the corner label, the view presets,
// the yaw / pitch sliders and the Z clip range are drawn or laid over the
// GL surface exactly as in the design.

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <QImage>
#include <QOpenGLExtraFunctions>
#include <QOpenGLWidget>
#include <QString>

#include <sirius/buffer.hpp>

namespace sirius::app {

    class VolumeView : public QOpenGLWidget, protected QOpenGLExtraFunctions {
        Q_OBJECT
    public:
        struct Channel {
            const float* data = nullptr;     // (z, y, x) host volume
            Index z = 0, y = 0, x = 0;
            float lo = 0.0f, hi = 1.0f;      // display window
            std::array<float, 3> color{1.f, 1.f, 1.f};
        };

        explicit VolumeView(QWidget* parent = nullptr);
        ~VolumeView() override;

        // `key` identifies the (output, t, channels, windows) combination; the
        // textures are rebuilt only when it changes.
        void setVolumes(quint64 key, const std::vector<Channel>& channels, const std::array<double, 3>& voxelUm);
        void clearVolumes();
        bool hasVolumes() const noexcept { return !channels_.empty(); }

        // Instance labels of the same (z, y, x) grid, composited over the
        // volume in their palette colours; `key` changes with every edit.
        void setLabels(quint64 key, const std::uint32_t* labels, Index z, Index y, Index x, float opacity, std::uint32_t only = 0);
        void clearLabels();
        bool hasLabels() const noexcept { return labels_ != nullptr; }

        void setOrientation(double yawDeg, double pitchDeg);
        double yaw() const noexcept { return yaw_; }
        double pitch() const noexcept { return pitch_; }
        void setClip(double lo, double hi);
        void setBoundingBox(bool on);
        void setZoom(double zoom);
        double zoom() const noexcept { return zoom_; }
        // Transfer function: opacity ramps from 0 at `lo` to `alpha` at `hi`
        // (normalized intensity), sampled every `stepVoxels`; `mip` switches
        // to a maximum projection.
        void setTransfer(float lo, float hi, float alpha, float stepVoxels, bool mip);
        void setMethodText(const QString& text);   // "Ray casting"
        QImage grabImage();

    signals:
        void orientationChanged(double yawDeg, double pitchDeg);   // from dragging / sliders / presets
        void clipChanged(double lo, double hi);
        void zoomChanged(double zoom);

    protected:
        void initializeGL() override;
        void resizeGL(int w, int h) override;
        void paintGL() override;
        void mousePressEvent(QMouseEvent*) override;
        void mouseMoveEvent(QMouseEvent*) override;
        void mouseReleaseEvent(QMouseEvent*) override;
        void wheelEvent(QWheelEvent*) override;
        void resizeEvent(QResizeEvent*) override;

    private:
        struct Gl;
        void uploadTextures();
        void uploadLabels();
        void layoutOverlays();
        void applyOrientation(double yaw, double pitch, bool emitSignal);

        std::unique_ptr<Gl> gl_;
        std::vector<Channel> channels_;
        quint64 key_ = 0, uploadedKey_ = 0;
        const std::uint32_t* labels_ = nullptr;
        Index lz_ = 0, ly_ = 0, lx_ = 0;
        quint64 labelsKey_ = 0, uploadedLabelsKey_ = 0;
        float labelOpacity_ = 0.45f;
        std::uint32_t labelOnly_ = 0;   // non-zero: that label alone
        std::array<double, 3> voxelUm_{0.1, 0.1, 0.2};
        double yaw_ = 35.0, pitch_ = 22.0, zoom_ = 1.0;
        double clipLo_ = 0.0, clipHi_ = 1.0;
        bool box_ = true;
        float tfLo_ = 0.05f, tfHi_ = 0.6f, tfAlpha_ = 0.9f, stepVoxels_ = 0.5f;
        bool mip_ = false;
        QString method_ = QStringLiteral("Ray casting");
        bool glOk_ = false;
        QString glError_;
        QPointF dragLast_;
        bool dragging_ = false;
        class Overlays;
        Overlays* overlays_ = nullptr;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_VIEWER_VOLUME_VIEW_HPP
