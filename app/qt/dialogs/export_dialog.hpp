#ifndef SIRIUS_APP_EXPORT_DIALOG_HPP
#define SIRIUS_APP_EXPORT_DIALOG_HPP

// File ▸ Export result…: the design's 640 px dialog. Format list on the
// left (TIFF variants, zarr, N5, raw) with a note and a size estimate;
// on the right the source step, the t / z / c range, the pixel type and
// scaling rule, the container's own knobs (compression, tiles, pyramid,
// chunks, codec, sharding), the destination and the sidecar options.

#include <memory>

#include <QDialog>

#include "core/export.hpp"
#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    class ExportDialog : public QDialog {
        Q_OBJECT
    public:
        explicit ExportDialog(WorkbenchBridge& bridge, QWidget* parent = nullptr);
        ~ExportDialog() override;

        ExportOptions options() const;
        int stepIndex() const;

    private:
        void selectFormat(int index);
        void refresh();
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_EXPORT_DIALOG_HPP
