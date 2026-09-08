#ifndef SIRIUS_APP_TRAINING_EXPORT_DIALOG_HPP
#define SIRIUS_APP_TRAINING_EXPORT_DIALOG_HPP

// File ▸ Export training data…: turns the labels of one step into a dataset
// folder -- instance masks, a semantic mask, bounding boxes, optionally one
// image and one YOLO file per plane -- and appends the sample to the index so
// many exports accumulate into one training set.

#include <memory>

#include <QDialog>

#include "core/training_export.hpp"
#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    class TrainingExportDialog : public QDialog {
        Q_OBJECT
    public:
        explicit TrainingExportDialog(WorkbenchBridge& bridge, QWidget* parent = nullptr);
        ~TrainingExportDialog() override;

        TrainingExportOptions options() const;
        int stepIndex() const;

    private:
        void refresh();
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_TRAINING_EXPORT_DIALOG_HPP
