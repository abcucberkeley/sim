#ifndef SIRIUS_APP_MODEL_HUB_DIALOG_HPP
#define SIRIUS_APP_MODEL_HUB_DIALOG_HPP

// Segmentation models: search Hugging Face, download a TorchScript / ONNX
// file into the local model cache, or pick a model family the worker's
// Python packages provide (Cellpose, micro-SAM). The chosen model spec is
// what the Segmentation step's "model" parameter accepts.

#include <QDialog>
#include <QString>

#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    class ModelHubDialog : public QDialog {
        Q_OBJECT
    public:
        explicit ModelHubDialog(WorkbenchBridge& bridge, QWidget* parent = nullptr);
        ~ModelHubDialog() override;

        // Model spec to put into a segmentation step ("/path/model.pt",
        // "hf:repo/name:file.onnx", "cellpose:cyto3", "microsam:vit_b_lm"); empty when cancelled.
        QString chosenModel() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_MODEL_HUB_DIALOG_HPP
