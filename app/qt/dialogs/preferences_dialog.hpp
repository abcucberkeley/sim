#ifndef SIRIUS_APP_PREFERENCES_DIALOG_HPP
#define SIRIUS_APP_PREFERENCES_DIALOG_HPP

// File ▸ Preferences…: default backend and CUDA device, the HPC worker
// connection, the Python interpreter for the local worker, and the
// assistant provider (Ollama / OpenRouter / custom OpenAI-compatible
// endpoint). Values live in QSettings; the workbench is updated on OK.

#include <memory>

#include <QDialog>

#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    class PreferencesDialog : public QDialog {
        Q_OBJECT
    public:
        explicit PreferencesDialog(WorkbenchBridge& bridge, QWidget* parent = nullptr);
        ~PreferencesDialog() override;

        // Applies the stored preferences to a fresh workbench at start-up.
        static void applyStored(Workbench& wb);

    signals:
        void assistantSettingsChanged();

    private:
        void apply();
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_PREFERENCES_DIALOG_HPP
