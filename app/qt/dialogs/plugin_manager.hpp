#ifndef SIRIUS_APP_PLUGIN_MANAGER_HPP
#define SIRIUS_APP_PLUGIN_MANAGER_HPP

// User operations: the plugin folders and files the worker knows, their
// load status, a code editor with Python highlighting for quick changes
// (Save reloads the plugins), New from template, Open folder, Delete.

#include <QDialog>
#include <QString>

#include "qt/workbench_bridge.hpp"

namespace sirius::app {

    class PluginManagerDialog : public QDialog {
        Q_OBJECT
    public:
        explicit PluginManagerDialog(WorkbenchBridge& bridge, QWidget* parent = nullptr);
        ~PluginManagerDialog() override;

        void openFile(const QString& path);   // show a plugin file in the editor

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_PLUGIN_MANAGER_HPP
