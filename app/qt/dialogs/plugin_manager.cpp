// placeholder until the plugin manager module lands
#include "qt/dialogs/plugin_manager.hpp"

namespace sirius::app {
    struct PluginManagerDialog::Impl {};
    PluginManagerDialog::PluginManagerDialog(WorkbenchBridge&, QWidget* parent) : QDialog(parent), impl_(std::make_unique<Impl>()) {}
    PluginManagerDialog::~PluginManagerDialog() = default;
    void PluginManagerDialog::openFile(const QString&) {}
} // namespace sirius::app
