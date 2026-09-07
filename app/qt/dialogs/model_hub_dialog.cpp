// placeholder until the model hub module lands
#include "qt/dialogs/model_hub_dialog.hpp"

namespace sirius::app {
    struct ModelHubDialog::Impl { QString chosen; };
    ModelHubDialog::ModelHubDialog(WorkbenchBridge&, QWidget* parent) : QDialog(parent), impl_(std::make_unique<Impl>()) {}
    ModelHubDialog::~ModelHubDialog() = default;
    QString ModelHubDialog::chosenModel() const { return impl_->chosen; }
} // namespace sirius::app
