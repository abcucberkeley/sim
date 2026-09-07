// placeholder until the multi-file dataset module lands
#include "qt/dialogs/folder_dataset_dialog.hpp"

namespace sirius::app {
    struct FolderDatasetDialog::Impl { QString folder; };
    FolderDatasetDialog::FolderDatasetDialog(WorkbenchBridge&, const QString& folder, QWidget* parent)
        : QDialog(parent), impl_(std::make_unique<Impl>()) { impl_->folder = folder; }
    FolderDatasetDialog::~FolderDatasetDialog() = default;
    QString FolderDatasetDialog::folder() const { return impl_->folder; }
} // namespace sirius::app
