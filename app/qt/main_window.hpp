#ifndef SIRIUS_APP_MAIN_WINDOW_HPP
#define SIRIUS_APP_MAIN_WINDOW_HPP

// Top-level window: parameter dock on the left, raw/result stack viewers in
// the middle, fit table and log at the bottom. Owns the ReconSession and the
// worker thread that runs reconstructions on it.
//
// Threading contract: the session is edited only from the GUI thread and only
// while no reconstruction is running (the controls are disabled meanwhile), so
// the worker never races with an edit and no locking is required.

#include <memory>

#include <QMainWindow>
#include <QString>
#include <QThread>

#include <sirius/device.hpp>

#include "core/session.hpp"

class QAction;
class QComboBox;
class QLabel;
class QPlainTextEdit;
class QPushButton;
class QTabWidget;
class QTableWidget;

namespace sirius::app {

    class ParameterPanel;
    class ReconWorker;
    class StackView;

    class MainWindow : public QMainWindow {
        Q_OBJECT
    public:
        explicit MainWindow(QWidget* parent = nullptr);
        ~MainWindow() override;

        // Command-line conveniences; errors are reported in the log.
        void openRaw(const QString& path);
        void openOtf(const QString& path);
        void openParameters(const QString& path);

    public slots:
        // No-op (with a log line) while the inputs do not validate.
        void startReconstruction();

    private slots:
        void chooseRaw();
        void chooseOtf();
        void chooseParameters();
        void saveParameters();
        void saveResult();
        void onReconStarted();
        void onReconFinished();
        void onReconFailed(const QString& message);
        void onParametersEdited();

    private:
        void buildMenus();
        void buildDock();
        void buildCentral();
        void buildDevices();
        void setBusy(bool busy);
        void refreshState();     // validate() -> run button + status
        void log(const QString& line);
        void showFit(const SimFit& fit);
        Device selectedDevice() const;
        PlanRigor selectedRigor() const;

        ReconSession session_;
        std::unique_ptr<ReconResult> result_;
        std::shared_ptr<const Buffer<double>> resultVolume_;

        QThread workerThread_;
        ReconWorker* worker_ = nullptr;   // lives on workerThread_
        bool busy_ = false;

        ParameterPanel* params_ = nullptr;
        QComboBox* device_ = nullptr;
        QComboBox* rigor_ = nullptr;
        QPushButton* run_ = nullptr;
        QLabel* rawLabel_ = nullptr;
        QLabel* otfLabel_ = nullptr;
        QLabel* validation_ = nullptr;
        QTabWidget* views_ = nullptr;
        StackView* rawView_ = nullptr;
        StackView* resultView_ = nullptr;
        QTableWidget* fitTable_ = nullptr;
        QPlainTextEdit* log_ = nullptr;
        QAction* saveResultAction_ = nullptr;
        QString lastDir_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_MAIN_WINDOW_HPP
