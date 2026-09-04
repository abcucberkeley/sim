#ifndef SIRIUS_APP_MAIN_WINDOW_HPP
#define SIRIUS_APP_MAIN_WINDOW_HPP

// Top-level window: parameter dock on the left, tabbed viewers in the middle
// (raw stack, OTF, reconstruction, captured band spectra, crops), fit table
// and log at the bottom. Owns the ReconSession and the worker thread that
// runs reconstructions on it.
//
// Threading contract: the session is edited only from the GUI thread and only
// while no reconstruction is running (the controls are disabled meanwhile), so
// the worker never races with an edit and no locking is required.

#include <memory>

#include <QMainWindow>
#include <QRect>
#include <QString>
#include <QThread>

#include <sirius/device.hpp>

#include "core/session.hpp"

class QAction;
class QCheckBox;
class QComboBox;
class QLabel;
class QPlainTextEdit;
class QPushButton;
class QTabWidget;
class QTableWidget;
class QTextBrowser;
class QTimer;

namespace sirius::app {

    class BandGridView;
    class OtfView;
    class ParameterPanel;
    class ReconWorker;
    class StackView;
    struct SpectrumOverlay;

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
        void useIdealOtf();
        void chooseParameters();
        void saveParameters();
        void saveResult();
        void onReconStarted();
        void onReconFinished();
        void onReconFailed(const QString& message);
        void onParametersEdited();
        void onCaptureToggled(bool on);
        void onCropRequested(QRect selection);
        void onTabCloseRequested(int index);
        void openBandTab(int direction, int bandItem, int stage);
        void showHelp(const QString& anchor);

    private:
        void buildMenus();
        void buildDock();
        void buildCentral();
        void buildDevices();
        void setBusy(bool busy);
        void refreshState();     // validate() -> run button + status
        void refreshOtfView();   // recompute / reload the OTF shown in its tab
        void applyRawGeometry(); // pixel sizes + overlays of the raw viewer
        SpectrumOverlay rawOverlay() const;
        void log(const QString& line);
        void showFit(const SimFit& fit);
        Device selectedDevice() const;
        PlanRigor selectedRigor() const;

        ReconSession session_;
        std::unique_ptr<ReconResult> result_;
        std::shared_ptr<const Buffer<double>> resultVolume_;
        std::shared_ptr<const SimDiagnostics> diagnostics_;

        QThread workerThread_;
        ReconWorker* worker_ = nullptr;   // lives on workerThread_
        bool busy_ = false;

        ParameterPanel* params_ = nullptr;
        QComboBox* device_ = nullptr;
        QComboBox* rigor_ = nullptr;
        QCheckBox* capture_ = nullptr;
        QPushButton* run_ = nullptr;
        QLabel* rawLabel_ = nullptr;
        QLabel* otfLabel_ = nullptr;
        QPushButton* otfIdeal_ = nullptr;
        QLabel* validation_ = nullptr;
        QTabWidget* views_ = nullptr;
        StackView* rawView_ = nullptr;
        OtfView* otfView_ = nullptr;
        StackView* resultView_ = nullptr;
        BandGridView* bandGrid_ = nullptr;
        int fixedTabs_ = 0;               // tabs before the first crop tab
        QTimer* otfRefresh_ = nullptr;    // debounces parameter edits
        QTabWidget* bottom_ = nullptr;
        QTableWidget* fitTable_ = nullptr;
        QTextBrowser* help_ = nullptr;
        QPlainTextEdit* log_ = nullptr;
        QAction* saveResultAction_ = nullptr;
        QString lastDir_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_MAIN_WINDOW_HPP
