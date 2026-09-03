#include "qt/main_window.hpp"

#include <algorithm>
#include <exception>

#include <QAction>
#include <QApplication>
#include <QComboBox>
#include <QDockWidget>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSplitter>
#include <QStatusBar>
#include <QTabWidget>
#include <QTableWidget>
#include <QTime>
#include <QVBoxLayout>

#include <sirius/tiff_io.hpp>

#include "qt/parameter_panel.hpp"
#include "qt/qt_strings.hpp"
#include "qt/recon_worker.hpp"
#include "qt/stack_view.hpp"

namespace sirius::app {

    namespace {

        constexpr const char* kTiffFilter = "TIFF images (*.tif *.tiff);;All files (*)";
        constexpr const char* kParamFilter =
            "Parameter files (*.toml *.cfg *.txt *.config);;All files (*)";

        QString fileName(const std::string& path) {
            return path.empty() ? QObject::tr("(none)") : QFileInfo(fromStd(path)).fileName();
        }

        // Packs a device into a combo box item so no lookup table is needed.
        QVariant deviceData(Device d) { return d.isCpu() ? -1 : d.index; }
        Device deviceFrom(const QVariant& v) {
            const int i = v.toInt();
            return i < 0 ? Device::cpu() : Device::cuda(i);
        }

    } // namespace

    MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
        setWindowTitle(tr("Sirius SIM reconstruction"));
        resize(1400, 900);

        buildMenus();
        buildDock();
        buildCentral();
        buildDevices();

        worker_ = new ReconWorker(session_);   // parented to nothing: moved to the thread
        worker_->moveToThread(&workerThread_);
        connect(&workerThread_, &QThread::finished, worker_, &QObject::deleteLater);
        connect(worker_, &ReconWorker::started, this, &MainWindow::onReconStarted);
        connect(worker_, &ReconWorker::finished, this, &MainWindow::onReconFinished);
        connect(worker_, &ReconWorker::failed, this, &MainWindow::onReconFailed);
        workerThread_.start();

        session_.setParameters(params_->parameters());
        refreshState();
        log(tr("Ready. CUDA %1, %2 device(s).")
                .arg(builtWithCuda() ? tr("enabled") : tr("not compiled in"))
                .arg(cudaDeviceCount()));
    }

    MainWindow::~MainWindow() {
        // A running reconstruction cannot be cancelled; wait for it so the
        // session outlives the worker that uses it.
        workerThread_.quit();
        workerThread_.wait();
    }

    // --- construction ----------------------------------------------------

    void MainWindow::buildMenus() {
        QMenu* file = menuBar()->addMenu(tr("&File"));
        file->addAction(tr("Open &raw stack..."), this, &MainWindow::chooseRaw, QKeySequence::Open);
        file->addAction(tr("Open &OTF..."), this, &MainWindow::chooseOtf);
        file->addSeparator();
        file->addAction(tr("&Load parameters..."), this, &MainWindow::chooseParameters);
        file->addAction(tr("&Save parameters..."), this, &MainWindow::saveParameters);
        file->addSeparator();
        saveResultAction_ = file->addAction(tr("Save r&esult..."), this, &MainWindow::saveResult,
                                            QKeySequence::Save);
        saveResultAction_->setEnabled(false);
        file->addSeparator();
        file->addAction(tr("&Quit"), qApp, &QApplication::closeAllWindows, QKeySequence::Quit);
    }

    void MainWindow::buildDock() {
        auto* dock = new QDockWidget(tr("Reconstruction"), this);
        dock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);

        auto* body = new QWidget(dock);
        auto* layout = new QVBoxLayout(body);

        auto* inputs = new QFormLayout;
        rawLabel_ = new QLabel(tr("(none)"), body);
        otfLabel_ = new QLabel(tr("(none)"), body);
        inputs->addRow(tr("Raw stack"), rawLabel_);
        inputs->addRow(tr("OTF"), otfLabel_);
        layout->addLayout(inputs);

        params_ = new ParameterPanel(body);
        connect(params_, &ParameterPanel::changed, this, &MainWindow::onParametersEdited);
        auto* scroll = new QScrollArea(body);
        scroll->setWidget(params_);
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        layout->addWidget(scroll, 1);

        auto* runForm = new QFormLayout;
        device_ = new QComboBox(body);
        rigor_ = new QComboBox(body);
        rigor_->addItem(tr("Estimate (fast planning)"), static_cast<int>(PlanRigor::Estimate));
        rigor_->addItem(tr("Measure"), static_cast<int>(PlanRigor::Measure));
        rigor_->addItem(tr("Patient (slow planning)"), static_cast<int>(PlanRigor::Patient));
        rigor_->setCurrentIndex(1);
        runForm->addRow(tr("Device"), device_);
        runForm->addRow(tr("FFT planning"), rigor_);
        layout->addLayout(runForm);

        validation_ = new QLabel(body);
        validation_->setWordWrap(true);
        validation_->setStyleSheet(QStringLiteral("color: #b00020;"));
        layout->addWidget(validation_);

        run_ = new QPushButton(tr("Reconstruct"), body);
        run_->setDefault(true);
        connect(run_, &QPushButton::clicked, this, &MainWindow::startReconstruction);
        layout->addWidget(run_);

        dock->setWidget(body);
        dock->setMinimumWidth(360);
        addDockWidget(Qt::LeftDockWidgetArea, dock);
    }

    void MainWindow::buildCentral() {
        views_ = new QTabWidget(this);
        rawView_ = new StackView(views_);
        resultView_ = new StackView(views_);
        views_->addTab(rawView_, tr("Raw"));
        views_->addTab(resultView_, tr("Reconstruction"));

        fitTable_ = new QTableWidget(0, 5, this);
        fitTable_->setHorizontalHeaderLabels(
            {tr("Dir"), tr("kx (1/um)"), tr("ky (1/um)"), tr("Spacing (um)"), tr("Angle (deg)")});
        fitTable_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        fitTable_->verticalHeader()->setVisible(false);
        fitTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);

        log_ = new QPlainTextEdit(this);
        log_->setReadOnly(true);
        log_->setMaximumBlockCount(2000);

        auto* bottom = new QTabWidget(this);
        bottom->addTab(log_, tr("Log"));
        bottom->addTab(fitTable_, tr("Pattern fit"));

        auto* splitter = new QSplitter(Qt::Vertical, this);
        splitter->addWidget(views_);
        splitter->addWidget(bottom);
        splitter->setStretchFactor(0, 4);
        splitter->setStretchFactor(1, 1);
        setCentralWidget(splitter);
    }

    void MainWindow::buildDevices() {
        device_->addItem(tr("CPU"), deviceData(Device::cpu()));
        const int n = cudaDeviceCount();
        for (int i = 0; i < n; ++i) {
            QString name = tr("CUDA %1").arg(i);
            try {
                name += QStringLiteral(": ") + fromStd(deviceProperties(Device::cuda(i)).name);
            } catch (const std::exception&) {
                // unnamed device is still selectable
            }
            device_->addItem(name, deviceData(Device::cuda(i)));
        }
        if (n > 0) device_->setCurrentIndex(1);   // prefer the GPU when there is one
    }

    // --- file actions ----------------------------------------------------

    void MainWindow::openRaw(const QString& path) {
        rawView_->clear();   // the view points into the session's buffer, which is about to change
        try {
            session_.loadRaw(toStd(path));
        } catch (const std::exception& e) {
            log(tr("Failed to open %1: %2").arg(path, QString::fromUtf8(e.what())));
            refreshState();
            return;
        }
        const Buffer<double>& raw = session_.raw();
        // non-owning: the session keeps the buffer alive; openRaw() clears the
        // view before replacing it
        rawView_->setVolume(std::shared_ptr<const Buffer<double>>(&raw, [](const Buffer<double>*) {}));
        views_->setCurrentWidget(rawView_);
        lastDir_ = QFileInfo(path).absolutePath();
        log(tr("Loaded %1: %2 sections of %3 x %4")
                .arg(path).arg(raw.dim(0)).arg(raw.dim(1)).arg(raw.dim(2)));
        refreshState();
    }

    void MainWindow::openOtf(const QString& path) {
        session_.setOtfPath(toStd(path));
        lastDir_ = QFileInfo(path).absolutePath();
        log(tr("OTF: %1").arg(path));
        refreshState();
    }

    void MainWindow::openParameters(const QString& path) {
        try {
            ParameterFormat format{};
            const SIMParameters p = loadParametersAuto(toStd(path), &format);
            params_->setParameters(p);
            session_.setParameters(p);
            log(tr("Loaded %1 parameters from %2")
                    .arg(format == ParameterFormat::Toml ? tr("TOML") : tr("legacy"), path));
        } catch (const std::exception& e) {
            log(tr("Failed to load parameters %1: %2").arg(path, QString::fromUtf8(e.what())));
        }
        lastDir_ = QFileInfo(path).absolutePath();
        refreshState();
    }

    void MainWindow::chooseRaw() {
        const QString path = QFileDialog::getOpenFileName(this, tr("Open raw SIM stack"), lastDir_, kTiffFilter);
        if (!path.isEmpty()) openRaw(path);
    }

    void MainWindow::chooseOtf() {
        const QString path = QFileDialog::getOpenFileName(this, tr("Open OTF"), lastDir_, kTiffFilter);
        if (!path.isEmpty()) openOtf(path);
    }

    void MainWindow::chooseParameters() {
        const QString path = QFileDialog::getOpenFileName(this, tr("Load parameters"), lastDir_, kParamFilter);
        if (!path.isEmpty()) openParameters(path);
    }

    void MainWindow::saveParameters() {
        QString path = QFileDialog::getSaveFileName(this, tr("Save parameters"), lastDir_,
                                                    tr("TOML files (*.toml)"));
        if (path.isEmpty()) return;
        if (!path.endsWith(QStringLiteral(".toml"), Qt::CaseInsensitive)) path += QStringLiteral(".toml");
        try {
            sirius::saveParameters(toStd(path), params_->parameters());
            log(tr("Saved parameters to %1").arg(path));
        } catch (const std::exception& e) {
            log(tr("Failed to save parameters: %1").arg(QString::fromUtf8(e.what())));
        }
    }

    void MainWindow::saveResult() {
        if (!resultVolume_) return;
        QString path = QFileDialog::getSaveFileName(this, tr("Save reconstruction"), lastDir_,
                                                    tr("TIFF images (*.tif *.tiff)"));
        if (path.isEmpty()) return;
        if (!path.endsWith(QStringLiteral(".tif"), Qt::CaseInsensitive) &&
            !path.endsWith(QStringLiteral(".tiff"), Qt::CaseInsensitive))
            path += QStringLiteral(".tif");
        try {
            // 32-bit float: half the size of the double result and what most
            // viewers expect for reconstructed data
            const Buffer<double>& src = *resultVolume_;
            Buffer<float> out(src.shape());
            std::transform(src.data(), src.data() + src.size(), out.data(),
                           [](double v) { return static_cast<float>(v); });
            writeTiff<float>(toStd(path), out, TiffCompression::Deflate);
            log(tr("Saved reconstruction to %1").arg(path));
        } catch (const std::exception& e) {
            log(tr("Failed to save reconstruction: %1").arg(QString::fromUtf8(e.what())));
        }
    }

    // --- reconstruction --------------------------------------------------

    Device MainWindow::selectedDevice() const { return deviceFrom(device_->currentData()); }

    PlanRigor MainWindow::selectedRigor() const {
        return static_cast<PlanRigor>(rigor_->currentData().toInt());
    }

    void MainWindow::startReconstruction() {
        if (busy_) return;
        session_.setParameters(params_->parameters());
        const std::string problem = session_.validate();
        if (!problem.empty()) {
            log(tr("Cannot reconstruct: %1").arg(fromStd(problem)));
            return;
        }
        setBusy(true);
        const Device device = selectedDevice();
        const PlanRigor rigor = selectedRigor();
        log(tr("Reconstructing on %1 ...").arg(fromStd(toString(device))));
        // queued to the worker thread; run() emits started/finished/failed
        QMetaObject::invokeMethod(worker_, [w = worker_, device, rigor] { w->run(device, rigor); },
                                  Qt::QueuedConnection);
    }

    void MainWindow::onReconStarted() { statusBar()->showMessage(tr("Reconstructing...")); }

    void MainWindow::onReconFinished() {
        result_ = worker_->takeResult();
        setBusy(false);
        if (!result_) return;

        // the viewer shares the volume; the ReconResult keeps the fit and timings
        resultVolume_ = std::make_shared<Buffer<double>>(std::move(result_->volume));
        resultView_->setVolume(resultVolume_);
        views_->setCurrentWidget(resultView_);
        showFit(result_->fit);
        saveResultAction_->setEnabled(true);

        const QString msg = tr("Done in %1 s on %2 (%3)")
                                .arg(result_->seconds, 0, 'f', 2)
                                .arg(fromStd(toString(result_->device)))
                                .arg(result_->plansReused ? tr("plans reused") : tr("plans rebuilt"));
        log(msg);
        statusBar()->showMessage(msg, 10000);
    }

    void MainWindow::onReconFailed(const QString& message) {
        setBusy(false);
        log(tr("Reconstruction failed: %1").arg(message));
        statusBar()->showMessage(tr("Reconstruction failed"), 10000);
        QMessageBox::critical(this, tr("Reconstruction failed"), message);
    }

    void MainWindow::onParametersEdited() {
        session_.setParameters(params_->parameters());
        refreshState();
    }

    // --- state -----------------------------------------------------------

    void MainWindow::setBusy(bool busy) {
        busy_ = busy;
        // nothing that touches the session may be reachable while it runs
        params_->setEnabled(!busy);
        device_->setEnabled(!busy);
        rigor_->setEnabled(!busy);
        run_->setEnabled(!busy);
        menuBar()->setEnabled(!busy);
        if (!busy) refreshState();
    }

    void MainWindow::refreshState() {
        rawLabel_->setText(fileName(session_.rawPath()));
        otfLabel_->setText(fileName(session_.otfPath()));
        const std::string problem = session_.validate();
        validation_->setText(fromStd(problem));
        run_->setEnabled(!busy_ && problem.empty());
        if (problem.empty() && session_.hasRaw())
            statusBar()->showMessage(tr("%1 z-plane(s) per direction/phase").arg(session_.inferredNz()));
    }

    void MainWindow::log(const QString& line) {
        log_->appendPlainText(QTime::currentTime().toString(QStringLiteral("HH:mm:ss  ")) + line);
    }

    void MainWindow::showFit(const SimFit& fit) {
        const std::vector<FitRow> rows = summarizeFit(fit);
        int norders = 0;
        for (const FitRow& r : rows) norders = std::max(norders, static_cast<int>(r.ampMagnitude.size()));

        fitTable_->setColumnCount(5 + norders);
        QStringList headers{tr("Dir"), tr("kx (1/um)"), tr("ky (1/um)"), tr("Spacing (um)"), tr("Angle (deg)")};
        for (int o = 0; o < norders; ++o) headers << tr("|amp| order %1").arg(o);
        fitTable_->setHorizontalHeaderLabels(headers);

        fitTable_->setRowCount(static_cast<int>(rows.size()));
        auto cell = [this](int r, int c, const QString& text) {
            auto* item = new QTableWidgetItem(text);
            item->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
            fitTable_->setItem(r, c, item);
        };
        for (int r = 0; r < static_cast<int>(rows.size()); ++r) {
            const FitRow& row = rows[static_cast<std::size_t>(r)];
            cell(r, 0, QString::number(row.direction));
            cell(r, 1, QString::number(row.kx, 'f', 4));
            cell(r, 2, QString::number(row.ky, 'f', 4));
            cell(r, 3, QString::number(row.spacingUm, 'f', 4));
            cell(r, 4, QString::number(row.angleDeg, 'f', 2));
            for (int o = 0; o < static_cast<int>(row.ampMagnitude.size()); ++o)
                cell(r, 5 + o, QString::number(row.ampMagnitude[static_cast<std::size_t>(o)], 'f', 3));
        }
    }

} // namespace sirius::app
