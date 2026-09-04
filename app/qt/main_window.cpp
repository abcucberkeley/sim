#include "qt/main_window.hpp"

#include <algorithm>
#include <cmath>
#include <exception>

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDockWidget>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGridLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSplitter>
#include <QStatusBar>
#include <QTabBar>
#include <QTabWidget>
#include <QTableWidget>
#include <QTextBrowser>
#include <QTime>
#include <QTimer>
#include <QVBoxLayout>

#include <sirius/tiff_io.hpp>

#include "core/volume_ops.hpp"
#include "qt/band_view.hpp"
#include "qt/help_text.hpp"
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

        // Non-owning shared_ptr to a buffer the session keeps alive; the
        // views are cleared before the session replaces it.
        std::shared_ptr<const Buffer<double>> borrow(const Buffer<double>& b) {
            return std::shared_ptr<const Buffer<double>>(&b, [](const Buffer<double>*) {});
        }

    } // namespace

    MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
        setWindowTitle(tr("Sirius SIM reconstruction"));
        resize(1500, 950);

        otfRefresh_ = new QTimer(this);
        otfRefresh_->setSingleShot(true);
        otfRefresh_->setInterval(400);
        connect(otfRefresh_, &QTimer::timeout, this, &MainWindow::refreshOtfView);

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
        refreshOtfView();
        log(tr("Ready. CUDA %1, %2 device(s). Without an OTF file the theoretical OTF for the NA is used.")
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
        file->addAction(tr("Use &ideal OTF (from NA)"), this, &MainWindow::useIdealOtf);
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

        // inputs: name, open button (and the ideal-OTF button)
        auto* inputs = new QGridLayout;
        rawLabel_ = new QLabel(tr("(none)"), body);
        otfLabel_ = new QLabel(tr("(none)"), body);
        for (QLabel* l : {rawLabel_, otfLabel_}) l->setTextInteractionFlags(Qt::TextSelectableByMouse);
        auto* rawOpen = new QPushButton(tr("Open..."), body);
        rawOpen->setToolTip(tr("Open the raw SIM stack: a TIFF of directions × phases × z sections in "
                               "direction → z → phase order"));
        auto* otfOpen = new QPushButton(tr("Open..."), body);
        otfOpen->setToolTip(tr("Open a radially averaged OTF TIFF (one plane per order, real/imaginary "
                               "interleaved along the last axis, as cudasirecon's makeotf writes it)"));
        otfIdeal_ = new QPushButton(tr("Ideal"), body);
        otfIdeal_->setToolTip(tr("Drop the OTF file and use the theoretical OTF computed from NA, "
                                 "immersion index and wavelength"));
        inputs->addWidget(new QLabel(tr("Raw stack"), body), 0, 0);
        inputs->addWidget(rawLabel_, 0, 1);
        inputs->addWidget(rawOpen, 0, 2);
        inputs->addWidget(new QLabel(tr("OTF"), body), 1, 0);
        inputs->addWidget(otfLabel_, 1, 1);
        inputs->addWidget(otfOpen, 1, 2);
        inputs->addWidget(otfIdeal_, 1, 3);
        inputs->setColumnStretch(1, 1);
        layout->addLayout(inputs);
        connect(rawOpen, &QPushButton::clicked, this, &MainWindow::chooseRaw);
        connect(otfOpen, &QPushButton::clicked, this, &MainWindow::chooseOtf);
        connect(otfIdeal_, &QPushButton::clicked, this, &MainWindow::useIdealOtf);

        params_ = new ParameterPanel(body);
        connect(params_, &ParameterPanel::changed, this, &MainWindow::onParametersEdited);
        connect(params_, &ParameterPanel::helpRequested, this, &MainWindow::showHelp);
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
        device_->setToolTip(tr("Where the reconstruction runs: the CPU (FFTW, OpenMP) or a CUDA GPU (cuFFT). "
                               "The numerics are identical."));
        rigor_->setToolTip(tr("FFTW planner effort (CPU only): Estimate plans instantly but transforms slower, "
                              "Measure tries a few algorithms, Patient many. Plans are reused between runs with "
                              "the same parameters and stack size."));
        runForm->addRow(tr("Device"), device_);
        runForm->addRow(tr("FFT planning"), rigor_);
        layout->addLayout(runForm);

        capture_ = new QCheckBox(tr("Capture intermediate spectra"), body);
        capture_->setToolTip(tr("Keep the separated and Wiener-filtered band spectra of the next run for the "
                                "Bands tab. Costs memory (two copies of every band) and time."));
        connect(capture_, &QCheckBox::toggled, this, &MainWindow::onCaptureToggled);
        layout->addWidget(capture_);

        validation_ = new QLabel(body);
        validation_->setWordWrap(true);
        validation_->setStyleSheet(QStringLiteral("color: #b00020;"));
        layout->addWidget(validation_);

        run_ = new QPushButton(tr("Reconstruct"), body);
        run_->setDefault(true);
        run_->setToolTip(tr("Run the full pipeline on the selected device (see Help): preprocessing, band "
                            "separation, pattern vector and amplitude fit, Wiener filter, assembly"));
        connect(run_, &QPushButton::clicked, this, &MainWindow::startReconstruction);
        layout->addWidget(run_);

        dock->setWidget(body);
        dock->setMinimumWidth(400);
        addDockWidget(Qt::LeftDockWidgetArea, dock);
    }

    void MainWindow::buildCentral() {
        views_ = new QTabWidget(this);
        rawView_ = new StackView(views_);
        otfView_ = new OtfView(views_);
        bandGrid_ = new BandGridView(views_);
        resultView_ = new StackView(views_);
        views_->addTab(rawView_, tr("Raw"));
        views_->addTab(otfView_, tr("OTF"));
        views_->addTab(bandGrid_, tr("Bands"));
        views_->addTab(resultView_, tr("Reconstruction"));
        views_->setTabToolTip(0, tr("The loaded raw stack, one section per slice"));
        views_->setTabToolTip(1, tr("The OTF the reconstruction will use, rendered on the data grid"));
        views_->setTabToolTip(2, tr("Separated and Wiener-filtered band spectra of the last run (needs "
                                    "\"Capture intermediate spectra\")"));
        views_->setTabToolTip(3, tr("The reconstructed super-resolution volume"));
        connect(bandGrid_, &BandGridView::openRequested, this, &MainWindow::openBandTab);
        fixedTabs_ = views_->count();
        views_->setTabsClosable(true);
        for (int i = 0; i < fixedTabs_; ++i) {   // only crop tabs get a close button
            views_->tabBar()->setTabButton(i, QTabBar::RightSide, nullptr);
            views_->tabBar()->setTabButton(i, QTabBar::LeftSide, nullptr);
        }
        connect(views_, &QTabWidget::tabCloseRequested, this, &MainWindow::onTabCloseRequested);
        connect(rawView_, &StackView::cropRequested, this, &MainWindow::onCropRequested);
        connect(resultView_, &StackView::cropRequested, this, &MainWindow::onCropRequested);

        fitTable_ = new QTableWidget(0, 5, this);
        fitTable_->setHorizontalHeaderLabels(
            {tr("Dir"), tr("kx (1/um)"), tr("ky (1/um)"), tr("Spacing (um)"), tr("Angle (deg)")});
        fitTable_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        fitTable_->verticalHeader()->setVisible(false);
        fitTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);

        log_ = new QPlainTextEdit(this);
        log_->setReadOnly(true);
        log_->setMaximumBlockCount(2000);

        help_ = new QTextBrowser(this);
        help_->setOpenExternalLinks(false);
        help_->setHtml(helpHtml());

        bottom_ = new QTabWidget(this);
        bottom_->addTab(log_, tr("Log"));
        bottom_->addTab(fitTable_, tr("Pattern fit"));
        bottom_->addTab(help_, tr("Help"));
        bottom_->setTabToolTip(1, tr("Pattern vectors and modulation amplitudes fitted by the last run"));
        bottom_->setTabToolTip(2, tr("How the reconstruction works and what each parameter does"));

        auto* splitter = new QSplitter(Qt::Vertical, this);
        splitter->addWidget(views_);
        splitter->addWidget(bottom_);
        splitter->setStretchFactor(0, 5);
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
        rawView_->setVolume(borrow(raw));
        applyRawGeometry();
        views_->setCurrentWidget(rawView_);
        lastDir_ = QFileInfo(path).absolutePath();
        log(tr("Loaded %1: %2 sections of %3 x %4")
                .arg(path).arg(raw.dim(0)).arg(raw.dim(1)).arg(raw.dim(2)));
        refreshState();
        refreshOtfView();   // the OTF is rendered on the stack's grid (and 2D vs 3D may change)
    }

    void MainWindow::openOtf(const QString& path) {
        session_.setOtfPath(toStd(path));
        lastDir_ = QFileInfo(path).absolutePath();
        log(tr("OTF: %1").arg(path));
        refreshState();
        refreshOtfView();
    }

    void MainWindow::useIdealOtf() {
        if (session_.usesIdealOtf()) return;
        session_.setOtfPath({});
        log(tr("OTF: theoretical OTF from NA %1, n %2, %3 nm")
                .arg(session_.parameters().na).arg(session_.parameters().nimm)
                .arg(session_.parameters().wavelength_nm));
        refreshState();
        refreshOtfView();
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
        applyRawGeometry();
        refreshState();
        refreshOtfView();
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

    // --- viewers ---------------------------------------------------------

    SpectrumOverlay MainWindow::rawOverlay() const {
        const SIMParameters& p = session_.parameters();
        SpectrumOverlay o;
        o.supportRadius = otfSupportRadius(p);
        o.norders = p.norders > 0 ? p.norders : p.nphases / 2 + 1;
        o.predictedK0 = predictedK0(p, std::max<Index>(session_.inferredNz(), 1));
        if (result_) {
            o.fittedK0 = result_->fit.k0;
            for (const auto& amps : result_->fit.amps) {
                std::vector<double> mags;
                for (const auto& a : amps) mags.push_back(std::abs(a));
                o.ampMagnitude.push_back(std::move(mags));
            }
        }
        return o;
    }

    void MainWindow::applyRawGeometry() {
        const SIMParameters& p = session_.parameters();
        rawView_->setPixelSize(p.dx, p.dy, p.dz);
        rawView_->setOverlay(rawOverlay());
        if (result_) {
            resultView_->setPixelSize(p.dx / p.zoomfact, p.dy / p.zoomfact, p.dz / p.z_zoom);
            resultView_->setOverlay(rawOverlay());
        }
    }

    void MainWindow::refreshOtfView() {
        otfRefresh_->stop();
        std::shared_ptr<const OTFRadiallyAveraged> otf;
        try {
            otf = session_.otf();
        } catch (const std::exception& e) {
            otfView_->clear();
            statusBar()->showMessage(tr("OTF unavailable: %1").arg(QString::fromUtf8(e.what())), 10000);
            return;
        }
        Index nx = 256, ny = 256, nz = 1;
        if (session_.hasRaw()) {
            nx = session_.raw().dim(2);
            ny = session_.raw().dim(1);
            nz = std::max<Index>(session_.inferredNz(), 1);
        } else if (otf->data().dimension(2) > 1) {
            nz = otf->data().dimension(2);
        }
        const QString source = session_.usesIdealOtf()
                                   ? tr("Ideal OTF (NA %1, n %2, %3 nm)")
                                         .arg(session_.parameters().na).arg(session_.parameters().nimm)
                                         .arg(session_.parameters().wavelength_nm)
                                   : fileName(session_.otfPath());
        otfView_->setOtf(otf, session_.parameters(), nx, ny, nz, source);
    }

    void MainWindow::onCropRequested(QRect r) {
        auto* source = qobject_cast<StackView*>(sender());
        if (!source || !source->volume()) return;
        const Buffer<double>& v = *source->volume();
        try {
            auto crop = std::make_shared<Buffer<double>>(
                cropVolume(v.view(), 0, v.dim(0), r.top(), r.bottom() + 1, r.left(), r.right() + 1));
            auto* view = new StackView(views_);
            const auto px = source->pixelSize();
            view->setPixelSize(px[0], px[1], px[2]);
            view->setVolumeIsSpectrum(source->volumeIsSpectrum());
            view->setLogScale(source->logScale());
            view->setVolume(crop);
            connect(view, &StackView::cropRequested, this, &MainWindow::onCropRequested);
            const int i = views_->addTab(view, tr("%1 [%2,%3 %4x%5]")
                                                   .arg(views_->tabText(views_->indexOf(source)))
                                                   .arg(r.left()).arg(r.top()).arg(r.width()).arg(r.height()));
            views_->setCurrentIndex(i);
            log(tr("Cropped %1 x %2 x %3 at (x %4, y %5)")
                    .arg(crop->dim(0)).arg(crop->dim(1)).arg(crop->dim(2)).arg(r.left()).arg(r.top()));
        } catch (const std::exception& e) {
            log(tr("Crop failed: %1").arg(QString::fromUtf8(e.what())));
        }
    }

    void MainWindow::onTabCloseRequested(int index) {
        if (index < fixedTabs_) return;
        QWidget* w = views_->widget(index);
        views_->removeTab(index);
        delete w;
    }

    void MainWindow::openBandTab(int direction, int bandItem, int stage) {
        if (!diagnostics_ || !result_) return;
        auto* view = new BandView(views_);
        view->setResult(diagnostics_, result_->fit, session_.parameters());
        view->select(direction, bandItem, stage);
        const int i = views_->addTab(view, tr("Band d%1 #%2 %3")
                                               .arg(direction).arg(bandItem)
                                               .arg(stage == 1 ? tr("filtered") : tr("separated")));
        views_->setTabToolTip(i, tr("One captured band in a full viewer (closable)"));
        views_->setCurrentIndex(i);
    }

    void MainWindow::showHelp(const QString& anchor) {
        bottom_->setCurrentWidget(help_);
        help_->scrollToAnchor(anchor);
    }

    void MainWindow::onCaptureToggled(bool on) { session_.setCaptureDiagnostics(on); }

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
        log(tr("Reconstructing on %1 with %2%3 ...")
                .arg(fromStd(toString(device)))
                .arg(session_.usesIdealOtf() ? tr("the ideal OTF") : tr("OTF %1").arg(fileName(session_.otfPath())))
                .arg(session_.captureDiagnostics() ? tr(", capturing intermediate spectra") : QString()));
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
        const SIMParameters& p = session_.parameters();
        resultView_->setPixelSize(p.dx / p.zoomfact, p.dy / p.zoomfact, p.dz / p.z_zoom);
        resultView_->setVolume(resultVolume_);
        applyRawGeometry();   // fitted pattern vectors now overlay both spectra
        showFit(result_->fit);
        saveResultAction_->setEnabled(true);

        if (result_->diagnostics.captured) {
            diagnostics_ = std::make_shared<SimDiagnostics>(std::move(result_->diagnostics));
            bandGrid_->setResult(diagnostics_, result_->fit, p);
        } else {
            diagnostics_.reset();
            bandGrid_->clear();
        }
        views_->setCurrentWidget(resultView_);

        const QString msg = tr("Done in %1 s on %2 (%3, %4)")
                                .arg(result_->seconds, 0, 'f', 2)
                                .arg(fromStd(toString(result_->device)))
                                .arg(result_->plansReused ? tr("plans reused") : tr("plans rebuilt"))
                                .arg(result_->idealOtf ? tr("ideal OTF") : tr("measured OTF"));
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
        applyRawGeometry();
        refreshState();
        otfRefresh_->start();   // recompute the OTF once the edits settle
    }

    // --- state -----------------------------------------------------------

    void MainWindow::setBusy(bool busy) {
        busy_ = busy;
        // nothing that touches the session may be reachable while it runs
        params_->setEnabled(!busy);
        device_->setEnabled(!busy);
        rigor_->setEnabled(!busy);
        capture_->setEnabled(!busy);
        run_->setEnabled(!busy);
        otfIdeal_->setEnabled(!busy);
        menuBar()->setEnabled(!busy);
        if (!busy) refreshState();
    }

    void MainWindow::refreshState() {
        rawLabel_->setText(fileName(session_.rawPath()));
        otfLabel_->setText(session_.usesIdealOtf() ? tr("ideal (from NA)") : fileName(session_.otfPath()));
        otfIdeal_->setEnabled(!busy_ && !session_.usesIdealOtf());
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
