#include "qt/main_window.hpp"

#include <algorithm>
#include <exception>
#include <fstream>

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QBoxLayout>
#include <QCloseEvent>
#include <QDesktopServices>
#include <QDockWidget>
#include <QFileDialog>
#include <QFileInfo>
#include <QInputDialog>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPainter>
#include <QProgressBar>
#include <QSettings>
#include <QStatusBar>
#include <QUrl>

#include <sirius/device.hpp>
#include <sirius/tiff_io.hpp>

#include "core/export.hpp"
#include "qt/dialogs/export_dialog.hpp"
#include "qt/dialogs/open_dataset_dialog.hpp"
#include "qt/dialogs/preferences_dialog.hpp"
#include "qt/panels/assistant_panel.hpp"
#include "qt/panels/diagnostics_panel.hpp"
#include "qt/panels/help_window.hpp"
#include "qt/panels/ops_panel.hpp"
#include "qt/panels/params_panel.hpp"
#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"
#include "qt/viewer/viewer_widget.hpp"
#include "qt/widgets/controls.hpp"

namespace sirius::app {

    namespace {

        constexpr const char* kIssuesUrl = "https://github.com/abcucberkeley/sirius/issues";

        // "✦ Assistant" toggle: 26 px, 1.5 px border, accent fill when open.
        class AssistantButton : public QAbstractButton {
        public:
            explicit AssistantButton(QWidget* parent) : QAbstractButton(parent) {
                setCheckable(true);
                setFixedHeight(26);
                setCursor(Qt::PointingHandCursor);
                setToolTip(QStringLiteral("Assistant (⌥5)"));
                setFont(theme::heading(12));
                setFixedWidth(QFontMetrics(font()).horizontalAdvance(QStringLiteral("Assistant")) + 40);
            }

        protected:
            void paintEvent(QPaintEvent*) override {
                QPainter p(this);
                const bool on = isChecked();
                const QColor border = on ? theme::kAccent : (underMouse() ? theme::kAccent : theme::kText);
                p.setPen(QPen(border, 1.5));
                p.setBrush(on ? QBrush(theme::kAccent) : Qt::NoBrush);
                p.drawRect(QRectF(rect()).adjusted(0.75, 0.75, -0.75, -0.75));
                p.setPen(on ? theme::kBg : theme::kText);
                p.setFont(theme::font(13));
                p.drawText(QRect(10, 0, 16, height()), Qt::AlignVCenter | Qt::AlignLeft, QStringLiteral("✦"));
                p.setFont(theme::heading(12));
                p.drawText(rect().adjusted(27, 0, -10, 0), Qt::AlignVCenter | Qt::AlignLeft, QStringLiteral("Assistant"));
            }
            void enterEvent(QEnterEvent*) override { update(); }
            void leaveEvent(QEvent*) override { update(); }
        };

        QDockWidget* makeDock(const QString& name, QWidget* content, QMainWindow* window) {
            auto* dock = new QDockWidget(name, window);
            dock->setObjectName(name);
            dock->setWidget(content);
            dock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable |
                              QDockWidget::DockWidgetClosable);
            // No title bar chrome: the panels carry their own headers.
            auto* title = new QWidget(dock);
            title->setFixedHeight(0);
            dock->setTitleBarWidget(title);
            return dock;
        }

        QString gpuText(const Workbench& wb) {
            if (!cudaAvailable()) return QStringLiteral("CPU only");
            try {
                const DeviceProperties p = deviceProperties(Device::cuda(wb.cudaDevice()));
                return QStringLiteral("GPU · %1 · %2 GB")
                    .arg(fromStd(p.name))
                    .arg(static_cast<double>(p.totalMemoryBytes) / (1024.0 * 1024.0 * 1024.0), 0, 'f', 1);
            } catch (const std::exception&) {
                return QStringLiteral("GPU");
            }
        }

        QString bytesText(std::size_t bytes) {
            return QStringLiteral("%1 GB").arg(static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), 0, 'f', 1);
        }

    } // namespace

    struct MainWindow::Impl {
        MainWindow* self;
        WorkbenchBridge& bridge;

        ViewerWidget* viewer = nullptr;
        OpsPanel* ops = nullptr;
        ParamsPanel* params = nullptr;
        DiagnosticsPanel* diagnostics = nullptr;
        AssistantPanel* assistant = nullptr;
        HelpWindow* help = nullptr;
        QDockWidget* opsDock = nullptr;
        QDockWidget* paramsDock = nullptr;
        QDockWidget* diagDock = nullptr;
        QDockWidget* assistantDock = nullptr;

        QLabel* datasetLabel = nullptr;
        QLabel* gpuLabel = nullptr;
        AssistantButton* assistantButton = nullptr;

        QLabel* statusShape = nullptr;
        QLabel* statusDtype = nullptr;
        QLabel* statusZoom = nullptr;
        QLabel* statusCursor = nullptr;
        QWidget* statusProgress = nullptr;
        QProgressBar* progressBar = nullptr;
        QLabel* progressText = nullptr;
        QLabel* statusRight = nullptr;

        QMenu* recentMenu = nullptr;
        QAction* undo = nullptr;
        QAction* redo = nullptr;
        QAction* pasteParams = nullptr;
        QAction* cancelRun = nullptr;
        QAction* viewOrtho = nullptr;
        QAction* view3d = nullptr;
        QAction* viewCompare = nullptr;
        QAction* crosshair = nullptr;
        QAction* labels = nullptr;
        QAction* scaleBar = nullptr;
        QAction* syncZT = nullptr;
        QAction* backendCuda = nullptr;
        QAction* backendCpu = nullptr;
        QAction* backendHpc = nullptr;
        QAction* helpPage = nullptr;
        QAction* assistantAction = nullptr;
        QAction* enableStep = nullptr;
        QAction* removeStep = nullptr;
        QAction* duplicateStep = nullptr;
        QAction* moveUp = nullptr;
        QAction* moveDown = nullptr;
        QAction* runSelected = nullptr;
        QAction* runAllAct = nullptr;
        QAction* runTo = nullptr;
        QAction* clearCache = nullptr;
        QAction* clearAll = nullptr;
        QAction* exportResult = nullptr;
        QAction* exportPython = nullptr;
        QAction* exportFigure = nullptr;
        QAction* savePipeline = nullptr;
        QAction* savePipelineAs = nullptr;
        QAction* closeDataset = nullptr;
        QByteArray defaultState;
        QString lastDir;

        Impl(MainWindow* s, WorkbenchBridge& b) : self(s), bridge(b) {}
        Workbench& wb() { return bridge.wb(); }

        // --- building -----------------------------------------------------------
        QAction* action(QMenu* menu, const QString& text, const QKeySequence& key, std::function<void()> fn,
                        const QString& tip = {}) {
            auto* a = menu->addAction(text);
            if (!key.isEmpty()) a->setShortcut(key);
            a->setShortcutContext(Qt::WindowShortcut);
            if (!tip.isEmpty()) a->setStatusTip(tip);
            if (fn) QObject::connect(a, &QAction::triggered, self, [fn] { fn(); });
            return a;
        }

        void buildMenus() {
            auto* bar = new QMenuBar(self);
            bar->setNativeMenuBar(false);
            bar->setFixedHeight(theme::kTitleBarH);
            self->setMenuBar(bar);

            // brand
            auto* brand = new QWidget(bar);
            auto* bl = new QHBoxLayout(brand);
            bl->setContentsMargins(14, 0, 10, 0);
            bl->setSpacing(8);
            bl->addWidget(widgets::colorChip(theme::kAccent, 12, 12, brand));
            bl->addWidget(widgets::heading(QStringLiteral("SIRIUS"), theme::kBrandPx, brand));
            bar->setCornerWidget(brand, Qt::TopLeftCorner);

            // right: dataset · GPU · ✦ Assistant
            auto* right = new QWidget(bar);
            auto* rl = new QHBoxLayout(right);
            rl->setContentsMargins(0, 0, 14, 0);
            rl->setSpacing(18);
            datasetLabel = widgets::label(QString(), 12, theme::kNeutral600, -1, right);
            gpuLabel = widgets::label(gpuText(wb()), 12, theme::kNeutral600, -1, right);
            assistantButton = new AssistantButton(right);
            rl->addWidget(datasetLabel);
            rl->addWidget(gpuLabel);
            rl->addWidget(assistantButton);
            bar->setCornerWidget(right, Qt::TopRightCorner);

            // File
            QMenu* file = bar->addMenu(QStringLiteral("File"));
            action(file, QStringLiteral("Open dataset…"), QKeySequence::Open, [this] { openDataset(); });
            recentMenu = file->addMenu(QStringLiteral("Open recent"));
            closeDataset = action(file, QStringLiteral("Close dataset"), QKeySequence::Close, [this] { wb().closeDataset(); });
            file->addSeparator();
            savePipeline = action(file, QStringLiteral("Save pipeline"), QKeySequence::Save, [this] { savePipelineTo(fromStd(wb().pipelinePath())); });
            savePipelineAs = action(file, QStringLiteral("Save pipeline as…"), QKeySequence::SaveAs, [this] { savePipelineTo(QString()); });
            action(file, QStringLiteral("Load pipeline preset…"), QKeySequence(), [this] { loadPipeline(); });
            file->addSeparator();
            exportResult = action(file, QStringLiteral("Export result…"), QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_E),
                                  [this] { exportResultDialog(); });
            exportPython = action(file, QStringLiteral("Export pipeline as Python script"), QKeySequence(), [this] { exportPythonScript(); });
            exportFigure = action(file, QStringLiteral("Export figure (current view)…"), QKeySequence(Qt::ALT | Qt::CTRL | Qt::Key_E),
                                  [this] { exportFigureImage(); });
            file->addSeparator();
            action(file, QStringLiteral("Preferences…"), QKeySequence::Preferences, [this] { preferences(); });
            file->addSeparator();
            action(file, QStringLiteral("Quit"), QKeySequence::Quit, [this] { self->close(); });

            // Edit
            QMenu* edit = bar->addMenu(QStringLiteral("Edit"));
            undo = action(edit, QStringLiteral("Undo"), QKeySequence::Undo, [this] { wb().undo(); });
            redo = action(edit, QStringLiteral("Redo"), QKeySequence::Redo, [this] { wb().redo(); });
            edit->addSeparator();
            duplicateStep = action(edit, QStringLiteral("Duplicate step"), QKeySequence(Qt::CTRL | Qt::Key_D),
                                   [this] { wb().duplicateStep(wb().selectedIndex()); });
            removeStep = action(edit, QStringLiteral("Remove step"), QKeySequence(Qt::Key_Backspace),
                                [this] { wb().removeStep(wb().selectedIndex()); });
            enableStep = action(edit, QStringLiteral("Enable / skip step"), QKeySequence(Qt::Key_Space), [this] {
                const int i = wb().selectedIndex();
                if (i > 0 && i < wb().pipeline().size()) wb().setStepEnabled(i, !wb().pipeline().at(i).enabled);
            });
            edit->addSeparator();
            moveUp = action(edit, QStringLiteral("Move step up"), QKeySequence(Qt::ALT | Qt::Key_Up),
                            [this] { wb().moveStep(wb().selectedIndex(), -1); });
            moveDown = action(edit, QStringLiteral("Move step down"), QKeySequence(Qt::ALT | Qt::Key_Down),
                              [this] { wb().moveStep(wb().selectedIndex(), +1); });
            edit->addSeparator();
            action(edit, QStringLiteral("Copy parameters"), QKeySequence::Copy, [this] { wb().copyParameters(wb().selectedIndex()); });
            pasteParams = action(edit, QStringLiteral("Paste parameters"), QKeySequence::Paste, [this] {
                if (!wb().pasteParameters(wb().selectedIndex()))
                    wb().logLine("Paste parameters: the copied parameters belong to another operation kind.");
            });

            // View
            QMenu* view = bar->addMenu(QStringLiteral("View"));
            auto* modes = new QActionGroup(self);
            viewOrtho = action(view, QStringLiteral("Ortho views"), QKeySequence(Qt::Key_1), [this] { wb().setViewMode(ViewMode::Ortho); });
            view3d = action(view, QStringLiteral("3D volume"), QKeySequence(Qt::Key_2), [this] { wb().setViewMode(ViewMode::Volume); });
            viewCompare = action(view, QStringLiteral("Compare raw vs. step"), QKeySequence(Qt::Key_3),
                                 [this] { wb().setViewMode(ViewMode::Compare); });
            for (QAction* a : {viewOrtho, view3d, viewCompare}) {
                a->setCheckable(true);
                modes->addAction(a);
            }
            view->addSeparator();
            crosshair = action(view, QStringLiteral("Crosshair"), QKeySequence(Qt::Key_H), [this] { wb().toggleCrosshair(); });
            crosshair->setCheckable(true);
            labels = action(view, QStringLiteral("Labels overlay"), QKeySequence(Qt::Key_L), [this] { wb().toggleLabels(); });
            labels->setCheckable(true);
            scaleBar = action(view, QStringLiteral("Scale bar"), QKeySequence(), [this] {
                ViewState s = wb().viewState();
                s.scaleBar = !s.scaleBar;
                wb().setViewState(s);
            });
            scaleBar->setCheckable(true);
            view->addSeparator();
            action(view, QStringLiteral("Zoom in"), QKeySequence(Qt::Key_Plus), [this] { viewer->zoomIn(); });
            action(view, QStringLiteral("Zoom out"), QKeySequence(Qt::Key_Minus), [this] { viewer->zoomOut(); });
            action(view, QStringLiteral("Fit to window"), QKeySequence(Qt::Key_0), [this] { viewer->fitToWindow(); });
            view->addSeparator();
            action(view, QStringLiteral("Auto contrast (display)"), QKeySequence(Qt::SHIFT | Qt::Key_A),
                   [this] { viewer->autoContrast(); });
            action(view, QStringLiteral("Reset contrast (display)"), QKeySequence(Qt::SHIFT | Qt::Key_R),
                   [this] { viewer->resetContrast(); });
            syncZT = action(view, QStringLiteral("Sync Z / T across viewers"), QKeySequence(), [this] {
                ViewState s = wb().viewState();
                s.syncZT = !s.syncZT;
                wb().setViewState(s);
            });
            syncZT->setCheckable(true);

            // Process
            QMenu* process = bar->addMenu(QStringLiteral("Process"));
            action(process, QStringLiteral("Add operation…"), QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_A), [this] {
                opsDock->show();
                ops->openAddMenu();
            });
            process->addSeparator();
            runAllAct = action(process, QStringLiteral("Run all enabled"), QKeySequence(Qt::CTRL | Qt::Key_R), [this] { bridge.startRun(-1); });
            runSelected = action(process, QStringLiteral("Run selected step"), QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_R),
                                 [this] { bridge.startRun(wb().selectedIndex()); });
            runTo = action(process, QStringLiteral("Run to selected step"), QKeySequence(), [this] { bridge.startRun(wb().selectedIndex()); });
            cancelRun = action(process, QStringLiteral("Cancel"), QKeySequence(Qt::Key_Escape), [this] {
                if (bridge.running()) bridge.cancelRun();
                if (bridge.taskRunning()) bridge.cancelTask();
            });
            process->addSeparator();
            clearCache = action(process, QStringLiteral("Clear cache for step"), QKeySequence(), [this] { wb().clearCache(wb().selectedIndex()); });
            clearAll = action(process, QStringLiteral("Clear all caches"), QKeySequence(), [this] { wb().clearAllCaches(); });
            process->addSeparator();
            action(process, QStringLiteral("Reload plugins"), QKeySequence(), [this] { wb().loadPlugins(true); });
            process->addSeparator();
            auto* backends = new QActionGroup(self);
            backendCuda = action(process, QStringLiteral("Backend: CUDA"), QKeySequence(), [this] { wb().setBackend(Backend::Cuda); });
            backendCpu = action(process, QStringLiteral("Backend: CPU"), QKeySequence(), [this] { wb().setBackend(Backend::Cpu); });
            backendHpc = action(process, QStringLiteral("Backend: HPC (Slurm)"), QKeySequence(), [this] { wb().setBackend(Backend::Hpc); });
            for (QAction* a : {backendCuda, backendCpu, backendHpc}) {
                a->setCheckable(true);
                backends->addAction(a);
            }
            backendCuda->setEnabled(cudaAvailable());
            if (!cudaAvailable()) backendCuda->setStatusTip(QStringLiteral("No CUDA device available"));

            // Segment
            QMenu* segment = bar->addMenu(QStringLiteral("Segment"));
            action(segment, QStringLiteral("Load Torch model…"), QKeySequence(Qt::CTRL | Qt::Key_M), [this] { loadTorchModel(); });
            action(segment, QStringLiteral("Run segmentation"), QKeySequence(), [this] {
                const int i = segmentationStep();
                if (i < 0) wb().logLine("Run segmentation: add a segmentation step first (Process ▸ Add operation).");
                else bridge.startRun(i);
            });
            segment->addSeparator();
            action(segment, QStringLiteral("Paint labels"), QKeySequence(Qt::Key_B), [this] {
                wb().setTool(ViewerTool::Paint);
                wb().setPaintTool(PaintTool::Brush);
                if (!wb().viewState().labels) wb().toggleLabels();
            });
            action(segment, QStringLiteral("Erase"), QKeySequence(Qt::Key_E), [this] {
                wb().setTool(ViewerTool::Paint);
                wb().setPaintTool(PaintTool::Erase);
            });
            action(segment, QStringLiteral("Merge selected labels"), QKeySequence(Qt::CTRL | Qt::Key_G), [this] { mergeLabelsDialog(); });
            action(segment, QStringLiteral("Split label"), QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_G), [this] {
                wb().setTool(ViewerTool::Paint);
                wb().setPaintTool(PaintTool::Split);
                wb().logLine("Split: click two seeds inside the label in the viewer.");
            });
            action(segment, QStringLiteral("Delete label"), QKeySequence(Qt::CTRL | Qt::Key_Backspace), [this] {
                const std::uint32_t id = wb().viewState().selectedLabel;
                if (id) wb().deleteLabel(id);
            });
            segment->addSeparator();
            action(segment, QStringLiteral("Next flagged label"), QKeySequence(Qt::Key_Right), [this] { selectFlagged(true); });
            action(segment, QStringLiteral("Previous flagged label"), QKeySequence(Qt::Key_Left), [this] { selectFlagged(false); });
            segment->addSeparator();
            action(segment, QStringLiteral("Accept all reviewed"), QKeySequence(), [this] { wb().acceptAllReviewed(); });
            segment->addSeparator();
            action(segment, QStringLiteral("Export labels…"), QKeySequence(), [this] { exportLabels(); });

            // Window
            QMenu* window = bar->addMenu(QStringLiteral("Window"));
            QAction* opsToggle = opsDock->toggleViewAction();
            opsToggle->setText(QStringLiteral("Operations"));
            opsToggle->setShortcut(QKeySequence(Qt::ALT | Qt::Key_1));
            QAction* paramsToggle = paramsDock->toggleViewAction();
            paramsToggle->setText(QStringLiteral("Parameters"));
            paramsToggle->setShortcut(QKeySequence(Qt::ALT | Qt::Key_2));
            QAction* diagToggle = diagDock->toggleViewAction();
            diagToggle->setText(QStringLiteral("Diagnostics"));
            diagToggle->setShortcut(QKeySequence(Qt::ALT | Qt::Key_3));
            window->addAction(opsToggle);
            window->addAction(paramsToggle);
            window->addAction(diagToggle);
            helpPage = action(window, QStringLiteral("Help page"), QKeySequence(Qt::ALT | Qt::Key_4), [this] { toggleHelp(); });
            helpPage->setCheckable(true);
            window->addSeparator();
            action(window, QStringLiteral("Float diagnostics"), QKeySequence(), [this] {
                diagDock->show();
                diagDock->setFloating(true);
            });
            action(window, QStringLiteral("Reset layout"), QKeySequence(), [this] { self->restoreState(defaultState); });
            action(window, QStringLiteral("Save layout…"), QKeySequence(), [this] {
                QSettings settings;
                settings.setValue(QStringLiteral("window/state"), self->saveState());
                settings.setValue(QStringLiteral("window/geometry"), self->saveGeometry());
                wb().logLine("Layout saved.");
            });
            window->addSeparator();
            assistantAction = assistantDock->toggleViewAction();
            assistantAction->setText(QStringLiteral("Assistant"));
            assistantAction->setShortcut(QKeySequence(Qt::ALT | Qt::Key_5));
            window->addAction(assistantAction);

            // Help
            QMenu* helpMenu = bar->addMenu(QStringLiteral("Help"));
            action(helpMenu, QStringLiteral("Help for this step"), QKeySequence(Qt::Key_F1), [this] { showHelpForSelected(); });
            action(helpMenu, QStringLiteral("Sirius manual"), QKeySequence(), [this] {
                help->showManual();
                help->show();
                help->raise();
            });
            action(helpMenu, QStringLiteral("Keyboard shortcuts"), QKeySequence(Qt::CTRL | Qt::Key_Slash), [this] {
                help->showShortcuts();
                help->show();
                help->raise();
            });
            helpMenu->addSeparator();
            action(helpMenu, QStringLiteral("Operation plugin API"), QKeySequence(), [this] {
                help->showKind("plugin-api");
                help->show();
                help->raise();
            });
            action(helpMenu, QStringLiteral("Report a problem…"), QKeySequence(),
                   [] { QDesktopServices::openUrl(QUrl(QLatin1String(kIssuesUrl))); });
            helpMenu->addSeparator();
            action(helpMenu, QStringLiteral("About Sirius"), QKeySequence(), [this] { about(); });

            QObject::connect(assistantButton, &QAbstractButton::toggled, self, [this](bool on) { assistantDock->setVisible(on); });
            QObject::connect(assistantDock, &QDockWidget::visibilityChanged, self, [this](bool on) {
                QSignalBlocker b(assistantButton);
                assistantButton->setChecked(on);
                if (on) assistant->focusInput();
            });
            QObject::connect(recentMenu, &QMenu::aboutToShow, self, [this] { rebuildRecent(); });
        }

        void buildStatusBar() {
            auto* sb = self->statusBar();
            sb->setSizeGripEnabled(false);
            auto* left = new QWidget(sb);
            auto* ll = new QHBoxLayout(left);
            ll->setContentsMargins(14, 0, 0, 0);
            ll->setSpacing(24);
            statusShape = widgets::label(QString(), 11, theme::kNeutral600, -1, left);
            statusDtype = widgets::label(QString(), 11, theme::kNeutral600, -1, left);
            statusZoom = widgets::label(QStringLiteral("zoom 100 %"), 11, theme::kNeutral600, -1, left);
            statusCursor = widgets::label(QString(), 11, theme::kNeutral600, -1, left);
            statusProgress = new QWidget(left);
            auto* pl = new QHBoxLayout(statusProgress);
            pl->setContentsMargins(0, 0, 0, 0);
            pl->setSpacing(8);
            progressBar = new QProgressBar(statusProgress);
            progressBar->setRange(0, 1000);
            progressBar->setTextVisible(false);
            progressBar->setFixedSize(120, 4);
            progressText = widgets::label(QStringLiteral("0 %"), 11, theme::kText, -1, statusProgress);
            pl->addWidget(progressBar);
            pl->addWidget(progressText);
            statusProgress->hide();
            for (QWidget* w : {static_cast<QWidget*>(statusShape), static_cast<QWidget*>(statusDtype),
                               static_cast<QWidget*>(statusZoom), static_cast<QWidget*>(statusCursor), statusProgress})
                ll->addWidget(w);
            ll->addStretch(1);
            sb->addWidget(left, 1);
            statusRight = widgets::label(QString(), 11, theme::kNeutral600, -1, sb);
            statusRight->setContentsMargins(0, 0, 14, 0);
            sb->addPermanentWidget(statusRight);
        }

        // --- state refresh ------------------------------------------------------------
        void refreshTitle() {
            const Workbench& w = wb();
            const QString name = w.hasDataset() ? fromStd(w.dataset().name) : QStringLiteral("no dataset");
            datasetLabel->setText(name);
            gpuLabel->setText(gpuText(w));
            QString title = QStringLiteral("SIRIUS");
            if (w.hasDataset()) title += QStringLiteral(" — ") + name;
            if (!w.pipelinePath().empty()) title += QStringLiteral(" · ") + QFileInfo(fromStd(w.pipelinePath())).fileName();
            self->setWindowTitle(title);
        }

        void refreshStatus() {
            const Workbench& w = wb();
            if (w.hasDataset()) {
                const DatasetMeta& m = w.dataset();
                statusShape->setText(fromStd(m.shapeString()));
                statusDtype->setText(QStringLiteral("%1 → float32").arg(QString::fromLatin1(toString(m.sourceType))));
            } else {
                statusShape->setText(QStringLiteral("no dataset"));
                statusDtype->clear();
            }
            statusZoom->setText(QStringLiteral("zoom %1").arg(viewer->zoomText()));
            statusCursor->setText(viewer->cursorText());
            const Pipeline& p = w.pipeline();
            const bool lazy = !w.output(0) || !w.output(0)->array;
            statusRight->setText(QStringLiteral("%1 of %2 steps enabled · %3 · %4 cached")
                                     .arg(p.enabledCount())
                                     .arg(p.size())
                                     .arg(lazy ? QStringLiteral("lazy") : QStringLiteral("in memory"))
                                     .arg(bytesText(w.cachedBytes())));
        }

        void refreshActions() {
            Workbench& w = wb();   // history() is non-const
            const int i = w.selectedIndex();
            const Pipeline& p = w.pipeline();
            const bool stepOk = i >= 0 && i < p.size();
            const bool movable = stepOk && i > 0;
            const bool running = bridge.running();
            undo->setEnabled(w.history().canUndo());
            undo->setText(w.history().canUndo() ? QStringLiteral("Undo %1").arg(fromStd(w.history().undoLabel()))
                                                : QStringLiteral("Undo"));
            redo->setEnabled(w.history().canRedo());
            redo->setText(w.history().canRedo() ? QStringLiteral("Redo %1").arg(fromStd(w.history().redoLabel()))
                                                : QStringLiteral("Redo"));
            pasteParams->setEnabled(w.hasCopiedParameters() && stepOk);
            duplicateStep->setEnabled(movable);
            removeStep->setEnabled(movable && !running);
            enableStep->setEnabled(movable);
            moveUp->setEnabled(movable && i > 1);
            moveDown->setEnabled(movable && i < p.size() - 1);
            runAllAct->setEnabled(w.hasDataset() && !running);
            runSelected->setEnabled(w.hasDataset() && !running && stepOk);
            runTo->setEnabled(w.hasDataset() && !running && stepOk);
            cancelRun->setEnabled(running || bridge.taskRunning());
            clearCache->setEnabled(stepOk);
            clearAll->setEnabled(!running);
            exportResult->setEnabled(w.hasDataset() && !running);
            exportPython->setEnabled(true);
            closeDataset->setEnabled(w.hasDataset() && !running);
            savePipeline->setEnabled(true);
            const ViewState& v = w.viewState();
            viewOrtho->setChecked(v.mode == ViewMode::Ortho);
            view3d->setChecked(v.mode == ViewMode::Volume);
            viewCompare->setChecked(v.mode == ViewMode::Compare);
            crosshair->setChecked(v.crosshair);
            labels->setChecked(v.labels);
            scaleBar->setChecked(v.scaleBar);
            syncZT->setChecked(v.syncZT);
            backendCuda->setChecked(w.backend() == Backend::Cuda);
            backendCpu->setChecked(w.backend() == Backend::Cpu);
            backendHpc->setChecked(w.backend() == Backend::Hpc);
        }

        void rebuildRecent() {
            recentMenu->clear();
            const QStringList recent = OpenDatasetDialog::recentFiles();
            if (recent.isEmpty()) {
                recentMenu->addAction(QStringLiteral("No recent datasets"))->setEnabled(false);
                return;
            }
            for (const QString& path : recent) {
                QAction* a = recentMenu->addAction(QFileInfo(path).fileName());
                a->setToolTip(path);
                QObject::connect(a, &QAction::triggered, self, [this, path] { self->openDatasetPath(path); });
            }
            recentMenu->addSeparator();
            QAction* clear = recentMenu->addAction(QStringLiteral("Clear list"));
            QObject::connect(clear, &QAction::triggered, self, [] {
                QSettings settings;
                settings.remove(QStringLiteral("recent/datasets"));
            });
        }

        // --- commands -----------------------------------------------------------------
        void openDataset() {
            OpenDatasetDialog dialog(self, wb().hasDataset() ? fromStd(wb().dataset().sourcePath) : QString());
            if (dialog.exec() != QDialog::Accepted) return;
            openWith(dialog.path(), dialog.options());
        }

        void openWith(const QString& path, const OpenOptions& options) {
            try {
                wb().openDataset(toStd(path), options);
                OpenDatasetDialog::addRecentFile(path);
                lastDir = QFileInfo(path).absolutePath();
            } catch (const std::exception& e) {
                wb().logLine(std::string("Open failed: ") + e.what());
                QMessageBox::warning(self, QStringLiteral("Open dataset"), QString::fromUtf8(e.what()));
            }
        }

        void savePipelineTo(QString path) {
            if (path.isEmpty()) {
                path = QFileDialog::getSaveFileName(self, QStringLiteral("Save pipeline"), lastDir,
                                                    QStringLiteral("SIRIUS pipeline (*.sirius.toml *.toml)"));
                if (path.isEmpty()) return;
                if (!path.endsWith(QLatin1String(".toml"))) path += QStringLiteral(".sirius.toml");
            }
            try {
                wb().savePipeline(toStd(path));
                lastDir = QFileInfo(path).absolutePath();
            } catch (const std::exception& e) {
                QMessageBox::warning(self, QStringLiteral("Save pipeline"), QString::fromUtf8(e.what()));
            }
            refreshTitle();
        }

        void loadPipeline() {
            const QString path = QFileDialog::getOpenFileName(self, QStringLiteral("Load pipeline"), lastDir,
                                                              QStringLiteral("SIRIUS pipeline (*.sirius.toml *.toml);;All files (*)"));
            if (!path.isEmpty()) self->openPipelinePath(path);
        }

        void exportResultDialog() {
            if (!wb().hasDataset()) return;
            ExportDialog dialog(bridge, self);
            if (dialog.exec() != QDialog::Accepted) return;
            const int step = dialog.stepIndex();
            ExportOptions options = dialog.options();
            std::shared_ptr<const StepOutput> out = wb().output(step);
            if (!out) {
                QMessageBox::information(self, QStringLiteral("Export"),
                                         QStringLiteral("Step %1 has not been computed yet. Run it first.").arg(fromStd(Step::number(step))));
                return;
            }
            const std::string pipelinePath = options.path + ".pipeline.toml";
            const bool sidecar = options.includePipeline;
            options.includePipeline = false;
            if (sidecar) {
                try {
                    wb().savePipeline(pipelinePath);
                } catch (const std::exception& e) {
                    wb().logLine(std::string("Pipeline sidecar: ") + e.what());
                }
            }
            bridge.startTask(QStringLiteral("Export"), [out, options](const WorkbenchBridge::TaskProgress& progress,
                                                                     const WorkbenchBridge::TaskCancelled& cancelled) {
                ArrayPtr array = out->asInput().materialize(progress);
                exportArray(*array, out->meta, out->labels.get(), options, progress, cancelled);
            });
        }

        void exportPythonScript() {
            const QString path = QFileDialog::getSaveFileName(self, QStringLiteral("Export pipeline as Python"), lastDir,
                                                              QStringLiteral("Python (*.py)"));
            if (path.isEmpty()) return;
            const std::string script = wb().pipeline().toPythonScript(wb().hasDataset() ? wb().dataset().sourcePath : std::string());
            std::ofstream f(toStd(path));
            if (!f) {
                QMessageBox::warning(self, QStringLiteral("Export"), QStringLiteral("Cannot write %1").arg(path));
                return;
            }
            f << script;
            wb().logLine("Python script written to " + toStd(path));
        }

        void exportFigureImage() {
            const QImage image = viewer->grabView();
            if (image.isNull()) return;
            const QString path = QFileDialog::getSaveFileName(self, QStringLiteral("Export figure"), lastDir,
                                                              QStringLiteral("PNG (*.png);;TIFF (*.tif)"));
            if (path.isEmpty()) return;
            if (!image.save(path)) QMessageBox::warning(self, QStringLiteral("Export figure"), QStringLiteral("Cannot write %1").arg(path));
            else wb().logLine("Figure written to " + toStd(path));
        }

        void exportLabels() {
            std::shared_ptr<LabelVolume> labelsVol = wb().viewedLabels();
            if (!labelsVol || labelsVol->empty()) {
                wb().logLine("Export labels: the viewed step has no labels.");
                return;
            }
            const QString path = QFileDialog::getSaveFileName(self, QStringLiteral("Export labels"), lastDir,
                                                              QStringLiteral("TIFF (*.tif)"));
            if (path.isEmpty()) return;
            try {
                writeTiffStack<std::uint32_t>(toStd(path), labelsVol->view().asStack(), TiffCompression::Deflate);
                wb().logLine("Labels written to " + toStd(path));
            } catch (const std::exception& e) {
                QMessageBox::warning(self, QStringLiteral("Export labels"), QString::fromUtf8(e.what()));
            }
        }

        void preferences() {
            PreferencesDialog dialog(bridge, self);
            QObject::connect(&dialog, &PreferencesDialog::assistantSettingsChanged, self,
                             [this] { assistant->setSettings(AssistantSettings::load()); });
            dialog.exec();
            refreshTitle();
        }

        int segmentationStep() const {
            const Pipeline& p = bridge.wb().pipeline();
            for (int i = p.size() - 1; i >= 0; --i)
                if (p.at(i).op().info().producesLabels && p.at(i).kind != "threshold") return i;
            for (int i = p.size() - 1; i >= 0; --i)
                if (p.at(i).op().info().producesLabels) return i;
            return -1;
        }

        void loadTorchModel() {
            int i = segmentationStep();
            if (i < 0 || wb().pipeline().at(i).kind != "seg") {
                if (!findOperation("seg")) return;
                i = wb().pipeline().indexOf(wb().addStep("seg"));
            }
            const Step& s = wb().pipeline().at(i);
            std::string pathKey;
            for (const ParamSpec& spec : s.op().info().params)
                if (spec.type == ParamType::Path) {
                    pathKey = spec.key;
                    break;
                }
            const QString path = QFileDialog::getOpenFileName(self, QStringLiteral("Load Torch model"), lastDir,
                                                              QStringLiteral("TorchScript / ONNX (*.pt *.pth *.ts *.onnx);;All files (*)"));
            if (path.isEmpty() || pathKey.empty()) return;
            wb().setStepParam(i, pathKey, toStd(path));
            wb().select(i);
        }

        void mergeLabelsDialog() {
            bool ok = false;
            const QString text = QInputDialog::getText(self, QStringLiteral("Merge labels"),
                                                       QStringLiteral("Label ids to merge (comma-separated)"), QLineEdit::Normal,
                                                       QString::number(wb().viewState().selectedLabel), &ok);
            if (!ok) return;
            std::vector<std::uint32_t> ids;
            for (const QString& part : text.split(QLatin1Char(','), Qt::SkipEmptyParts)) {
                bool num = false;
                const uint v = part.trimmed().toUInt(&num);
                if (num && v > 0) ids.push_back(v);
            }
            if (ids.size() >= 2) wb().mergeLabels(ids);
            else wb().logLine("Merge labels: give at least two label ids.");
        }

        void selectFlagged(bool forward) {
            const std::uint32_t id = wb().nextFlaggedLabel(forward);
            if (!id) {
                wb().logLine("No flagged labels.");
                return;
            }
            ViewState s = wb().viewState();
            s.selectedLabel = id;
            s.labels = true;
            wb().setViewState(s);
        }

        void toggleHelp() {
            if (help->isVisible()) help->hide();
            else showHelpForSelected();
        }

        void showHelpForSelected() {
            const int i = wb().selectedIndex();
            if (i >= 0 && i < wb().pipeline().size()) help->showKind(wb().pipeline().at(i).kind);
            help->show();
            help->raise();
        }

        void about() {
            QMessageBox::about(self, QStringLiteral("About SIRIUS"),
                               QStringLiteral("<b>SIRIUS</b> — Structured Illumination Reconstruction and Image Utility Suite<br>"
                                              "Microscopy processing workbench.<br><br>"
                                              "CPU, CUDA and HPC backends · TIFF, OME-TIFF%1 · Qt %2<br>"
                                              "<a href=\"https://github.com/abcucberkeley/sirius\">github.com/abcucberkeley/sirius</a>")
                                   .arg(zarrSupported() ? QStringLiteral(", zarr / N5 (TensorStore)") : QString(), QLatin1String(qVersion())));
        }
    };

    MainWindow::MainWindow(WorkbenchBridge& bridge, QWidget* parent)
        : QMainWindow(parent), impl_(std::make_unique<Impl>(this, bridge)) {
        setObjectName(QStringLiteral("MainWindow"));
        resize(1600, 960);
        setDockOptions(QMainWindow::AnimatedDocks | QMainWindow::AllowNestedDocks | QMainWindow::AllowTabbedDocks);
        setDockNestingEnabled(true);
        Impl& d = *impl_;

        d.viewer = new ViewerWidget(bridge, this);
        setCentralWidget(d.viewer);

        d.ops = new OpsPanel(bridge, this);
        d.params = new ParamsPanel(bridge, this);
        d.diagnostics = new DiagnosticsPanel(bridge, this);
        d.assistant = new AssistantPanel(bridge, this);
        d.help = new HelpWindow(bridge, this);
        d.help->hide();

        d.opsDock = makeDock(QStringLiteral("Operations"), d.ops, this);
        d.paramsDock = makeDock(QStringLiteral("Parameters"), d.params, this);
        d.diagDock = makeDock(QStringLiteral("Diagnostics"), d.diagnostics, this);
        d.assistantDock = makeDock(QStringLiteral("Assistant"), d.assistant, this);
        d.diagnostics->setDock(d.diagDock);
        addDockWidget(Qt::LeftDockWidgetArea, d.opsDock);
        addDockWidget(Qt::RightDockWidgetArea, d.paramsDock);
        addDockWidget(Qt::RightDockWidgetArea, d.assistantDock);
        splitDockWidget(d.paramsDock, d.assistantDock, Qt::Horizontal);
        addDockWidget(Qt::BottomDockWidgetArea, d.diagDock);
        d.assistantDock->hide();
        // Sizing a hidden dock here makes the first layout pass reserve its
        // width and grow the window past 1600 px; the assistant dock is
        // sized when it is first shown instead (toggleAssistant).
        resizeDocks({d.opsDock, d.paramsDock}, {theme::kOpsDockW, theme::kParamsDockW}, Qt::Horizontal);
        resizeDocks({d.diagDock}, {theme::kDiagnosticsH}, Qt::Vertical);
        setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);
        setCorner(Qt::BottomRightCorner, Qt::RightDockWidgetArea);

        d.buildMenus();
        d.buildStatusBar();
        d.defaultState = saveState();

        // panel <-> window wiring
        connect(d.ops, &OpsPanel::exportRequested, this, [this] { impl_->exportResultDialog(); });
        connect(d.params, &ParamsPanel::helpRequested, this, [this](bool open) {
            if (open) impl_->showHelpForSelected();
            else impl_->help->hide();
        });
        connect(d.help, &HelpWindow::visibilityChanged, this, [this](bool visible) {
            impl_->params->setHelpOpen(visible);
            impl_->helpPage->setChecked(visible);
        });
        connect(d.assistant, &AssistantPanel::closeRequested, this, [this] { impl_->assistantDock->hide(); });
        connect(d.viewer, &ViewerWidget::cursorChanged, this, [this](const QString& t) { impl_->statusCursor->setText(t); });
        connect(d.viewer, &ViewerWidget::zoomChanged, this, [this](const QString& t) {
            impl_->statusZoom->setText(QStringLiteral("zoom %1").arg(t));
        });
        connect(&bridge, &WorkbenchBridge::selectionChanged, this, [this] {
            impl_->refreshActions();
            if (impl_->help->isVisible()) impl_->showHelpForSelected();
        });
        auto refreshAll = [this] {
            impl_->refreshTitle();
            impl_->refreshStatus();
            impl_->refreshActions();
        };
        connect(&bridge, &WorkbenchBridge::datasetChanged, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::pipelineChanged, this, refreshAll);
        connect(&bridge, &WorkbenchBridge::stepChanged, this, [this](int) { impl_->refreshStatus(); impl_->refreshActions(); });
        connect(&bridge, &WorkbenchBridge::viewStateChanged, this, [this] { impl_->refreshActions(); });
        connect(&bridge, &WorkbenchBridge::outputsChanged, this, [this] { impl_->refreshStatus(); });
        connect(&bridge, &WorkbenchBridge::historyChanged, this, [this] { impl_->refreshActions(); });
        connect(&bridge, &WorkbenchBridge::backendChanged, this, [this] {
            impl_->refreshActions();
            impl_->refreshTitle();
        });
        connect(&bridge, &WorkbenchBridge::runStarted, this, [this] {
            impl_->statusProgress->show();
            impl_->progressBar->setValue(0);
            impl_->progressText->setText(QStringLiteral("0 %"));
            impl_->refreshActions();
        });
        connect(&bridge, &WorkbenchBridge::runProgress, this, [this](double f, int, const QString& msg) {
            impl_->statusProgress->show();
            impl_->progressBar->setValue(static_cast<int>(f * 1000.0));
            impl_->progressText->setText(QStringLiteral("%1 %").arg(static_cast<int>(f * 100.0 + 0.5)));
            if (!msg.isEmpty()) statusBar()->showMessage(msg, 2000);
        });
        connect(&bridge, &WorkbenchBridge::runFinished, this, [this](bool ok, const QString& error) {
            impl_->statusProgress->hide();
            refreshAllLater();
            if (!ok && !error.isEmpty() && error != QLatin1String("cancelled"))
                QMessageBox::warning(this, QStringLiteral("Run failed"), error);
        });
        connect(&bridge, &WorkbenchBridge::taskStarted, this, [this](const QString&) {
            impl_->statusProgress->show();
            impl_->progressBar->setValue(0);
            impl_->refreshActions();
        });
        connect(&bridge, &WorkbenchBridge::taskProgress, this, [this](double f, const QString& msg) {
            impl_->progressBar->setValue(static_cast<int>(f * 1000.0));
            impl_->progressText->setText(QStringLiteral("%1 %").arg(static_cast<int>(f * 100.0 + 0.5)));
            if (!msg.isEmpty()) statusBar()->showMessage(msg, 2000);
        });
        connect(&bridge, &WorkbenchBridge::taskFinished, this, [this](bool ok, const QString& error) {
            impl_->statusProgress->hide();
            impl_->refreshActions();
            if (!ok && !error.isEmpty()) QMessageBox::warning(this, impl_->bridge.taskLabel(), error);
        });
        connect(&bridge, &WorkbenchBridge::logged, this, [this](const QString& line) { statusBar()->showMessage(line, 4000); });

        QSettings settings;
        if (settings.contains(QStringLiteral("window/geometry"))) restoreGeometry(settings.value(QStringLiteral("window/geometry")).toByteArray());
        if (settings.contains(QStringLiteral("window/state"))) restoreState(settings.value(QStringLiteral("window/state")).toByteArray());
        refreshAll();
    }

    MainWindow::~MainWindow() = default;

    void MainWindow::refreshAllLater() {
        impl_->refreshTitle();
        impl_->refreshStatus();
        impl_->refreshActions();
    }

    void MainWindow::openDatasetPath(const QString& path) { impl_->openWith(path, OpenOptions{}); }

    void MainWindow::openPipelinePath(const QString& path) {
        try {
            impl_->wb().loadPipeline(toStd(path));
            impl_->lastDir = QFileInfo(path).absolutePath();
        } catch (const std::exception& e) {
            impl_->wb().logLine(std::string("Load pipeline failed: ") + e.what());
            QMessageBox::warning(this, QStringLiteral("Load pipeline"), QString::fromUtf8(e.what()));
        }
        refreshAllLater();
    }

    void MainWindow::runAll() { impl_->bridge.startRun(-1); }

    void MainWindow::askAssistant(const QString& text) {
        impl_->assistantDock->show();
        impl_->assistant->ask(text);
    }

    void MainWindow::closeEvent(QCloseEvent* event) {
        if (impl_->bridge.running()) {
            const auto answer = QMessageBox::question(this, QStringLiteral("Quit"),
                                                      QStringLiteral("A run is in progress. Cancel it and quit?"));
            if (answer != QMessageBox::Yes) {
                event->ignore();
                return;
            }
            impl_->bridge.cancelRun();
        }
        QSettings settings;
        settings.setValue(QStringLiteral("window/geometry"), saveGeometry());
        settings.setValue(QStringLiteral("window/state"), saveState());
        impl_->help->hide();
        QMainWindow::closeEvent(event);
    }

} // namespace sirius::app
