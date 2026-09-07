#include "qt/dialogs/model_hub_dialog.hpp"

#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <vector>

#include <QBoxLayout>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMetaObject>
#include <QPointer>
#include <QProgressBar>
#include <QPushButton>
#include <QStyle>
#include <QTabWidget>
#include <QTableWidget>
#include <QThread>

#include "core/ops/builtin.hpp"
#include "qt/qt_strings.hpp"
#include "qt/theme.hpp"
#include "qt/widgets/controls.hpp"
#include "qt/worker_launcher.hpp"

namespace sirius::app {

    using widgets::CaptionLabel;
    using widgets::Rule;

    namespace {

        // Every hub call goes through the Python worker (huggingface_hub and
        // the model packages live there, and the HPC backend shares the
        // cache logic). The dialog spawns its own worker through a
        // WorkerLauncher created on the hub thread -- QProcess must be
        // driven from the thread that created it -- and posts results back
        // to the dialog with queued invocations, so the UI never blocks on
        // the network or on the worker's start-up.
        class HubClient : public QObject {
        public:
            std::atomic<bool> cancel{false};

            RemoteWorker& worker() {
                if (!launcher_) launcher_ = std::make_unique<WorkerLauncher>();
                if (!remote_ || !remote_->isOpen()) remote_ = launcher_->connect();
                return *remote_;
            }

            ~HubClient() override {
                remote_.reset();
                launcher_.reset();   // stops the worker process
            }

        private:
            std::unique_ptr<WorkerLauncher> launcher_;
            std::unique_ptr<RemoteWorker> remote_;
        };

        QString countText(long long n) {
            if (n >= 1000000) return QStringLiteral("%1M").arg(static_cast<double>(n) / 1e6, 0, 'f', 1);
            if (n >= 1000) return QStringLiteral("%1k").arg(static_cast<double>(n) / 1e3, 0, 'f', n >= 10000 ? 0 : 1);
            return QString::number(n);
        }

        QString bytesText(long long n) { return n < 0 ? QStringLiteral("?") : fromStd(formatBytes(static_cast<std::uint64_t>(n))); }

        QTableWidget* makeTable(const QStringList& headers, int stretchColumn, QWidget* parent) {
            auto* t = new QTableWidget(0, headers.size(), parent);
            t->setHorizontalHeaderLabels(headers);
            for (int c = 0; c < headers.size(); ++c)
                t->horizontalHeader()->setSectionResizeMode(c, c == stretchColumn ? QHeaderView::Stretch : QHeaderView::ResizeToContents);
            t->verticalHeader()->hide();
            t->setSelectionBehavior(QAbstractItemView::SelectRows);
            t->setSelectionMode(QAbstractItemView::SingleSelection);
            t->setEditTriggers(QAbstractItemView::NoEditTriggers);
            t->setShowGrid(false);
            return t;
        }

        QTableWidgetItem* cell(const QString& text, const QString& data = {}) {
            auto* it = new QTableWidgetItem(text);
            if (!data.isEmpty()) it->setData(Qt::UserRole, data);
            return it;
        }

        void setLabelClass(QLabel* label, const QString& cls) {
            label->setProperty("class", cls);
            label->style()->unpolish(label);
            label->style()->polish(label);
        }

        struct FamilyCard {
            QString spec;
            QLabel* status = nullptr;
            QPushButton* use = nullptr;
        };

    } // namespace

    struct ModelHubDialog::Impl {
        ModelHubDialog* dialog;
        WorkbenchBridge& bridge;
        QString chosen;
        QThread thread;
        HubClient* client = nullptr;   // lives on `thread`

        QLabel* status = nullptr;
        QLabel* selected = nullptr;
        QPushButton* ok = nullptr;

        // Hugging Face
        QLineEdit* query = nullptr;
        QPushButton* search = nullptr;
        QTableWidget* results = nullptr;
        QTableWidget* files = nullptr;
        QPushButton* download = nullptr;
        QPushButton* useFile = nullptr;
        QProgressBar* progress = nullptr;
        QLabel* fileNote = nullptr;
        QString repo;
        QString downloadedPath;   // of the selected file, once it is in the cache

        // families and the local cache
        std::vector<FamilyCard> cards;
        QTableWidget* cache = nullptr;
        QLabel* cacheNote = nullptr;
        bool familiesChecked = false;
        bool cacheListed = false;

        Impl(ModelHubDialog* d, WorkbenchBridge& b) : dialog(d), bridge(b) {}

        void choose(const QString& spec) {
            chosen = spec;
            selected->setText(spec.isEmpty() ? QStringLiteral("No model chosen") : QStringLiteral("Model: %1").arg(spec));
            ok->setEnabled(!spec.isEmpty());
        }

        void setStatus(const QString& text, bool error) {
            status->setText(text);
            setLabelClass(status, error ? QStringLiteral("error") : QStringLiteral("small"));
        }

        // Calls `method` on the hub thread and `done(result)` back on the
        // dialog's; a throw becomes a status line. Results cross threads as
        // JSON text. Progress frames (downloads) reach onProgress.
        void callWorker(const QString& what, const nlohmann::json& params, const std::string& method,
                        std::function<void(const nlohmann::json&)> done, bool withProgress = false) {
            setStatus(what + QStringLiteral("…"), false);
            QPointer<ModelHubDialog> self(dialog);
            HubClient* c = client;
            Impl* impl = this;
            QMetaObject::invokeMethod(
                c,
                [self, c, impl, what, params, method, done, withProgress] {
                    std::string text;
                    std::string error;
                    try {
                        RemoteWorker& w = c->worker();
                        std::function<void(double, const std::string&)> progress;
                        if (withProgress)
                            progress = [self, impl](double f, const std::string& m) {
                                if (!self) return;
                                QMetaObject::invokeMethod(
                                    self.data(), [self, impl, f, m] { if (self) impl->onProgress(f, fromStd(m)); }, Qt::QueuedConnection);
                            };
                        WorkerResult r = w.call(method, params, {}, progress, [c] { return c->cancel.load(); });
                        text = r.result.dump();
                    } catch (const std::exception& e) {
                        error = e.what();
                    }
                    if (!self) return;
                    QMetaObject::invokeMethod(
                        self.data(),
                        [self, impl, what, text, error, done] {
                            if (!self) return;
                            if (!error.empty()) {
                                impl->setStatus(what + QStringLiteral(" failed: ") + fromStd(error), true);
                                return;
                            }
                            impl->setStatus(QString(), false);
                            try {
                                done(nlohmann::json::parse(text));
                            } catch (const std::exception& e) {
                                impl->setStatus(what + QStringLiteral(": ") + QString::fromUtf8(e.what()), true);
                            }
                        },
                        Qt::QueuedConnection);
                },
                Qt::QueuedConnection);
        }

        void onProgress(double fraction, const QString& message) {
            progress->setValue(static_cast<int>(fraction * 1000.0));
            if (!message.isEmpty()) setStatus(QStringLiteral("Downloading… ") + message, false);
        }

        void runSearch() {
            const QString q = query->text().trimmed();
            results->setRowCount(0);
            files->setRowCount(0);
            download->setEnabled(false);
            useFile->setEnabled(false);
            repo.clear();
            callWorker(QStringLiteral("Searching Hugging Face"), {{"query", toStd(q)}, {"limit", 40}}, "hub_search",
                       [this](const nlohmann::json& r) {
                           const nlohmann::json models = r.value("models", nlohmann::json::array());
                           results->setRowCount(0);
                           for (const nlohmann::json& m : models) {
                               const int row = results->rowCount();
                               results->insertRow(row);
                               const QString id = fromStd(m.value("id", std::string()));
                               auto* idItem = cell(id, id);
                               idItem->setToolTip(id);
                               results->setItem(row, 0, idItem);
                               results->setItem(row, 1, cell(countText(m.value("downloads", 0LL))));
                               results->setItem(row, 2, cell(countText(m.value("likes", 0LL))));
                               QStringList tags;
                               for (const nlohmann::json& t : m.value("tags", nlohmann::json::array()))
                                   if (t.is_string()) tags << fromStd(t.get<std::string>());
                               auto* tagItem = cell(tags.join(QStringLiteral(", ")));
                               tagItem->setToolTip(tags.join(QStringLiteral("\n")));
                               results->setItem(row, 3, tagItem);
                           }
                           setStatus(models.empty() ? QStringLiteral("No models found.") : QStringLiteral("%1 models").arg(models.size()), false);
                       });
        }

        void listFiles(const QString& id) {
            if (id == repo) return;
            repo = id;
            files->setRowCount(0);
            download->setEnabled(false);
            useFile->setEnabled(false);
            downloadedPath.clear();
            callWorker(QStringLiteral("Listing files of ") + id, {{"repo", toStd(id)}}, "hub_files", [this, id](const nlohmann::json& r) {
                if (id != repo) return;
                files->setRowCount(0);
                int firstModel = -1;
                for (const nlohmann::json& f : r.value("files", nlohmann::json::array())) {
                    const int row = files->rowCount();
                    files->insertRow(row);
                    const QString name = fromStd(f.value("name", std::string()));
                    auto* nameItem = cell(name, name);
                    const bool model = f.value("model", false);
                    if (!model) nameItem->setForeground(theme::kNeutral500);
                    else if (firstModel < 0) firstModel = row;
                    files->setItem(row, 0, nameItem);
                    files->setItem(row, 1, cell(bytesText(f.value("size", -1LL))));
                }
                if (firstModel >= 0) files->selectRow(firstModel);
                fileNote->setText(firstModel >= 0
                                      ? QStringLiteral("Model files (.pt, .pts, .pth, .onnx) are highlighted; the rest is shown for reference.")
                                      : QStringLiteral("This repository has no TorchScript / ONNX file; SIRIUS cannot run its weights directly."));
            });
        }

        void fileSelected() {
            const auto items = files->selectedItems();
            downloadedPath.clear();
            useFile->setEnabled(false);
            progress->setValue(0);
            if (items.isEmpty()) {
                download->setEnabled(false);
                return;
            }
            const QString file = files->item(items.first()->row(), 0)->data(Qt::UserRole).toString();
            download->setEnabled(true);
            // already in the cache? then "Use" is available right away
            const QString spec = QStringLiteral("hf:%1:%2").arg(repo, file);
            callWorker(QStringLiteral("Checking the cache"), {{"spec", toStd(spec)}}, "model_info", [this, spec](const nlohmann::json& info) {
                if (fromStd(info.value("spec", std::string())) != spec || !info.value("cached", true)) return;
                const QString path = fromStd(info.value("path", std::string()));
                if (path.isEmpty()) return;
                downloadedPath = path;
                progress->setValue(1000);
                useFile->setEnabled(true);
                setStatus(QStringLiteral("In the cache: %1 · %2").arg(path, fromStd(torchModelSummary(info))), false);
            });
        }

        void startDownload() {
            const auto items = files->selectedItems();
            if (items.isEmpty() || repo.isEmpty()) return;
            const QString file = files->item(items.first()->row(), 0)->data(Qt::UserRole).toString();
            download->setEnabled(false);
            useFile->setEnabled(false);
            progress->setValue(0);
            client->cancel = false;
            const QString id = repo;
            callWorker(QStringLiteral("Downloading ") + file, {{"repo", toStd(id)}, {"file", toStd(file)}}, "hub_download",
                       [this, id, file](const nlohmann::json& r) {
                           const QString path = fromStd(r.value("path", std::string()));
                           download->setEnabled(true);
                           progress->setValue(1000);
                           if (id == repo) {
                               downloadedPath = path;
                               useFile->setEnabled(true);
                           }
                           cacheListed = false;   // the Local tab lists it on its next visit
                           setStatus(QStringLiteral("Downloaded %1 (%2) to %3").arg(file, bytesText(r.value("bytes", -1LL)), path), false);
                       },
                       true);
        }

        void checkFamilies() {
            familiesChecked = true;
            for (const FamilyCard& card : cards) {
                QLabel* status = card.status;
                callWorker(QStringLiteral("Checking ") + card.spec, {{"spec", toStd(card.spec)}}, "model_info", [status](const nlohmann::json& info) {
                    const bool available = info.value("available", false);
                    const QString hint = fromStd(info.value("install_hint", std::string()));
                    status->setText(available ? QStringLiteral("available in the worker's Python")
                                              : QStringLiteral("not installed · %1").arg(hint.isEmpty() ? QStringLiteral("see app/python/README.md") : hint));
                    setLabelClass(status, available ? QStringLiteral("small") : QStringLiteral("error"));
                });
            }
        }

        void listCache() {
            cacheListed = true;
            callWorker(QStringLiteral("Starting the worker"), nlohmann::json::object(), "models_list", [this](const nlohmann::json& r) {
                cache->setRowCount(0);
                for (const nlohmann::json& m : r.value("models", nlohmann::json::array())) {
                    const int row = cache->rowCount();
                    cache->insertRow(row);
                    const QString path = fromStd(m.value("path", std::string()));
                    const QString spec = fromStd(m.value("spec", std::string()));
                    cache->setItem(row, 0, cell(spec.startsWith(QStringLiteral("hf:")) ? spec : QFileInfo(path).fileName(), path));
                    cache->setItem(row, 1, cell(path));
                    cache->setItem(row, 2, cell(bytesText(m.value("bytes", -1LL))));
                }
                cacheNote->setText(QStringLiteral("Cache: %1 (SIRIUS_MODEL_CACHE overrides)").arg(fromStd(r.value("cache", std::string()))));
            });
        }
    };

    ModelHubDialog::ModelHubDialog(WorkbenchBridge& bridge, QWidget* parent)
        : QDialog(parent), impl_(std::make_unique<Impl>(this, bridge)) {
        setWindowTitle(QStringLiteral("Model hub"));
        setMinimumWidth(720);
        resize(720, 640);

        impl_->client = new HubClient();
        impl_->client->moveToThread(&impl_->thread);
        impl_->thread.setObjectName(QStringLiteral("sirius-model-hub"));
        impl_->thread.start();

        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(22, 18, 22, 18);
        root->setSpacing(12);
        root->addWidget(widgets::heading(QStringLiteral("Model hub"), theme::kH4Px, this));
        auto* intro = widgets::label(
            QStringLiteral("Segmentation models for the Torch segmentation step: a TorchScript / ONNX file from Hugging Face, a model "
                           "family the worker's Python packages provide, or a file on this machine."),
            11, theme::kNeutral600, -1, this);
        intro->setWordWrap(true);
        root->addWidget(intro);

        auto* tabs = new QTabWidget(this);
        tabs->setDocumentMode(true);

        // --- Hugging Face
        auto* hf = new QWidget(tabs);
        auto* hl = new QVBoxLayout(hf);
        hl->setContentsMargins(0, 12, 0, 0);
        hl->setSpacing(8);
        auto* searchRow = new QHBoxLayout();
        searchRow->setSpacing(6);
        impl_->query = new QLineEdit(hf);
        impl_->query->setPlaceholderText(QStringLiteral("search models… (e.g. nuclei segmentation 3d, cellpose, unet)"));
        impl_->search = new QPushButton(QStringLiteral("Search"), hf);
        widgets::setButtonClass(impl_->search, "secondary small");
        searchRow->addWidget(impl_->query, 1);
        searchRow->addWidget(impl_->search);
        hl->addLayout(searchRow);
        impl_->results = makeTable({QStringLiteral("Model"), QStringLiteral("Downloads"), QStringLiteral("Likes"), QStringLiteral("Tags")}, 0, hf);
        impl_->results->setMinimumHeight(160);
        hl->addWidget(impl_->results, 2);
        hl->addWidget(new CaptionLabel(QStringLiteral("Files"), hf));
        impl_->files = makeTable({QStringLiteral("File"), QStringLiteral("Size")}, 0, hf);
        impl_->files->setMinimumHeight(100);
        hl->addWidget(impl_->files, 1);
        auto* dlRow = new QHBoxLayout();
        dlRow->setSpacing(8);
        impl_->progress = new QProgressBar(hf);
        impl_->progress->setRange(0, 1000);
        impl_->progress->setValue(0);
        impl_->progress->setTextVisible(false);
        impl_->progress->setFixedHeight(8);
        impl_->download = new QPushButton(QStringLiteral("Download"), hf);
        widgets::setButtonClass(impl_->download, "secondary small");
        impl_->download->setEnabled(false);
        impl_->useFile = new QPushButton(QStringLiteral("Use"), hf);
        widgets::setButtonClass(impl_->useFile, "primary small");
        impl_->useFile->setEnabled(false);
        dlRow->addWidget(impl_->progress, 1);
        dlRow->addWidget(impl_->download);
        dlRow->addWidget(impl_->useFile);
        hl->addLayout(dlRow);
        impl_->fileNote = widgets::label(QStringLiteral("Downloads land in $SIRIUS_MODEL_CACHE or ~/.sirius/models."), 11,
                                         theme::kNeutral600, -1, hf);
        impl_->fileNote->setWordWrap(true);
        hl->addWidget(impl_->fileNote);
        tabs->addTab(hf, QStringLiteral("Hugging Face"));

        // --- Model families
        auto* fam = new QWidget(tabs);
        auto* fl = new QVBoxLayout(fam);
        fl->setContentsMargins(0, 12, 0, 0);
        fl->setSpacing(10);
        auto* famNote = widgets::label(
            QStringLiteral("Packages in the worker's Python that segment cells and nuclei out of the box and return instance "
                           "labels directly (no threshold or watershed step). Any Cellpose model name or micro-SAM model type can "
                           "also be typed into the step's Model field as cellpose:<model> / microsam:<type>."),
            11, theme::kNeutral600, -1, fam);
        famNote->setWordWrap(true);
        fl->addWidget(famNote);
        auto* grid = new QGridLayout();
        grid->setHorizontalSpacing(10);
        grid->setVerticalSpacing(10);
        struct CardSpec {
            const char* spec;
            const char* title;
            const char* blurb;
        };
        const CardSpec specs[] = {
            {"cellpose:cyto3", "Cellpose cyto3", "Generalist cytoplasm model (Cellpose 3): whole cells in 2D and 3D, any modality."},
            {"cellpose:nuclei", "Cellpose nuclei", "Nuclear model for DAPI / Hoechst-like stains; 3D through per-plane stitching or do_3D."},
            {"microsam:vit_b_lm", "micro-SAM vit_b_lm", "Segment Anything fine-tuned for light microscopy (ViT-B): automatic instance segmentation."},
            {"microsam:vit_l_lm", "micro-SAM vit_l_lm", "The larger ViT-L light-microscopy model: better masks, more GPU memory and time."},
        };
        int i = 0;
        for (const CardSpec& c : specs) {
            auto* card = new QFrame(fam);
            card->setProperty("class", QStringLiteral("floating"));
            auto* cl = new QVBoxLayout(card);
            cl->setContentsMargins(12, 10, 12, 10);
            cl->setSpacing(4);
            cl->addWidget(widgets::label(QString::fromUtf8(c.title), 13, theme::kText, QFont::Bold, card));
            auto* blurb = widgets::label(QString::fromUtf8(c.blurb), 11, theme::kNeutral700, -1, card);
            blurb->setWordWrap(true);
            cl->addWidget(blurb);
            auto* bottom = new QHBoxLayout();
            FamilyCard fc;
            fc.spec = QString::fromUtf8(c.spec);
            fc.status = widgets::label(QStringLiteral("checking…"), 11, theme::kNeutral600, -1, card);
            fc.status->setWordWrap(true);
            fc.use = new QPushButton(QStringLiteral("Use"), card);
            widgets::setButtonClass(fc.use, "secondary small");
            bottom->addWidget(fc.status, 1);
            bottom->addWidget(fc.use);
            cl->addLayout(bottom);
            grid->addWidget(card, i / 2, i % 2);
            connect(fc.use, &QPushButton::clicked, this, [this, spec = fc.spec] { impl_->choose(spec); });
            impl_->cards.push_back(fc);
            ++i;
        }
        fl->addLayout(grid);
        fl->addStretch(1);
        tabs->addTab(fam, QStringLiteral("Model families"));

        // --- Local
        auto* local = new QWidget(tabs);
        auto* ll = new QVBoxLayout(local);
        ll->setContentsMargins(0, 12, 0, 0);
        ll->setSpacing(8);
        ll->addWidget(new CaptionLabel(QStringLiteral("Model cache"), local));
        impl_->cache = makeTable({QStringLiteral("Model"), QStringLiteral("Path"), QStringLiteral("Size")}, 1, local);
        ll->addWidget(impl_->cache, 1);
        impl_->cacheNote = widgets::label(QString(), 11, theme::kNeutral600, -1, local);
        impl_->cacheNote->setWordWrap(true);
        ll->addWidget(impl_->cacheNote);
        auto* localRow = new QHBoxLayout();
        auto* browse = new QPushButton(QStringLiteral("Browse…"), local);
        widgets::setButtonClass(browse, "secondary small");
        auto* useCached = new QPushButton(QStringLiteral("Use"), local);
        widgets::setButtonClass(useCached, "primary small");
        useCached->setEnabled(false);
        localRow->addWidget(browse);
        localRow->addStretch(1);
        localRow->addWidget(useCached);
        ll->addLayout(localRow);
        tabs->addTab(local, QStringLiteral("Local"));
        root->addWidget(tabs, 1);

        // --- status and buttons
        impl_->status = widgets::label(QString(), 11, theme::kNeutral600, -1, this);
        impl_->status->setWordWrap(true);
        root->addWidget(impl_->status);
        root->addWidget(new Rule(2, Qt::Horizontal, this));
        auto* buttons = new QHBoxLayout();
        impl_->selected = widgets::label(QStringLiteral("No model chosen"), 12, theme::kText, -1, this);
        impl_->selected->setWordWrap(true);
        buttons->addWidget(impl_->selected, 1);
        auto* cancel = new QPushButton(QStringLiteral("Cancel"), this);
        widgets::setButtonClass(cancel, "ghost");
        impl_->ok = new QPushButton(QStringLiteral("OK"), this);
        widgets::setButtonClass(impl_->ok, "primary");
        impl_->ok->setDefault(true);
        impl_->ok->setEnabled(false);
        buttons->addWidget(cancel);
        buttons->addWidget(impl_->ok);
        root->addLayout(buttons);
        connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
        connect(impl_->ok, &QPushButton::clicked, this, &QDialog::accept);

        // --- wiring
        connect(impl_->search, &QPushButton::clicked, this, [this] { impl_->runSearch(); });
        connect(impl_->query, &QLineEdit::returnPressed, this, [this] { impl_->runSearch(); });
        connect(impl_->results, &QTableWidget::itemSelectionChanged, this, [this] {
            const auto items = impl_->results->selectedItems();
            if (!items.isEmpty()) impl_->listFiles(impl_->results->item(items.first()->row(), 0)->data(Qt::UserRole).toString());
        });
        connect(impl_->files, &QTableWidget::itemSelectionChanged, this, [this] { impl_->fileSelected(); });
        connect(impl_->files, &QTableWidget::itemDoubleClicked, this, [this](QTableWidgetItem*) {
            if (!impl_->downloadedPath.isEmpty()) {
                impl_->choose(impl_->downloadedPath);
                accept();
            } else if (impl_->download->isEnabled()) {
                impl_->startDownload();
            }
        });
        connect(impl_->download, &QPushButton::clicked, this, [this] { impl_->startDownload(); });
        connect(impl_->useFile, &QPushButton::clicked, this, [this] {
            if (!impl_->downloadedPath.isEmpty()) impl_->choose(impl_->downloadedPath);
        });
        connect(browse, &QPushButton::clicked, this, [this] {
            const QString f = QFileDialog::getOpenFileName(this, QStringLiteral("Choose model"), QString(),
                                                           QStringLiteral("Models (*.pt *.pts *.pth *.onnx);;All files (*)"));
            if (!f.isEmpty()) impl_->choose(f);
        });
        connect(impl_->cache, &QTableWidget::itemSelectionChanged, this,
                [this, useCached] { useCached->setEnabled(!impl_->cache->selectedItems().isEmpty()); });
        connect(useCached, &QPushButton::clicked, this, [this] {
            const auto items = impl_->cache->selectedItems();
            if (!items.isEmpty()) impl_->choose(impl_->cache->item(items.first()->row(), 0)->data(Qt::UserRole).toString());
        });
        connect(impl_->cache, &QTableWidget::itemDoubleClicked, this, [this](QTableWidgetItem* it) {
            impl_->choose(impl_->cache->item(it->row(), 0)->data(Qt::UserRole).toString());
            accept();
        });
        connect(tabs, &QTabWidget::currentChanged, this, [this](int index) {
            if (index == 1 && !impl_->familiesChecked) impl_->checkFamilies();
            if (index == 2 && !impl_->cacheListed) impl_->listCache();
        });
        // the worker takes a moment to come up: start it now by listing the cache
        impl_->listCache();
    }

    ModelHubDialog::~ModelHubDialog() {
        impl_->client->cancel = true;
        // the client's queued jobs finish (or abort on the cancel flag) before
        // it is deleted on its own thread, where the launcher stops the worker
        QMetaObject::invokeMethod(impl_->client, [c = impl_->client] { delete c; }, Qt::BlockingQueuedConnection);
        impl_->thread.quit();
        impl_->thread.wait();
    }

    QString ModelHubDialog::chosenModel() const { return impl_->chosen; }

} // namespace sirius::app
