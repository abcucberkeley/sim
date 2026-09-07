#include "qt/dialogs/model_hub_dialog.hpp"

#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <vector>

#include <QBoxLayout>
#include <QComboBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <QHeaderView>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMetaObject>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QStyle>
#include <QTabWidget>
#include <QScrollBar>
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

        // One card per model family: the package (installed or not), its
        // models, and Use / Install & use.
        struct FamilyCard {
            QString family;        // "cellpose" | "microsam"
            QString title;         // "Cellpose"
            QString package;       // what the installer adds
            QString probe;         // spec asked for model_info
            QComboBox* model = nullptr;
            QLabel* status = nullptr;
            QPushButton* use = nullptr;
            bool available = false;
            int weightsCached = -1;   // -1 unknown, 0 no, 1 yes
            nlohmann::json info;
            QString spec() const { return family + QLatin1Char(':') + model->currentData().toString(); }
        };

        QString hubToken() { return QSettings().value(QStringLiteral("hub/token")).toString().trimmed(); }

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
        QPushButton* tokenButton = nullptr;
        QString repo;
        bool repoGated = false;
        QString downloadedPath;   // of the selected file, once it is in the cache
        QString progressPrefix;   // "Downloading…", "Installing…"

        // families and the local cache
        std::vector<FamilyCard> cards;
        QPlainTextEdit* log = nullptr;   // install / fetch output lines
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
        void callWorker(const QString& what, nlohmann::json params, const std::string& method,
                        std::function<void(const nlohmann::json&)> done, bool withProgress = false) {
            setStatus(what + QStringLiteral("…"), false);
            progressPrefix = what;
            // gated / private repositories: the access token from the settings
            if (method.rfind("hub_", 0) == 0 || method == "model_prepare") {
                const QString token = hubToken();
                if (!token.isEmpty()) params["token"] = toStd(token);
            }
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
            if (message.isEmpty()) return;
            setStatus(progressPrefix + QStringLiteral("… ") + message, false);
            if (log && log->isVisible()) {
                log->appendPlainText(message);
                log->verticalScrollBar()->setValue(log->verticalScrollBar()->maximum());
            }
        }

        void showLog(const QString& first) {
            log->clear();
            log->show();
            if (!first.isEmpty()) log->appendPlainText(first);
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
                               const bool gated = m.contains("gated") && !(m["gated"].is_boolean() && !m["gated"].get<bool>());
                               auto* idItem = cell(gated ? id + QStringLiteral("  (gated)") : id, id);
                               idItem->setData(Qt::UserRole + 1, gated);
                               idItem->setToolTip(gated ? id + QStringLiteral("\nGated: accept the terms on huggingface.co while signed in, "
                                                                            "then add your access token (Token…)")
                                                        : id);
                               if (gated) idItem->setForeground(theme::kAccent);
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

        void listFiles(const QString& id, bool gated) {
            if (id == repo) return;
            repo = id;
            repoGated = gated;
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
                QString note = firstModel >= 0
                                   ? QStringLiteral("Model files (.pt, .pts, .pth, .onnx) are highlighted; the rest is shown for reference.")
                                   : QStringLiteral("This repository has no TorchScript / ONNX file; SIRIUS cannot run its weights directly.");
                if (repoGated)
                    note += QStringLiteral(" Gated repository: accept its terms at https://huggingface.co/%1 while signed in, then add "
                                           "your access token with Token….").arg(id);
                fileNote->setText(note);
            });
        }

        void askToken() {
            bool ok = false;
            const QString token = QInputDialog::getText(
                dialog, QStringLiteral("Hugging Face access token"),
                QStringLiteral("Token for gated or private repositories (huggingface.co ▸ Settings ▸ Access Tokens).\n"
                               "Stored in the application settings and passed to the local worker as HF_TOKEN."),
                QLineEdit::Password, hubToken(), &ok);
            if (!ok) return;
            QSettings().setValue(QStringLiteral("hub/token"), token.trimmed());
            tokenButton->setText(token.trimmed().isEmpty() ? QStringLiteral("Token…") : QStringLiteral("Token ✓"));
            setStatus(token.trimmed().isEmpty() ? QStringLiteral("Token cleared.") : QStringLiteral("Token stored."), false);
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
            for (std::size_t i = 0; i < cards.size(); ++i) checkFamily(i);
        }

        // model_info for the card's probe spec: installed?, version, the
        // package's model names, whether the weights are on disk.
        void checkFamily(std::size_t index) {
            FamilyCard& card = cards[index];
            callWorker(QStringLiteral("Checking ") + card.title, {{"spec", toStd(card.probe)}}, "model_info",
                       [this, index](const nlohmann::json& info) { applyFamilyInfo(index, info); });
        }

        void applyFamilyInfo(std::size_t index, const nlohmann::json& info) {
            FamilyCard& card = cards[index];
            card.info = info;
            card.available = info.value("available", false);
            card.weightsCached = info.contains("weights_cached") && info["weights_cached"].is_boolean()
                                     ? (info["weights_cached"].get<bool>() ? 1 : 0)
                                     : -1;
            // models: the package's own list once installed, the known names before
            const QString current = card.model->currentData().toString();
            QSignalBlocker block(card.model);
            card.model->clear();
            const std::string defaultModel = info.value("default_model", std::string());
            if (card.family == QLatin1String("cellpose"))
                card.model->addItem(defaultModel.empty() ? QStringLiteral("default (built-in model)")
                                                         : QStringLiteral("default · %1").arg(fromStd(defaultModel)),
                                    QStringLiteral("default"));
            for (const nlohmann::json& m : info.value("known_models", nlohmann::json::array()))
                if (m.is_string()) card.model->addItem(fromStd(m.get<std::string>()), fromStd(m.get<std::string>()));
            const int keep = card.model->findData(current);
            card.model->setCurrentIndex(keep >= 0 ? keep : 0);
            refreshFamilyStatus(index);
        }

        void refreshFamilyStatus(std::size_t index) {
            FamilyCard& card = cards[index];
            QString text;
            if (card.available) {
                const QString version = fromStd(card.info.value("version", std::string()));
                text = QStringLiteral("%1%2 in the worker's Python").arg(card.package, version.isEmpty() ? QString() : QLatin1Char(' ') + version);
                if (card.weightsCached == 1) text += QStringLiteral(" · weights cached");
                else if (card.weightsCached == 0) text += QStringLiteral(" · weights download on first use");
            } else {
                const nlohmann::json install = card.info.value("install", nlohmann::json::object());
                text = QStringLiteral("not installed · Install & use runs %1")
                           .arg(fromStd(install.value("display", install.value("command", std::string("pip install " + toStd(card.package))))));
            }
            card.status->setText(text);
            setLabelClass(card.status, card.available ? QStringLiteral("small") : QStringLiteral("error"));
            card.use->setText(card.available ? QStringLiteral("Use") : QStringLiteral("Install & use"));
        }

        // Use: install the package first when it is missing (asked), then
        // offer to fetch the weights now (asked), then choose the spec.
        void useFamily(std::size_t index) {
            FamilyCard& card = cards[index];
            const QString spec = card.spec();
            if (!card.available) {
                const nlohmann::json install = card.info.value("install", nlohmann::json::object());
                const QString command = fromStd(install.value("command", std::string("pip install " + toStd(card.package))));
                const QString note = fromStd(install.value("note", std::string()));
                const QString python = fromStd(card.info.value("python", std::string()));
                QMessageBox box(dialog);
                box.setIcon(QMessageBox::Question);
                box.setWindowTitle(QStringLiteral("Install %1").arg(card.title));
                box.setText(QStringLiteral("Install the %1 package into the worker's Python?").arg(card.package));
                box.setInformativeText(QStringLiteral("%1\n\n%2%3This changes the Python environment%4 and can take several "
                                                      "minutes; the output is shown in the log. The %5 weights are offered next.")
                                           .arg(command, note, note.isEmpty() ? QString() : QStringLiteral(". "),
                                                python.isEmpty() ? QString() : QStringLiteral(" (%1)").arg(python), card.title));
                QPushButton* go = box.addButton(QStringLiteral("Install"), QMessageBox::AcceptRole);
                box.addButton(QMessageBox::Cancel);
                box.setDefaultButton(go);
                box.exec();
                if (box.clickedButton() != go) return;
                runInstall(index);
                return;
            }
            if (card.weightsCached == 0) {
                QMessageBox box(dialog);
                box.setIcon(QMessageBox::Question);
                box.setWindowTitle(QStringLiteral("Model weights"));
                box.setText(QStringLiteral("Download the weights of %1 now?").arg(spec));
                box.setInformativeText(QStringLiteral("They come from the model's authors into the worker's cache%1. "
                                                      "Otherwise the first run of the step downloads them.")
                                           .arg(card.family == QLatin1String("cellpose")
                                                    ? QStringLiteral(" (Cellpose 4's built-in model is about 1.2 GB)")
                                                    : QString()));
                QPushButton* go = box.addButton(QStringLiteral("Download"), QMessageBox::AcceptRole);
                QPushButton* later = box.addButton(QStringLiteral("Not now"), QMessageBox::RejectRole);
                box.addButton(QMessageBox::Cancel);
                box.setDefaultButton(go);
                box.exec();
                if (box.clickedButton() == go) {
                    runPrepare(index, spec);
                    return;
                }
                if (box.clickedButton() != later) return;
            }
            choose(spec);
        }

        void runInstall(std::size_t index) {
            FamilyCard& card = cards[index];
            for (FamilyCard& c : cards) c.use->setEnabled(false);
            client->cancel = false;
            progress->setValue(0);
            showLog(QString());
            callWorker(QStringLiteral("Installing ") + card.package, {{"family", toStd(card.family)}}, "install",
                       [this, index](const nlohmann::json& r) {
                           for (FamilyCard& c : cards) c.use->setEnabled(true);
                           FamilyCard& c = cards[index];
                           const bool ok = r.value("ok", false) && r.value("available", false);
                           if (!ok) {
                               QStringList tail;
                               for (const nlohmann::json& line : r.value("tail", nlohmann::json::array()))
                                   if (line.is_string()) tail << fromStd(line.get<std::string>());
                               setStatus(QStringLiteral("Installing %1 failed (exit code %2): %3")
                                             .arg(c.package).arg(r.value("returncode", -1))
                                             .arg(tail.isEmpty() ? QStringLiteral("see the log") : tail.last()),
                                         true);
                               return;
                           }
                           setStatus(QStringLiteral("%1 installed.").arg(c.package), false);
                           // re-read the package's models and weights, then carry on with Use
                           callWorker(QStringLiteral("Checking ") + c.title, {{"spec", toStd(c.probe)}}, "model_info",
                                      [this, index](const nlohmann::json& info) {
                                          applyFamilyInfo(index, info);
                                          if (cards[index].available) useFamily(index);
                                      });
                       },
                       true);
        }

        void runPrepare(std::size_t index, const QString& spec) {
            for (FamilyCard& c : cards) c.use->setEnabled(false);
            client->cancel = false;
            progress->setValue(0);
            showLog(QString());
            callWorker(QStringLiteral("Fetching the weights of ") + spec, {{"spec", toStd(spec)}}, "model_prepare",
                       [this, index, spec](const nlohmann::json& r) {
                           for (FamilyCard& c : cards) c.use->setEnabled(true);
                           cards[index].weightsCached = 1;
                           refreshFamilyStatus(index);
                           setStatus(QStringLiteral("Weights ready: %1").arg(fromStd(r.value("path", std::string()))), false);
                           choose(spec);
                       },
                       true);
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
            QStringLiteral("Segmentation models for the Segmentation step: a TorchScript / ONNX file from Hugging Face, a model "
                           "family the worker's Python packages provide, or a file on this machine."),
            11, theme::kNeutral600, -1, this);
        intro->setWordWrap(true);
        root->addWidget(intro);

        auto* tabs = new QTabWidget(this);
        tabs->setDocumentMode(true);

        // --- Model families
        auto* fam = new QWidget(tabs);
        auto* fl = new QVBoxLayout(fam);
        fl->setContentsMargins(0, 12, 0, 0);
        fl->setSpacing(10);
        auto* famNote = widgets::label(
            QStringLiteral("Packages that segment cells and nuclei out of the box and return instance labels directly (no "
                           "threshold or watershed step). A missing package is installed into the worker's Python on request; "
                           "the weights come from the model's authors, no Hugging Face account needed. Any model name can "
                           "also be typed into the step's Model field as cellpose:<model> / microsam:<type>."),
            11, theme::kNeutral600, -1, fam);
        famNote->setWordWrap(true);
        fl->addWidget(famNote);
        auto* grid = new QGridLayout();
        grid->setHorizontalSpacing(10);
        grid->setVerticalSpacing(10);
        struct CardSpec {
            const char* family;
            const char* title;
            const char* package;
            const char* probe;
            const char* blurb;
            const char* models[6];
        };
        const CardSpec specs[] = {
            {"cellpose", "Cellpose", "cellpose", "cellpose:default",
             "Generalist cell and nucleus segmentation, 2D and 3D, any modality. Cellpose 4 ships one built-in model "
             "(cellpose:default); Cellpose 3 offers cyto3, nuclei and more. Recommended first choice for fluorescence.",
             {"default", nullptr}},
            {"microsam", "micro-SAM", "micro_sam", "microsam:vit_b_lm",
             "Segment Anything fine-tuned for microscopy with automatic instance segmentation: vit_b_lm / vit_l_lm for "
             "light microscopy, vit_b_em_organelles for electron microscopy. The generalist for unusual specimens.",
             {"vit_b_lm", "vit_l_lm", "vit_t_lm", "vit_b_em_organelles", "vit_l_em_organelles", nullptr}},
        };
        int i = 0;
        for (const CardSpec& c : specs) {
            auto* card = new QFrame(fam);
            card->setProperty("class", QStringLiteral("floating"));
            auto* cl = new QVBoxLayout(card);
            cl->setContentsMargins(12, 10, 12, 10);
            cl->setSpacing(6);
            cl->addWidget(widgets::label(QString::fromUtf8(c.title), 13, theme::kText, QFont::Bold, card));
            auto* blurb = widgets::label(QString::fromUtf8(c.blurb), 11, theme::kNeutral700, -1, card);
            blurb->setWordWrap(true);
            cl->addWidget(blurb);
            FamilyCard fc;
            fc.family = QString::fromUtf8(c.family);
            fc.title = QString::fromUtf8(c.title);
            fc.package = QString::fromUtf8(c.package);
            fc.probe = QString::fromUtf8(c.probe);
            auto* modelRow = new QHBoxLayout();
            modelRow->setSpacing(6);
            modelRow->addWidget(new CaptionLabel(QStringLiteral("Model"), card));
            fc.model = new QComboBox(card);
            fc.model->setSizeAdjustPolicy(QComboBox::AdjustToContents);
            for (const char* const* m = c.models; *m; ++m) {
                const QString name = QString::fromUtf8(*m);
                fc.model->addItem(name == QLatin1String("default") ? QStringLiteral("default (built-in model)") : name, name);
            }
            fc.model->setToolTip(QStringLiteral("The package's models; the list is read from the installed version"));
            modelRow->addWidget(fc.model, 1);
            cl->addLayout(modelRow);
            auto* bottom = new QHBoxLayout();
            fc.status = widgets::label(QStringLiteral("checking…"), 11, theme::kNeutral600, -1, card);
            fc.status->setWordWrap(true);
            fc.use = new QPushButton(QStringLiteral("Use"), card);
            widgets::setButtonClass(fc.use, "secondary small");
            bottom->addWidget(fc.status, 1);
            bottom->addWidget(fc.use);
            cl->addLayout(bottom);
            grid->addWidget(card, 0, i);
            const std::size_t index = impl_->cards.size();
            connect(fc.use, &QPushButton::clicked, this, [this, index] { impl_->useFamily(index); });
            impl_->cards.push_back(fc);
            ++i;
        }
        fl->addLayout(grid);
        impl_->log = new QPlainTextEdit(fam);
        impl_->log->setReadOnly(true);
        impl_->log->setMaximumHeight(140);
        impl_->log->setLineWrapMode(QPlainTextEdit::NoWrap);
        impl_->log->setStyleSheet(QStringLiteral("QPlainTextEdit { font-family: monospace; font-size: 12px; color: %1; "
                                                 "background: %2; border: 1px solid %3; }")
                                      .arg(theme::kNeutral700.name(), theme::kSurface.name(), theme::kNeutral300.name()));
        impl_->log->hide();
        fl->addWidget(impl_->log);
        fl->addStretch(1);
        tabs->addTab(fam, QStringLiteral("Model families"));

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
        impl_->tokenButton = new QPushButton(hubToken().isEmpty() ? QStringLiteral("Token…") : QStringLiteral("Token ✓"), hf);
        widgets::setButtonClass(impl_->tokenButton, "ghost small");
        impl_->tokenButton->setToolTip(QStringLiteral("Hugging Face access token for gated or private repositories"));
        searchRow->addWidget(impl_->query, 1);
        searchRow->addWidget(impl_->search);
        searchRow->addWidget(impl_->tokenButton);
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
            if (items.isEmpty()) return;
            QTableWidgetItem* id = impl_->results->item(items.first()->row(), 0);
            impl_->listFiles(id->data(Qt::UserRole).toString(), id->data(Qt::UserRole + 1).toBool());
        });
        connect(impl_->tokenButton, &QPushButton::clicked, this, [this] { impl_->askToken(); });
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
            if (index == 0 && !impl_->familiesChecked) impl_->checkFamilies();
            if (index == 2 && !impl_->cacheListed) impl_->listCache();
        });
        // the families tab is up first: checking it starts the worker
        impl_->checkFamilies();
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
