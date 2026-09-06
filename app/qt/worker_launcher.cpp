#include "qt/worker_launcher.hpp"

#include <stdexcept>

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRandomGenerator>
#include <QSettings>
#include <QStandardPaths>

#include "qt/qt_strings.hpp"

namespace sirius::app {

    WorkerLauncher::WorkerLauncher(QObject* parent) : QObject(parent) {
        // A per-session token so only this app talks to its worker.
        token_ = QString::number(QRandomGenerator::global()->generate64(), 16) +
                 QString::number(QRandomGenerator::global()->generate64(), 16);
    }

    WorkerLauncher::~WorkerLauncher() { stop(); }

    void WorkerLauncher::setPython(const QString& python) { python_ = python; }
    void WorkerLauncher::setScriptDir(const QString& dir) { scriptDir_ = dir; }
    void WorkerLauncher::setDevice(const QString& device) { device_ = device; }

    QString WorkerLauncher::python() const {
        if (!python_.isEmpty()) return python_;
        const QString fromSettings = QSettings().value(QStringLiteral("worker/python")).toString();
        if (!fromSettings.isEmpty()) return fromSettings;
        const QByteArray env = qgetenv("SIRIUS_PYTHON");
        if (!env.isEmpty()) return QString::fromLocal8Bit(env);
        return QStringLiteral("python3");
    }

    QString WorkerLauncher::scriptDir() const {
        if (!scriptDir_.isEmpty()) return scriptDir_;
        const QString fromSettings = QSettings().value(QStringLiteral("worker/dir")).toString();
        if (!fromSettings.isEmpty()) return fromSettings;
        // next to the executable (the build copies app/python there), then the source tree
        const QString beside = QCoreApplication::applicationDirPath() + QStringLiteral("/python");
        if (QFileInfo::exists(beside + QStringLiteral("/sirius_worker/__main__.py"))) return beside;
        return fromStd(workerScriptPath());
    }

    bool WorkerLauncher::isRunning() const { return process_ && process_->state() == QProcess::Running && port_ > 0; }

    void WorkerLauncher::start() {
        const QString dir = scriptDir();
        if (dir.isEmpty() || !QFileInfo::exists(dir + QStringLiteral("/sirius_worker/__main__.py")))
            throw std::runtime_error("the Python worker (sirius_worker) was not found next to the application; "
                                     "set Preferences ▸ Worker ▸ directory");
        process_ = std::make_unique<QProcess>();
        process_->setWorkingDirectory(dir);
        process_->setProcessChannelMode(QProcess::SeparateChannels);
        QStringList args{QStringLiteral("-m"), QStringLiteral("sirius_worker"), QStringLiteral("--host"),
                         QStringLiteral("127.0.0.1"), QStringLiteral("--port"), QStringLiteral("0"),
                         QStringLiteral("--token"), token_, QStringLiteral("--device"), device_};
        log_.clear();
        QObject::connect(process_.get(), &QProcess::readyReadStandardError, this, [this] {
            const QString text = QString::fromUtf8(process_->readAllStandardError());
            log_ += text;
            if (log_.size() > 20000) log_ = log_.right(10000);
            for (const QString& line : text.split('\n', Qt::SkipEmptyParts)) emit logged(line);
        });
        QObject::connect(process_.get(), qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
                         [this](int, QProcess::ExitStatus) {
                             port_ = 0;
                             emit stopped();
                         });
        process_->start(python(), args);
        if (!process_->waitForStarted(5000))
            throw std::runtime_error("cannot start " + toStd(python()) + ": " + toStd(process_->errorString()));
        // The worker prints one JSON line with its port once it listens.
        QByteArray line;
        while (line.isEmpty() && process_->state() == QProcess::Running) {
            if (!process_->waitForReadyRead(60000)) break;
            while (process_->canReadLine()) {
                line = process_->readLine().trimmed();
                if (!line.isEmpty()) break;
            }
        }
        const QJsonObject obj = QJsonDocument::fromJson(line).object();
        port_ = obj.value(QStringLiteral("port")).toInt(0);
        if (port_ <= 0) {
            const QString err = QString::fromUtf8(process_->readAllStandardError());
            log_ += err;
            stop();
            throw std::runtime_error("the Python worker did not start: " + toStd((log_ + err).trimmed().right(800)));
        }
        emit started(port_);
    }

    std::unique_ptr<RemoteWorker> WorkerLauncher::connect() {
        if (!isRunning()) start();
        try {
            return RemoteWorker::connect("127.0.0.1", port_, toStd(token_));
        } catch (const std::exception&) {
            // the process may have died between runs: start once more
            stop();
            start();
            return RemoteWorker::connect("127.0.0.1", port_, toStd(token_));
        }
    }

    void WorkerLauncher::stop() {
        if (!process_) return;
        if (process_->state() == QProcess::Running) {
            process_->terminate();
            if (!process_->waitForFinished(3000)) {
                process_->kill();
                process_->waitForFinished(1000);
            }
        }
        process_.reset();
        port_ = 0;
    }

} // namespace sirius::app
