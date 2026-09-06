#ifndef SIRIUS_APP_WORKER_LAUNCHER_HPP
#define SIRIUS_APP_WORKER_LAUNCHER_HPP

// Starts the bundled Python worker (app/python/sirius_worker) as a child
// process the first time a step needs it and hands out connections to it.
// The process is kept for the rest of the session: loading torch and a
// model takes seconds, connecting takes milliseconds. Installed into the
// workbench through Workbench::setLocalWorkerLauncher.

#include <memory>

#include <QObject>
#include <QProcess>
#include <QString>

#include "core/rpc.hpp"

namespace sirius::app {

    class WorkerLauncher : public QObject {
        Q_OBJECT
    public:
        explicit WorkerLauncher(QObject* parent = nullptr);
        ~WorkerLauncher() override;

        // Interpreter and worker directory; empty = QSettings "worker/python"
        // (then $SIRIUS_PYTHON, then "python3") and the directory next to
        // the executable / the source tree.
        void setPython(const QString& python);
        void setScriptDir(const QString& dir);
        void setDevice(const QString& device);   // "auto", "cuda", "cpu"
        QString python() const;
        QString scriptDir() const;

        // Starts the process when needed and connects; throws std::runtime_error
        // with the worker's stderr when it fails to come up.
        std::unique_ptr<RemoteWorker> connect();
        bool isRunning() const;
        int port() const noexcept { return port_; }
        void stop();
        QString lastLog() const { return log_; }

    signals:
        void started(int port);
        void stopped();
        void logged(const QString& line);

    private:
        void start();

        QString python_;
        QString scriptDir_;
        QString device_ = QStringLiteral("auto");
        QString token_;
        QString log_;
        std::unique_ptr<QProcess> process_;
        int port_ = 0;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_WORKER_LAUNCHER_HPP
