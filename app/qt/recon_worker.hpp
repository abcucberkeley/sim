#ifndef SIRIUS_APP_RECON_WORKER_HPP
#define SIRIUS_APP_RECON_WORKER_HPP

// Runs ReconSession::reconstruct off the GUI thread. The object is moved to a
// QThread by MainWindow; run() is invoked through a queued call and reports
// back with signals only (no custom types cross threads, so no metatype
// registration is needed on either Qt 5 or Qt 6). The result is handed over
// through takeResult() once finished() has been delivered.

#include <memory>

#include <QObject>
#include <QString>

#include <sirius/device.hpp>
#include <sirius/fft_common.hpp>

#include "core/session.hpp"

namespace sirius::app {

    class ReconWorker : public QObject {
        Q_OBJECT
    public:
        explicit ReconWorker(ReconSession& session, QObject* parent = nullptr);

        // Called on the worker thread. Exceptions become failed(message).
        void run(Device device, PlanRigor rigor);

        // Ownership of the last successful result; null if none is pending.
        std::unique_ptr<ReconResult> takeResult();

    signals:
        void started();
        void finished();
        void failed(const QString& message);

    private:
        ReconSession& session_;
        std::unique_ptr<ReconResult> result_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_RECON_WORKER_HPP
