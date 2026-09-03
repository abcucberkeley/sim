#include "qt/recon_worker.hpp"

#include <exception>

namespace sirius::app {

    ReconWorker::ReconWorker(ReconSession& session, QObject* parent)
        : QObject(parent), session_(session) {}

    void ReconWorker::run(Device device, PlanRigor rigor) {
        emit started();
        try {
            result_ = std::make_unique<ReconResult>(session_.reconstruct(device, rigor));
        } catch (const std::exception& e) {
            result_.reset();
            emit failed(QString::fromUtf8(e.what()));
            return;
        }
        emit finished();
    }

    std::unique_ptr<ReconResult> ReconWorker::takeResult() { return std::move(result_); }

} // namespace sirius::app
