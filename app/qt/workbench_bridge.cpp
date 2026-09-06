#include "qt/workbench_bridge.hpp"

#include <exception>

#include <QMetaObject>

namespace sirius::app {

    struct WorkbenchBridge::Relay final : Workbench::Observer {
        WorkbenchBridge& b;
        explicit Relay(WorkbenchBridge& bridge) : b(bridge) {}
        void datasetChanged() override { emit b.datasetChanged(); }
        void pipelineChanged() override { emit b.pipelineChanged(); }
        void stepChanged(int index) override { emit b.stepChanged(index); }
        void selectionChanged() override { emit b.selectionChanged(); }
        void viewedStepChanged() override { emit b.viewedStepChanged(); }
        void viewStateChanged() override { emit b.viewStateChanged(); }
        void outputsChanged() override { emit b.outputsChanged(); }
        void labelsChanged(StepId id) override { emit b.labelsChanged(static_cast<quint64>(id)); }
        void runStateChanged() override { b.runStateChanged(); }
        void historyChanged() override { emit b.historyChanged(); }
        void backendChanged() override { emit b.backendChanged(); }
        void logged(const std::string& line) override { emit b.logged(QString::fromStdString(line)); }
    };

    WorkbenchBridge::WorkbenchBridge(Workbench& wb, QObject* parent)
        : QObject(parent), wb_(wb), relay_(std::make_unique<Relay>(*this)) {
        wb_.addObserver(relay_.get());
        dispatcher_ = new QObject();
        dispatcher_->moveToThread(&worker_);
        worker_.setObjectName(QStringLiteral("sirius-worker"));
        worker_.start();
        progressTimer_.setInterval(60);
        connect(&progressTimer_, &QTimer::timeout, this, &WorkbenchBridge::pollProgress);
    }

    WorkbenchBridge::~WorkbenchBridge() {
        wb_.removeObserver(relay_.get());
        if (job_) job_->cancel();
        taskCancel_.store(true);
        // The dispatcher's queued jobs finish on the worker thread before it
        // quits, so nothing touches the workbench after this returns.
        QMetaObject::invokeMethod(dispatcher_, [d = dispatcher_] { delete d; }, Qt::BlockingQueuedConnection);
        worker_.quit();
        worker_.wait();
    }

    // --- runs ------------------------------------------------------------------

    bool WorkbenchBridge::startRun(int target) {
        if (wb_.running()) {
            wb_.logLine("A run is already in progress.");
            return false;
        }
        std::shared_ptr<RunJob> job = wb_.createRun(target);
        if (!job) return false;
        job_ = job;
        progressTimer_.start();
        emit runProgress(0.0, job->target(), QStringLiteral("Starting…"));
        QMetaObject::invokeMethod(
            dispatcher_,
            [this, job] {
                job->execute();
                QMetaObject::invokeMethod(this, [this] { onJobFinished(); }, Qt::QueuedConnection);
            },
            Qt::QueuedConnection);
        return true;
    }

    void WorkbenchBridge::cancelRun() {
        if (job_) job_->cancel();
        wb_.cancelRun();
    }

    void WorkbenchBridge::runStateChanged() {
        if (wb_.running()) emit runStarted();
    }

    void WorkbenchBridge::pollProgress() {
        if (job_) {
            RunProgress& p = job_->progress();
            emit runProgress(p.fraction.load(), p.stepIndex.load(), QString::fromStdString(p.messageCopy()));
        }
        if (taskActive_.load()) {
            std::string msg;
            {
                std::lock_guard<std::mutex> g(taskMutex_);
                msg = taskMessage_;
            }
            emit taskProgress(taskFraction_.load(), QString::fromStdString(msg));
        }
        if (!job_ && !taskActive_.load()) progressTimer_.stop();
    }

    void WorkbenchBridge::onJobFinished() {
        std::shared_ptr<RunJob> job = std::move(job_);
        job_.reset();
        if (!taskActive_.load()) progressTimer_.stop();
        if (!job) return;
        const bool ok = job->succeeded();
        const QString error = QString::fromStdString(job->error());
        wb_.finishRun(job);
        emit runFinished(ok, error);
    }

    // --- tasks -----------------------------------------------------------------

    bool WorkbenchBridge::startTask(const QString& label, Task task) {
        if (taskActive_.load()) {
            wb_.logLine("Another task is still running: " + taskLabel_.toStdString());
            return false;
        }
        taskActive_.store(true);
        taskCancel_.store(false);
        taskFraction_.store(0.0);
        {
            std::lock_guard<std::mutex> g(taskMutex_);
            taskMessage_.clear();
            taskError_.clear();
        }
        taskLabel_ = label;
        progressTimer_.start();
        emit taskStarted(label);
        QMetaObject::invokeMethod(
            dispatcher_,
            [this, task = std::move(task)] {
                try {
                    task(
                        [this](double f, const std::string& m) {
                            taskFraction_.store(f);
                            std::lock_guard<std::mutex> g(taskMutex_);
                            taskMessage_ = m;
                        },
                        [this] { return taskCancel_.load(); });
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> g(taskMutex_);
                    taskError_ = e.what();
                } catch (...) {
                    std::lock_guard<std::mutex> g(taskMutex_);
                    taskError_ = "unknown error";
                }
                QMetaObject::invokeMethod(this, [this] { onTaskFinished(); }, Qt::QueuedConnection);
            },
            Qt::QueuedConnection);
        return true;
    }

    void WorkbenchBridge::cancelTask() { taskCancel_.store(true); }

    void WorkbenchBridge::onTaskFinished() {
        std::string error;
        {
            std::lock_guard<std::mutex> g(taskMutex_);
            error = taskError_;
        }
        taskActive_.store(false);
        if (!job_) progressTimer_.stop();
        const QString label = taskLabel_;
        taskLabel_.clear();
        if (error.empty()) wb_.logLine(label.toStdString() + ": done");
        else wb_.logLine(label.toStdString() + ": " + error);
        emit taskFinished(error.empty(), QString::fromStdString(error));
    }

} // namespace sirius::app
