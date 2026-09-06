#ifndef SIRIUS_APP_WORKBENCH_BRIDGE_HPP
#define SIRIUS_APP_WORKBENCH_BRIDGE_HPP

// The one QObject between the Qt-free Workbench and the widgets: forwards
// the workbench's observer callbacks as signals, runs RunJobs on a worker
// thread and reports their progress on the GUI thread. Every panel takes a
// WorkbenchBridge& and talks to `wb()` directly for reads and edits.

#include <memory>

#include <QObject>
#include <QString>
#include <QThread>
#include <QTimer>

#include "core/workbench.hpp"

namespace sirius::app {

    class WorkbenchBridge : public QObject {
        Q_OBJECT
    public:
        explicit WorkbenchBridge(Workbench& wb, QObject* parent = nullptr);
        ~WorkbenchBridge() override;

        Workbench& wb() noexcept { return wb_; }
        const Workbench& wb() const noexcept { return wb_; }

        // Starts a run of step `target` (-1 = all) on the worker thread; false
        // (with a log line) when nothing can run or a run is active.
        bool startRun(int target = -1);
        void cancelRun();
        bool running() const noexcept { return wb_.running(); }

    signals:
        void datasetChanged();
        void pipelineChanged();
        void stepChanged(int index);
        void selectionChanged();
        void viewedStepChanged();
        void viewStateChanged();
        void outputsChanged();
        void labelsChanged(quint64 stepId);
        void runStarted();
        void runProgress(double fraction, int stepIndex, const QString& message);
        void runFinished(bool ok, const QString& error);
        void historyChanged();
        void backendChanged();
        void logged(const QString& line);

    private:
        // The Observer lives in a nested object: its callbacks share the
        // signals' names, and a class cannot both declare a signal and
        // override a virtual of the same signature.
        struct Relay final : Workbench::Observer {
            explicit Relay(WorkbenchBridge* b) : b(b) {}
            void datasetChanged() override { emit b->datasetChanged(); }
            void pipelineChanged() override { emit b->pipelineChanged(); }
            void stepChanged(int index) override { emit b->stepChanged(index); }
            void selectionChanged() override { emit b->selectionChanged(); }
            void viewedStepChanged() override { emit b->viewedStepChanged(); }
            void viewStateChanged() override { emit b->viewStateChanged(); }
            void outputsChanged() override { emit b->outputsChanged(); }
            void labelsChanged(StepId id) override { emit b->labelsChanged(static_cast<quint64>(id)); }
            void runStateChanged() override { b->onRunStateChanged(); }
            void historyChanged() override { emit b->historyChanged(); }
            void backendChanged() override { emit b->backendChanged(); }
            void logged(const std::string& line) override { emit b->logged(QString::fromStdString(line)); }
            WorkbenchBridge* b;
        };
        void onRunStateChanged();

        void pollProgress();

        Workbench& wb_;
        Relay relay_{this};
        QThread worker_;
        QTimer progressTimer_;
        std::shared_ptr<RunJob> job_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_WORKBENCH_BRIDGE_HPP
