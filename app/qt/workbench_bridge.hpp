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

    class WorkbenchBridge : public QObject, private Workbench::Observer {
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
        // Observer
        void datasetChanged_() {}
        void datasetChanged() override { emit datasetChanged(); }
        void pipelineChanged() override { emit pipelineChanged(); }
        void stepChanged(int index) override { emit stepChanged(index); }
        void selectionChanged() override { emit selectionChanged(); }
        void viewedStepChanged() override { emit viewedStepChanged(); }
        void viewStateChanged() override { emit viewStateChanged(); }
        void outputsChanged() override { emit outputsChanged(); }
        void labelsChanged(StepId id) override { emit labelsChanged(static_cast<quint64>(id)); }
        void runStateChanged() override;
        void historyChanged() override { emit historyChanged(); }
        void backendChanged() override { emit backendChanged(); }
        void logged(const std::string& line) override { emit logged(QString::fromStdString(line)); }

        void pollProgress();

        Workbench& wb_;
        QThread worker_;
        QTimer progressTimer_;
        std::shared_ptr<RunJob> job_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_WORKBENCH_BRIDGE_HPP
