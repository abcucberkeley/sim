#ifndef SIRIUS_APP_TRACE_HPP
#define SIRIUS_APP_TRACE_HPP

// SIRIUS_TRACE_VIEW=1 prints what the hot UI paths cost. ScopedTrace starts
// a timer when it is constructed and logs "<what> N us" when it leaves the
// scope; with the variable unset it does nothing at all, so a trace can sit
// in a paint or drag path.
//
//     ScopedTrace trace("layoutPanes");

#include <QElapsedTimer>
#include <QtGlobal>

namespace sirius::app {

    class ScopedTrace {
    public:
        static bool enabled() {
            static const bool on = qEnvironmentVariableIsSet("SIRIUS_TRACE_VIEW");
            return on;
        }

        explicit ScopedTrace(const char* what) : what_(what), on_(enabled()) {
            if (on_) clock_.start();
        }
        ~ScopedTrace() {
            if (on_) qInfo("%s %lld us", what_, clock_.nsecsElapsed() / 1000);
        }

        ScopedTrace(const ScopedTrace&) = delete;
        ScopedTrace& operator=(const ScopedTrace&) = delete;

    private:
        const char* what_;
        bool on_;
        QElapsedTimer clock_;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_TRACE_HPP
