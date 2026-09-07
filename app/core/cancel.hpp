#ifndef SIRIUS_APP_CANCEL_HPP
#define SIRIUS_APP_CANCEL_HPP

// Cancellation as a type. A step that notices the run was cancelled throws
// CancelledError; the executor and the run job recognise it by type and do
// not blame the step. Every throw site in the application layer
// (StepContext::throwIfCancelled, the export writers) uses the type.
//
// The library (src/) cannot include an application header, so its own
// cancellation -- writeTiffStack's progress callback in src/tiff_io.cpp --
// still throws std::runtime_error("cancelled"). isCancellation() therefore
// keeps accepting that message, which is the seam between the two layers
// rather than a leftover.

#include <exception>
#include <stdexcept>
#include <string>

namespace sirius::app {

    class CancelledError : public std::runtime_error {
    public:
        CancelledError() : std::runtime_error("cancelled") {}
        explicit CancelledError(const std::string& what) : std::runtime_error(what) {}
    };

    // True for a CancelledError or, as the fallback, any exception whose
    // message is exactly "cancelled".
    inline bool isCancellation(const std::exception& e) noexcept {
        if (dynamic_cast<const CancelledError*>(&e)) return true;
        const char* what = e.what();
        return what && std::string(what) == "cancelled";
    }

} // namespace sirius::app

#endif // SIRIUS_APP_CANCEL_HPP
