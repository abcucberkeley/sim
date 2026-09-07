#ifndef SIRIUS_APP_ERRORS_HPP
#define SIRIUS_APP_ERRORS_HPP

// The application layer's typed failures, next to CancelledError in
// app/core/cancel.hpp (which stays where it is: the executor and the run job
// reason about cancellation, not about errors).
//
// ProtocolError covers the worker wire protocol -- framing, the handshake,
// the transport -- so that "the connection to the worker broke" can be told
// from "the step the worker ran failed", which used to be a matter of
// noticing that one message starts with "rpc: " and the other does not.
//
// A failure the worker *reports* is not a ProtocolError: it travels over a
// healthy connection and stays a std::runtime_error carrying the worker's own
// message.

#include <sirius/errors.hpp>

namespace sirius::app {

    // The worker wire protocol failed: a frame that does not decode, a
    // header that is not plausible, a socket that will not connect, a
    // handshake the two ends do not agree on.
    class ProtocolError : public sirius::SiriusError {
    public:
        using sirius::SiriusError::SiriusError;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_ERRORS_HPP
