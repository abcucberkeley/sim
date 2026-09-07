#ifndef SIRIUS_ERRORS_HPP
#define SIRIUS_ERRORS_HPP

// The library's typed failures. Everything used to be a std::runtime_error,
// so a caller that wanted to tell "the file is unreadable" from "the caller
// passed the wrong shape" had to match on the message text -- exactly the
// coupling CancelledError (app/core/cancel.hpp) was introduced to remove.
//
// The hierarchy is deliberately shallow: SiriusError for the library's own
// run-time failures, IoError for anything a file did, ShapeError for an
// extent mismatch. Everything else keeps the standard type that already says
// what it means -- std::invalid_argument for a bad argument,
// std::out_of_range for a bad index, std::overflow_error for size arithmetic.
//
// Every type here keeps the base its throw sites already had, so this is a
// refinement and not a behaviour change: catch (const std::runtime_error&)
// and catch (const std::exception&) see the same errors as before, and so do
// the Python bindings, whose RuntimeError / ValueError mapping follows those
// bases.
//
// ShapeError is the one that does not sit under SiriusError. A rank or
// extent mismatch is a programming error in the caller, and the single funnel
// every one of them goes through (detail::throwShapeMismatch in buffer.cpp)
// has always thrown std::invalid_argument -- which pybind11 surfaces as
// ValueError, as bindings/tests assert. Re-rooting it under SiriusError would
// turn those into RuntimeError, so it stays where its meaning already put it.

#include <stdexcept>

namespace sirius {

    // Base of the library's run-time failures: something the environment or
    // the data did, not something the caller passed.
    class SiriusError : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    // A file could not be opened, read, written, or made sense of: a missing
    // TIFF, a truncated directory, a zarr store with no metadata, a config
    // line that does not parse. The message names the path where it has one.
    class IoError : public SiriusError {
    public:
        using SiriusError::SiriusError;
    };

    // Rank / extent mismatch between operands, or between a buffer and the
    // shape it is asked to take. See the note above on why this is an
    // std::invalid_argument rather than a SiriusError.
    class ShapeError : public std::invalid_argument {
    public:
        using std::invalid_argument::invalid_argument;
    };

} // namespace sirius

#endif // SIRIUS_ERRORS_HPP
