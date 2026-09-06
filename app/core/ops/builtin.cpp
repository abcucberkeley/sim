// Registers every built-in operation. Each operation file exposes a factory
// declared in ops/builtin.hpp; this file is the only place that lists them.
#include "core/ops/builtin.hpp"

#include <mutex>

namespace sirius::app {

    void registerBuiltinOperations() {
        static std::once_flag once;
        std::call_once(once, [] {
            for (auto factory : builtinOperationFactories()) registerOperation(factory());
        });
    }

} // namespace sirius::app
