#ifndef SIRIUS_APP_OPS_BUILTIN_HPP
#define SIRIUS_APP_OPS_BUILTIN_HPP

// Factories of the built-in operations, in menu order. Every ops/*.cpp
// defines one `std::unique_ptr<Operation> makeXxxOperation()`; the list in
// builtin_list.cpp is what registerBuiltinOperations() walks.

#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "core/operation.hpp"

namespace sirius::app {

    using OperationFactory = std::unique_ptr<Operation> (*)();
    std::vector<OperationFactory> builtinOperationFactories();

    std::unique_ptr<Operation> makeLoadOperation();
    std::unique_ptr<Operation> makeSimOperation();
    std::unique_ptr<Operation> makeDeconvolveOperation();
    std::unique_ptr<Operation> makeVolumeOperation();
    std::unique_ptr<Operation> makeEinsumOperation();
    std::unique_ptr<Operation> makeMaxProjectionOperation();
    std::unique_ptr<Operation> makeMeanOverTimeOperation();
    std::unique_ptr<Operation> makeContrastOperation();
    std::unique_ptr<Operation> makeFlatFieldOperation();
    std::unique_ptr<Operation> makeBleachOperation();
    std::unique_ptr<Operation> makeDeskewOperation();
    std::unique_ptr<Operation> makeCropPadOperation();
    std::unique_ptr<Operation> makeResampleOperation();
    std::unique_ptr<Operation> makeMergeOperation();
    std::unique_ptr<Operation> makeStitchOperation();
    std::unique_ptr<Operation> makeRegisterOperation();
    std::unique_ptr<Operation> makeTorchSegmentationOperation();
    std::unique_ptr<Operation> makeThresholdOperation();
    std::unique_ptr<Operation> makeLabelCleanupOperation();

    // Shared helpers for operation implementations (ops/common.cpp).
    // Runs `fn(c, t, progress)` for every (c, t) volume, reporting progress
    // 0..1 and honouring cancellation.
    void forEachVolume(const DatasetMeta& meta, const StepContext& ctx,
                       const std::function<void(Index c, Index t)>& fn);
    // "3 angles · 5 phases" style joining with " · ".
    std::string joinSummary(std::initializer_list<std::string> parts);
    // Channel label "488 α-actinin" for summaries.
    std::string channelName(const DatasetMeta& meta, Index c);

} // namespace sirius::app

#endif // SIRIUS_APP_OPS_BUILTIN_HPP
