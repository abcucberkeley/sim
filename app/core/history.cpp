#include "core/history.hpp"

#include <utility>

namespace sirius::app {

    void History::push(Command c) {
        redo_.clear();
        if (!c.mergeKey.empty() && !undo_.empty() && undo_.back().mergeKey == c.mergeKey) {
            // keep the oldest undo, take the newest redo
            undo_.back().redo = std::move(c.redo);
            undo_.back().label = std::move(c.label);
            return;
        }
        undo_.push_back(std::move(c));
        if (undo_.size() > limit_) undo_.erase(undo_.begin());
    }

    std::string History::undoLabel() const { return undo_.empty() ? std::string() : undo_.back().label; }
    std::string History::redoLabel() const { return redo_.empty() ? std::string() : redo_.back().label; }

    void History::undo() {
        if (undo_.empty()) return;
        Command c = std::move(undo_.back());
        undo_.pop_back();
        if (c.undo) c.undo();
        c.mergeKey.clear();   // a redone command never merges
        redo_.push_back(std::move(c));
    }

    void History::redo() {
        if (redo_.empty()) return;
        Command c = std::move(redo_.back());
        redo_.pop_back();
        if (c.redo) c.redo();
        undo_.push_back(std::move(c));
    }

    void History::clear() {
        undo_.clear();
        redo_.clear();
    }

} // namespace sirius::app
