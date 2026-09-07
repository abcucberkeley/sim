#include "core/history.hpp"

#include <utility>

namespace sirius::app {

    void History::push(Command c) {
        redo_.clear();
        if (mergesWith(c.mergeKey)) {
            undo_.back() = std::move(c);   // the newest command spans the whole group
            return;
        }
        mergeOpen_ = !c.mergeKey.empty();
        undo_.push_back(std::move(c));
        if (undo_.size() > limit_) undo_.erase(undo_.begin());
    }

    bool History::mergesWith(const std::string& key) const noexcept {
        return mergeOpen_ && !key.empty() && !undo_.empty() && undo_.back().mergeKey == key;
    }

    std::string History::undoLabel() const { return undo_.empty() ? std::string() : undo_.back().label; }
    std::string History::redoLabel() const { return redo_.empty() ? std::string() : redo_.back().label; }

    void History::undo() {
        mergeOpen_ = false;   // an undo ends the group, even when it uncovers it again
        if (undo_.empty()) return;
        Command c = std::move(undo_.back());
        undo_.pop_back();
        if (c.undo) c.undo();
        c.mergeKey.clear();   // a redone command never merges
        redo_.push_back(std::move(c));
    }

    void History::redo() {
        mergeOpen_ = false;
        if (redo_.empty()) return;
        Command c = std::move(redo_.back());
        redo_.pop_back();
        if (c.redo) c.redo();
        undo_.push_back(std::move(c));
    }

    void History::clear() {
        mergeOpen_ = false;
        undo_.clear();
        redo_.clear();
    }

} // namespace sirius::app
