#ifndef SIRIUS_APP_HISTORY_HPP
#define SIRIUS_APP_HISTORY_HPP

// Undo / redo as a stack of reversible commands. The workbench builds
// commands from closures that capture "before" and "after" state (pipeline
// JSON, view state, label diffs), so any edit -- from the UI, the assistant
// or a script -- is undone the same way.

#include <functional>
#include <string>
#include <vector>

namespace sirius::app {

    struct Command {
        std::string label;              // "Set Wiener 0.001 → 0.002"
        std::function<void()> undo;
        std::function<void()> redo;
        // Consecutive commands with the same non-empty merge key collapse into
        // one entry (slider drags, brush strokes): the newest command's
        // closures replace the entry's, so the caller composes them to span
        // the whole merged range (see Workbench::pushEdit). Any other push,
        // and an undo, ends the group: the history is the single source of
        // truth for what merges (mergesWith).
        std::string mergeKey;
    };

    class History {
    public:
        void push(Command c);            // clears the redo stack
        // True when a command with this key would merge into the top undo
        // entry (same non-empty key, and no other push, undo or redo since):
        // what the caller checks to compose the "before" state of a merged
        // group.
        bool mergesWith(const std::string& key) const noexcept;
        bool canUndo() const noexcept { return !undo_.empty(); }
        bool canRedo() const noexcept { return !redo_.empty(); }
        std::string undoLabel() const;   // "" when nothing
        std::string redoLabel() const;
        void undo();
        void redo();
        void clear();
        std::size_t size() const noexcept { return undo_.size(); }
        void setLimit(std::size_t n) noexcept { limit_ = n; }

    private:
        std::vector<Command> undo_, redo_;
        bool mergeOpen_ = false;         // the top entry is still taking merges
        std::size_t limit_ = 200;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_HISTORY_HPP
