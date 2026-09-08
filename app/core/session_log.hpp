// Recording what a user did, so the sequence can be replayed, audited or
// learned from.
//
// The point is not an undo history -- History already does that, in memory,
// for this session. This is a durable record of the decisions: which dataset,
// which steps, which parameter went from what to what, what a run produced,
// and every correction painted afterwards. Written as JSON lines so a long
// session streams to disk without being held in memory, and so a reader can
// take it a line at a time.
#ifndef SIRIUS_APP_SESSION_LOG_HPP
#define SIRIUS_APP_SESSION_LOG_HPP

#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

#include <nlohmann/json.hpp>

namespace sirius::app {

    class SessionLog {
    public:
        SessionLog() = default;
        ~SessionLog();
        SessionLog(const SessionLog&) = delete;
        SessionLog& operator=(const SessionLog&) = delete;

        // Opens `path` for appending and writes one "session" line carrying
        // `header`. Throws when the file cannot be opened.
        void start(const std::filesystem::path& path, nlohmann::json header = {});
        void stop();
        bool recording() const;
        const std::filesystem::path& path() const noexcept { return path_; }
        // Lines written since start(), the "session" line included.
        std::uint64_t lines() const;

        // One event: `{"t": seconds since start, "event": <event>, ...fields}`.
        // Does nothing when not recording, so call sites need no guard.
        void record(std::string event, nlohmann::json fields = nlohmann::json::object());

    private:
        void write(nlohmann::json line);

        mutable std::mutex mutex_;
        std::ofstream out_;
        std::filesystem::path path_;
        std::chrono::steady_clock::time_point began_{};
        std::uint64_t lines_ = 0;
    };

} // namespace sirius::app

#endif // SIRIUS_APP_SESSION_LOG_HPP
