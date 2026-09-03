#ifndef SIRIUS_TESTS_TEMP_PATH_HPP
#define SIRIUS_TESTS_TEMP_PATH_HPP

// Scratch-file naming shared by the test sources. catch_discover_tests runs
// every TEST_CASE as its own process, so a per-process counter alone repeats
// the same names across the processes `ctest -j` runs concurrently; a random
// per-process token keeps them apart.

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <system_error>

namespace sirius::test {

    // <tmp>/sirius_<tag>_<token>_<n><suffix>, unique within and across processes
    inline std::filesystem::path uniqueTempPath(const char* tag, const char* suffix) {
        static const std::string token = [] {
            std::random_device rd;
            char buf[17];
            std::snprintf(buf, sizeof buf, "%08x%08x", rd(), rd());
            return std::string(buf);
        }();
        static std::atomic<int> counter{0};
        return std::filesystem::temp_directory_path() /
               (std::string("sirius_") + tag + "_" + token + "_" +
                std::to_string(counter.fetch_add(1)) + suffix);
    }

    // RAII: removes the file on scope exit (missing file is not an error).
    struct TempFile {
        std::filesystem::path path;
        std::string str;   // path.string(), for APIs taking std::string

        explicit TempFile(const char* tag, const char* suffix)
            : path(uniqueTempPath(tag, suffix)), str(path.string()) {}
        ~TempFile() {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
        TempFile(const TempFile&) = delete;
        TempFile& operator=(const TempFile&) = delete;
    };

} // namespace sirius::test

#endif // SIRIUS_TESTS_TEMP_PATH_HPP
