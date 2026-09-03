// Standalone benchmark: time SIRIUS's TIFF stack reader on a BigTIFF, on the
// CPU (libtiff, OpenMP over pages) or on a CUDA device (nvTIFF decode straight
// into device memory).
//
// Built inside SIRIUS's CMake (gated by SIRIUS_ENABLE_BENCHMARKS) and linked
// against the sirius static lib. Kept separate from the cpp-tiff bench so the
// two libtiff builds never share a CMake target. See benchmarks/CMakeLists.txt.
//
//     usage: bench_tiff_sirius <path> [repeats] [cpu|cuda|cuda:N]
//
// Emits one "<name>\t<seconds>\t<bytes>" line for bench_tiff.py to merge; the
// name is "sirius" for the CPU and "sirius-cuda" for a GPU read. GPU timings
// include the decode and the implicit stream synchronization, i.e. the data is
// ready in device memory when the clock stops.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include <sirius/device.hpp>
#include <sirius/tiff_io.hpp>

#include "bench_timer.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path> [repeats] [cpu|cuda|cuda:N]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int repeats = (argc > 2) ? std::atoi(argv[2]) : 3;
    const std::string where = (argc > 3) ? argv[3] : "cpu";

    sirius::Device device = sirius::Device::cpu();
    if (where == "cuda") device = sirius::Device::cuda(0);
    else if (where.rfind("cuda:", 0) == 0) device = sirius::Device::cuda(std::atoi(where.c_str() + 5));
    else if (where != "cpu") {
        std::fprintf(stderr, "unknown device '%s'\n", where.c_str());
        return 2;
    }

    try {
        sirius::requireDevice(device);
        // Open once, outside the timed region: metadata parsing is a fixed
        // cost the production pipeline also pays once per file.
        sirius::TiffFile file(path);
        sirius::TiffReadOptions opts;
        opts.device = device;
        if (device.isCuda()) {
            std::string reason;
            if (!file.gpuDecodable(device, &reason))
                std::fprintf(stderr, "note: nvTIFF cannot decode this file (%s); timing the libtiff fallback + upload\n",
                             reason.c_str());
        }

        std::uint64_t bytes = 0;
        const double s = bench::time_min([&] {
            auto stack = file.readStack<std::uint16_t>(opts);
            bytes = static_cast<std::uint64_t>(stack.bytes());
        }, repeats);

        bench::report(device.isCuda() ? "sirius-cuda" : "sirius", s, bytes);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sirius read failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
