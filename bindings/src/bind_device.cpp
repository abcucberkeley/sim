#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <sirius/device.hpp>

#include <sstream>
#include <stdexcept>

namespace nb = nanobind;
using namespace sirius;

namespace {
    // "cpu", "cuda", "cuda:1" -> Device
    Device parseDevice(const std::string& s) {
        if (s == "cpu") return Device::cpu();
        if (s == "cuda") return Device::cuda(0);
        if (s.rfind("cuda:", 0) == 0) {
            try {
                return Device::cuda(std::stoi(s.substr(5)));
            } catch (const std::exception&) {
                throw std::invalid_argument("invalid device string: " + s);
            }
        }
        throw std::invalid_argument("invalid device string: '" + s + "' (expected 'cpu', 'cuda' or 'cuda:N')");
    }
} // namespace

void bind_device(nb::module_& m) {
    nb::enum_<DeviceType>(m, "DeviceType")
        .value("Cpu", DeviceType::Cpu)
        .value("Cuda", DeviceType::Cuda);

    nb::class_<Device>(m, "Device",
            "Where memory lives and kernels run: Device.cpu() or Device.cuda(index). "
            "Functions taking a device also accept the strings 'cpu', 'cuda', 'cuda:N'.")
        .def(nb::init<>())
        .def("__init__", [](Device* self, const std::string& s) { new (self) Device(parseDevice(s)); },
             nb::arg("spec"))
        .def_static("cpu", &Device::cpu)
        .def_static("cuda", &Device::cuda, nb::arg("index") = 0)
        .def_prop_ro("type", [](const Device& d) { return d.type; })
        .def_prop_ro("index", [](const Device& d) { return d.index; })
        .def_prop_ro("is_cpu", &Device::isCpu)
        .def_prop_ro("is_cuda", &Device::isCuda)
        .def("__eq__", [](const Device& a, const Device& b) { return a == b; }, nb::is_operator())
        .def("__ne__", [](const Device& a, const Device& b) { return a != b; }, nb::is_operator())
        .def("__hash__", [](const Device& d) { return static_cast<int>(d.type) * 4096 + d.index; })
        .def("__str__", [](const Device& d) { return toString(d); })
        .def("__repr__", [](const Device& d) { return "Device('" + toString(d) + "')"; });
    nb::implicitly_convertible<std::string, Device>();

    nb::class_<Stream>(m, "Stream",
            "Ordering handle for asynchronous GPU work. On the CPU it is a no-op.")
        .def(nb::init<>())
        .def(nb::init<Device>(), nb::arg("device"))
        .def_prop_ro("device", &Stream::device)
        .def("synchronize", &Stream::synchronize)
        .def("__repr__", [](const Stream& s) { return "Stream(" + toString(s.device()) + ")"; });

    nb::class_<DeviceProperties>(m, "DeviceProperties")
        .def_ro("name", &DeviceProperties::name)
        .def_ro("compute_major", &DeviceProperties::computeMajor)
        .def_ro("compute_minor", &DeviceProperties::computeMinor)
        .def_ro("multiprocessor_count", &DeviceProperties::multiprocessorCount)
        .def_ro("total_memory_bytes", &DeviceProperties::totalMemoryBytes)
        .def("__repr__", [](const DeviceProperties& p) {
            std::ostringstream os;
            os << "DeviceProperties(name='" << p.name << "', cc=" << p.computeMajor << "." << p.computeMinor
               << ", sms=" << p.multiprocessorCount << ", memory=" << (p.totalMemoryBytes >> 20) << " MiB)";
            return os.str();
        });

    m.def("built_with_cuda", &builtWithCuda, "True when the extension was compiled with CUDA support.");
    m.def("built_with_nvtiff", &builtWithNvTiff, "True when GPU TIFF decoding (nvTIFF) was compiled in.");
    m.def("cuda_device_count", &cudaDeviceCount, "Number of usable CUDA devices (0 without CUDA).");
    m.def("cuda_available", &cudaAvailable, "True when at least one CUDA device is usable.");
    m.def("device_properties", &deviceProperties, nb::arg("device"),
          "Name, compute capability and memory of a CUDA device.");
    m.def("synchronize_device", &synchronizeDevice, nb::arg("device"));
}
