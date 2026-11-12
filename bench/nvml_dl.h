// Minimal NVML loader, resolved at runtime with dlopen.
// The benchmark therefore builds without nvml.h and still runs (without
// energy readings) where no NVIDIA driver is present, e.g. on a login node.
#pragma once

#include <dlfcn.h>

#include <cctype>
#include <cstdio>
#include <string>

struct nvml_sample {
    bool               ok        = false;
    unsigned long long energy_mj = 0;  // cumulative energy since driver load (mJ)
    unsigned           power_mw  = 0;  // power.draw (on Hopper: averaged by the driver)
    unsigned           sm_mhz    = 0;
    unsigned           mem_mhz   = 0;
    unsigned           temp_c    = 0;
    unsigned long long reasons   = 0;  // clock throttle / clock event reasons bitmask
};

class nvml_dev {
    using nvmlDevice_t = void *;
    using ret_t        = int;  // nvmlReturn_t, 0 == NVML_SUCCESS

    void *       lib = nullptr;
    nvmlDevice_t dev = nullptr;

    ret_t (*p_init)()                                                   = nullptr;
    ret_t (*p_shutdown)()                                               = nullptr;
    ret_t (*p_count)(unsigned *)                                        = nullptr;
    ret_t (*p_by_index)(unsigned, nvmlDevice_t *)                       = nullptr;
    ret_t (*p_by_pci)(const char *, nvmlDevice_t *)                     = nullptr;
    ret_t (*p_name)(nvmlDevice_t, char *, unsigned)                     = nullptr;
    ret_t (*p_energy)(nvmlDevice_t, unsigned long long *)               = nullptr;
    ret_t (*p_power)(nvmlDevice_t, unsigned *)                          = nullptr;
    ret_t (*p_clock)(nvmlDevice_t, int, unsigned *)                     = nullptr;
    ret_t (*p_temp)(nvmlDevice_t, int, unsigned *)                      = nullptr;
    ret_t (*p_reasons)(nvmlDevice_t, unsigned long long *)              = nullptr;
    ret_t (*p_plimit)(nvmlDevice_t, unsigned *)                         = nullptr;

    template <typename F> void sym(F & f, const char * name) {
        f = reinterpret_cast<F>(dlsym(lib, name));
    }

  public:
    std::string name;
    std::string how;  // how the device was matched
    unsigned    power_limit_mw = 0;

    bool ok() const { return dev != nullptr; }

    // pci_bus_id: e.g. "0000:c1:00.0" (from ggml), may be null.
    bool open(const char * pci_bus_id) {
        lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib) {
            fprintf(stderr, "nvml: libnvidia-ml not found, energy will not be measured\n");
            return false;
        }
        sym(p_init, "nvmlInit_v2");
        sym(p_shutdown, "nvmlShutdown");
        sym(p_count, "nvmlDeviceGetCount_v2");
        sym(p_by_index, "nvmlDeviceGetHandleByIndex_v2");
        sym(p_by_pci, "nvmlDeviceGetHandleByPciBusId_v2");
        sym(p_name, "nvmlDeviceGetName");
        sym(p_energy, "nvmlDeviceGetTotalEnergyConsumption");
        sym(p_power, "nvmlDeviceGetPowerUsage");
        sym(p_clock, "nvmlDeviceGetClockInfo");
        sym(p_temp, "nvmlDeviceGetTemperature");
        sym(p_reasons, "nvmlDeviceGetCurrentClocksEventReasons");
        if (!p_reasons) sym(p_reasons, "nvmlDeviceGetCurrentClocksThrottleReasons");
        sym(p_plimit, "nvmlDeviceGetPowerManagementLimit");
        if (!p_init || !p_energy || !p_by_index || p_init() != 0) {
            fprintf(stderr, "nvml: initialisation failed, energy will not be measured\n");
            return false;
        }
        if (pci_bus_id && p_by_pci) {
            std::string id = pci_bus_id;
            for (auto & c : id) c = (char) toupper((unsigned char) c);
            if (p_by_pci(id.c_str(), &dev) == 0) how = std::string("pci ") + pci_bus_id;
            else dev = nullptr;
        }
        if (!dev) {
            unsigned n = 0;
            if (p_count) p_count(&n);
            if (n != 1) {
                fprintf(stderr, "nvml: could not match device by PCI id and %u GPUs are visible; using index 0\n", n);
            }
            if (p_by_index(0, &dev) != 0) {
                dev = nullptr;
                fprintf(stderr, "nvml: no device\n");
                return false;
            }
            how = "index 0";
        }
        char buf[96] = {0};
        if (p_name && p_name(dev, buf, sizeof(buf)) == 0) name = buf;
        if (p_plimit) p_plimit(dev, &power_limit_mw);
        return true;
    }

    nvml_sample sample() const {
        nvml_sample s;
        if (!dev) return s;
        s.ok = p_energy(dev, &s.energy_mj) == 0;
        if (p_power) p_power(dev, &s.power_mw);
        if (p_clock) {
            p_clock(dev, 1 /*NVML_CLOCK_SM*/, &s.sm_mhz);
            p_clock(dev, 2 /*NVML_CLOCK_MEM*/, &s.mem_mhz);
        }
        if (p_temp) p_temp(dev, 0 /*NVML_TEMPERATURE_GPU*/, &s.temp_c);
        if (p_reasons) p_reasons(dev, &s.reasons);
        return s;
    }

    ~nvml_dev() {
        if (lib) {
            if (p_shutdown && dev) p_shutdown();
            dlclose(lib);
        }
    }
};
