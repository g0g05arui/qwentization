// matvec_energy: energy and bandwidth of the decode matrix-vector product
// (ggml MUL_MAT with n = 1) for different GGUF weight formats.
//
// For each format the tool
//   1. creates `nmat` weight matrices of shape m x k (random values in [-1, 1],
//      quantized on the host with ggml_quantize_chunk, as test-backend-ops does),
//   2. builds one ggml graph that runs y_i = W_i * x for all matrices,
//      repeated `reps` times so that a single graph launch lasts tens of ms,
//   3. runs the graph for `warmup` seconds, then for `seconds` seconds while
//      reading the NVML cumulative energy counter before and after,
//   4. reports time per pass, effective bandwidth and pJ per weight / byte.
//
// The idle power of the GPU (with this process's CUDA context alive) is
// measured once at the start and once at the end, and is used to compute the
// "active" energy (total minus idle).
//
// A pass is one product over all nmat matrices: W = nmat * m * k weights.

#include "ggml.h"
#include "ggml-backend.h"

#include "nvml_dl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

static double now_s() {
    static const auto t0 = clk::now();
    return std::chrono::duration<double>(clk::now() - t0).count();
}

// ---------------------------------------------------------------------------
// options

struct options {
    std::vector<ggml_type> types;
    int64_t     m        = 17408;  // output rows   (Qwen MLP gate/up)
    int64_t     k        = 5120;   // input columns (hidden size)
    int         nmat     = 4;      // distinct matrices, working set must exceed L2
    bool        batched  = false;  // one 3-D tensor (as test-backend-ops bs={nmat,1}) vs separate tensors
    int         reps     = 0;      // repetitions of all matrices per graph (0 = auto)
    double      seconds  = 20.0;   // measured time per format
    double      warmup   = 3.0;    // unmeasured run time before each measurement
    double      idle     = 10.0;   // idle measurement at start and end
    double      cooldown = 5.0;    // idle pause between formats (logged, not measured)
    int         repeat   = 1;      // number of passes over the format list
    unsigned    seed     = 0;      // 0 = keep order in pass 1, shuffle later passes
    int         interval = 20;     // sampling interval (ms)
    int         threads  = 0;      // host threads for quantization (0 = auto)
    bool        check    = true;   // compare matrix 0 against a host reference
    std::string device   = "";     // ggml device name, e.g. CUDA0 (default: first GPU)
    std::string out      = "matvec_energy";  // output prefix
};

static const char * default_types = "f16,q8_0,q4_0,q4_1,q4_K,q5_K,q6_K,iq4_xs,iq4_nl";

static void usage(const char * argv0) {
    printf("usage: %s [options]\n"
           "  -t, --types LIST     comma separated ggml types (default: %s)\n"
           "  -m N                 rows of each matrix (default 17408)\n"
           "  -k N                 columns of each matrix (default 5120)\n"
           "      --nmat N         distinct matrices (default 4)\n"
           "      --batched        store the matrices as one 3-D tensor (one MUL_MAT op)\n"
           "      --reps N         repetitions per graph (default: auto, ~8 GB of weights per graph)\n"
           "  -s, --seconds S      measured seconds per format (default 20)\n"
           "      --warmup S       warmup seconds per format (default 3)\n"
           "      --idle S         idle measurement at start and end (default 10)\n"
           "      --cooldown S     idle pause between formats (default 5)\n"
           "  -r, --repeat N       passes over all formats (default 1)\n"
           "      --seed N         shuffle format order of passes 2.. with this seed (default 0)\n"
           "      --interval MS    power sampling interval (default 20)\n"
           "      --threads N      host threads for quantization\n"
           "      --no-check       skip the correctness check\n"
           "  -d, --device NAME    ggml device (default: first GPU, else CPU)\n"
           "  -o, --out PREFIX     output prefix (default matvec_energy)\n"
           "outputs: PREFIX_summary.csv (one row per measurement), PREFIX_samples.csv (power trace)\n",
           argv0, default_types);
}

static ggml_type parse_type(const std::string & s) {
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const char * name = ggml_type_name((ggml_type) i);
        if (name && strcasecmp(name, s.c_str()) == 0 && ggml_type_size((ggml_type) i) > 0) {
            return (ggml_type) i;
        }
    }
    fprintf(stderr, "unknown type: %s\n", s.c_str());
    exit(1);
}

static void parse_types(options & o, const std::string & list) {
    o.types.clear();
    size_t pos = 0;
    while (pos <= list.size()) {
        size_t end = list.find(',', pos);
        if (end == std::string::npos) end = list.size();
        if (end > pos) o.types.push_back(parse_type(list.substr(pos, end - pos)));
        pos = end + 1;
    }
}

static options parse_args(int argc, char ** argv) {
    options o;
    parse_types(o, default_types);
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if      (a == "-t" || a == "--types")    parse_types(o, next());
        else if (a == "-m")                      o.m        = std::stoll(next());
        else if (a == "-k")                      o.k        = std::stoll(next());
        else if (a == "--nmat")                  o.nmat     = std::stoi(next());
        else if (a == "--batched")               o.batched  = true;
        else if (a == "--reps")                  o.reps     = std::stoi(next());
        else if (a == "-s" || a == "--seconds")  o.seconds  = std::stod(next());
        else if (a == "--warmup")                o.warmup   = std::stod(next());
        else if (a == "--idle")                  o.idle     = std::stod(next());
        else if (a == "--cooldown")              o.cooldown = std::stod(next());
        else if (a == "-r" || a == "--repeat")   o.repeat   = std::stoi(next());
        else if (a == "--seed")                  o.seed     = (unsigned) std::stoul(next());
        else if (a == "--interval")              o.interval = std::stoi(next());
        else if (a == "--threads")               o.threads  = std::stoi(next());
        else if (a == "--no-check")              o.check    = false;
        else if (a == "-d" || a == "--device")   o.device   = next();
        else if (a == "-o" || a == "--out")      o.out      = next();
        else if (a == "-h" || a == "--help")     { usage(argv[0]); exit(0); }
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); usage(argv[0]); exit(1); }
    }
    if (o.types.empty() || o.m <= 0 || o.k <= 0 || o.nmat <= 0 || o.repeat <= 0 || o.interval <= 0) {
        fprintf(stderr, "invalid arguments\n");
        exit(1);
    }
    if (o.threads <= 0) o.threads = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    return o;
}

// ---------------------------------------------------------------------------
// power sampler: background thread that logs NVML readings with a phase label

struct sample_row {
    double      t;
    std::string phase;
    nvml_sample s;
};

class sampler {
    const nvml_dev &        nv;
    int                     interval_ms;
    std::atomic<bool>       stop_flag{false};
    std::thread             th;
    mutable std::mutex      mtx;
    std::string             phase_ = "init";
    std::vector<sample_row> rows;

  public:
    sampler(const nvml_dev & nv, int interval_ms) : nv(nv), interval_ms(interval_ms) {
        if (!nv.ok()) return;
        th = std::thread([this] {
            auto next = clk::now();
            while (!stop_flag.load()) {
                nvml_sample s = this->nv.sample();
                double      t = now_s();
                {
                    std::lock_guard<std::mutex> lk(mtx);
                    rows.push_back({t, phase_, s});
                }
                next += std::chrono::milliseconds(this->interval_ms);
                std::this_thread::sleep_until(next);
            }
        });
    }
    ~sampler() { stop(); }

    void stop() {
        stop_flag = true;
        if (th.joinable()) th.join();
    }

    void phase(const std::string & p) {
        std::lock_guard<std::mutex> lk(mtx);
        phase_ = p;
    }

    // samples of a phase, with `trim` seconds dropped at both ends
    std::vector<nvml_sample> get(const std::string & p, double trim) const {
        std::lock_guard<std::mutex> lk(mtx);
        double t0 = 1e30, t1 = -1e30;
        for (auto & r : rows) if (r.phase == p) { t0 = std::min(t0, r.t); t1 = std::max(t1, r.t); }
        std::vector<nvml_sample> v;
        for (auto & r : rows) {
            if (r.phase == p && r.t >= t0 + trim && r.t <= t1 - trim) v.push_back(r.s);
        }
        return v;
    }

    void write(const std::string & path) const {
        std::lock_guard<std::mutex> lk(mtx);
        FILE * f = fopen(path.c_str(), "w");
        if (!f) { perror(path.c_str()); return; }
        fprintf(f, "t_s,phase,power_w,energy_j,sm_mhz,mem_mhz,temp_c,clock_reasons\n");
        for (auto & r : rows) {
            fprintf(f, "%.4f,%s,%.3f,%.3f,%u,%u,%u,0x%llx\n", r.t, r.phase.c_str(), r.s.power_mw / 1000.0,
                    r.s.energy_mj / 1000.0, r.s.sm_mhz, r.s.mem_mhz, r.s.temp_c, r.s.reasons);
        }
        fclose(f);
    }
};

static double median(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// measure the energy of a time interval: returns average power (W) over [start, end]
struct energy_meter {
    const nvml_dev & nv;
    nvml_sample      s0;
    double           t0 = 0;

    explicit energy_meter(const nvml_dev & nv) : nv(nv) {}
    void   start() { s0 = nv.sample(); t0 = now_s(); }
    // returns {seconds, joules}; joules is NAN without NVML
    std::pair<double, double> stop() const {
        nvml_sample s1 = nv.sample();
        double      t  = now_s() - t0;
        double      e  = (s0.ok && s1.ok) ? (s1.energy_mj - s0.energy_mj) / 1000.0 : NAN;
        return {t, e};
    }
};

static double measure_idle(const nvml_dev & nv, sampler & smp, const std::string & label, double seconds) {
    if (seconds <= 0 || !nv.ok()) return NAN;
    printf("idle (%s): %.0f s ... ", label.c_str(), seconds);
    fflush(stdout);
    smp.phase(label);
    energy_meter em(nv);
    em.start();
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    auto [t, e] = em.stop();
    smp.phase("between");
    printf("%.1f W\n", e / t);
    return e / t;
}

// ---------------------------------------------------------------------------
// host side data

// random values in [-1, 1], converted to `type` with ggml_quantize_chunk
static std::vector<uint8_t> make_weights(ggml_type type, int64_t rows, int64_t k, unsigned seed, int n_threads) {
    std::vector<uint8_t> q(ggml_row_size(type, k) * rows);
    std::vector<float>   imatrix(k, 1.0f);  // dummy importance matrix
    const float *        im = ggml_quantize_requires_imatrix(type) ? imatrix.data() : nullptr;

    std::vector<std::thread> pool;
    for (int t = 0; t < n_threads; t++) {
        pool.emplace_back([&, t] {
            const int64_t r0 = rows * t / n_threads, r1 = rows * (t + 1) / n_threads;
            std::mt19937  gen(seed * 7919u + t);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            const int64_t chunk = 256;  // rows per quantize call
            std::vector<float> f(chunk * k);
            for (int64_t r = r0; r < r1; r += chunk) {
                const int64_t nr = std::min(chunk, r1 - r);
                for (int64_t i = 0; i < nr * k; i++) f[i] = dist(gen);
                // start = 0: src and dst are both offset by `start` inside ggml_quantize_chunk
                ggml_quantize_chunk(type, f.data(), q.data() + r * ggml_row_size(type, k), 0, nr, k, im);
            }
        });
    }
    for (auto & th : pool) th.join();
    return q;
}

// normalized mean squared error of y against a host reference
// (dequantized weights, double accumulation), matrix 0 only
static double check_matrix(ggml_type type, const std::vector<uint8_t> & w, const std::vector<float> & x,
                           const std::vector<float> & y, int64_t m, int64_t k) {
    const auto *       tt = ggml_get_type_traits(type);
    std::vector<float> row(k);
    const size_t       rs = ggml_row_size(type, k);
    double             err = 0, ref2 = 0;
    for (int64_t r = 0; r < m; r++) {
        tt->to_float(w.data() + r * rs, row.data(), k);
        double acc = 0;
        for (int64_t i = 0; i < k; i++) acc += (double) row[i] * x[i];
        err  += (acc - y[r]) * (acc - y[r]);
        ref2 += acc * acc;
    }
    return err / ref2;
}

// ---------------------------------------------------------------------------
// one measurement

struct result {
    int         pass, index;
    std::string type;
    double      bpw;           // bytes per weight
    int64_t     weights;       // weights per pass
    int64_t     bytes;         // weight bytes per pass
    int         reps;
    int64_t     passes;
    double      seconds, joules;
    double      p_med;         // median sampled power (W), ends trimmed
    double      nmse;
    unsigned    sm_min, sm_max, mem_min, mem_max, temp_max;
    unsigned long long reasons;
};

static bool run_one(const options & o, ggml_backend_t backend, const nvml_dev & nv, sampler & smp, ggml_type type,
                    int pass, int index, result & res) {
    const int64_t m = o.m, k = o.k;
    const int     nm = o.nmat;
    const char *  tname = ggml_type_name(type);

    if (k % ggml_blck_size(type) != 0) {
        fprintf(stderr, "%s: k=%lld is not a multiple of the block size %lld, skipped\n", tname, (long long) k,
                (long long) ggml_blck_size(type));
        return false;
    }

    // tensors -----------------------------------------------------------------
    ggml_init_params ip = { ggml_tensor_overhead() * (2 * nm + 8), nullptr, true };
    ggml_context *   ctx_w  = ggml_init(ip);
    ggml_context *   ctx_io = ggml_init(ip);

    std::vector<ggml_tensor *> W, Y;
    ggml_tensor *              x;
    if (o.batched) {
        W.push_back(ggml_new_tensor_3d(ctx_w, type, k, m, nm));
        x = ggml_new_tensor_3d(ctx_io, GGML_TYPE_F32, k, 1, nm);
        Y.push_back(ggml_mul_mat(ctx_io, W[0], x));
    } else {
        x = ggml_new_tensor_2d(ctx_io, GGML_TYPE_F32, k, 1);
        for (int i = 0; i < nm; i++) {
            W.push_back(ggml_new_tensor_2d(ctx_w, type, k, m));
            Y.push_back(ggml_mul_mat(ctx_io, W.back(), x));
        }
    }

    bool ok = true;
    for (auto * y : Y) ok = ok && ggml_backend_supports_op(backend, y);
    if (!ok) {
        fprintf(stderr, "%s: MUL_MAT not supported by %s, skipped\n", tname, ggml_backend_name(backend));
        ggml_free(ctx_w);
        ggml_free(ctx_io);
        return false;
    }

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_buffer_t buf_io = ggml_backend_alloc_ctx_tensors(ctx_io, backend);
    if (!buf_w || !buf_io) {
        fprintf(stderr, "%s: allocation failed\n", tname);
        if (buf_w) ggml_backend_buffer_free(buf_w);
        if (buf_io) ggml_backend_buffer_free(buf_io);
        ggml_free(ctx_w);
        ggml_free(ctx_io);
        return false;
    }
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // data -------------------------------------------------------------------
    printf("[%s] quantizing %d x %lld x %lld ... ", tname, nm, (long long) m, (long long) k);
    fflush(stdout);
    smp.phase("setup");
    const size_t         mat_bytes = ggml_row_size(type, k) * m;
    std::vector<uint8_t> w0;  // matrix 0, kept for the check
    for (int i = 0; i < nm; i++) {
        auto q = make_weights(type, m, k, 1000u * pass + 10u * index + i + 1, o.threads);
        if (o.batched) ggml_backend_tensor_set(W[0], q.data(), i * mat_bytes, mat_bytes);
        else           ggml_backend_tensor_set(W[i], q.data(), 0, mat_bytes);
        if (i == 0) w0 = std::move(q);
    }
    std::vector<float> xh(ggml_nelements(x));
    {
        std::mt19937 gen(12345 + pass);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto & v : xh) v = dist(gen);
        ggml_backend_tensor_set(x, xh.data(), 0, xh.size() * sizeof(float));
    }
    printf("done\n");

    const int64_t weights = (int64_t) nm * m * k;
    int64_t       bytes   = 0;
    for (auto * w : W) bytes += ggml_nbytes(w);

    // graph ------------------------------------------------------------------
    int reps = o.reps;
    if (reps <= 0) {
        const double target = 8.0 * (1ull << 30);  // ~8 GB of weights per graph launch
        reps = (int) std::clamp<double>(std::ceil(target / bytes), 1.0, std::max(1, 4096 / (int) Y.size()));
    }
    const size_t   gsize  = Y.size() * reps + 64;
    ggml_init_params gp   = { ggml_graph_overhead_custom(gsize, false), nullptr, true };
    ggml_context *   ctx_g = ggml_init(gp);
    ggml_cgraph *    gf    = ggml_new_graph_custom(ctx_g, gsize, false);
    for (auto * y : Y) ggml_build_forward_expand(gf, y);
    for (int r = 1; r < reps; r++) {
        for (auto * y : Y) ggml_graph_add_node(gf, y);
    }

    auto compute = [&]() {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: graph compute failed\n", tname);
            exit(1);
        }
    };

    // check ------------------------------------------------------------------
    smp.phase("warmup");
    compute();
    double nmse = NAN;
    if (o.check) {
        std::vector<float> y(m);
        ggml_backend_tensor_get(Y[0], y.data(), 0, m * sizeof(float));
        std::vector<float> x0(xh.begin(), xh.begin() + k);
        nmse = check_matrix(type, w0, x0, y, m, k);
        printf("[%s] check: nmse = %.2e%s\n", tname, nmse, nmse > 5e-4 ? "  (WARNING: above 5e-4)" : "");
    }
    w0.clear();
    w0.shrink_to_fit();

    // warmup -----------------------------------------------------------------
    double t_end = now_s() + o.warmup;
    while (now_s() < t_end) compute();

    // measure ----------------------------------------------------------------
    const std::string phase = std::string("run:") + tname + ":" + std::to_string(pass);
    smp.phase(phase);
    energy_meter em(nv);
    int64_t      graphs = 0;
    em.start();
    t_end = now_s() + o.seconds;
    do {
        compute();
        graphs++;
    } while (now_s() < t_end);
    auto [secs, joules] = em.stop();
    smp.phase("between");

    // summarize --------------------------------------------------------------
    res = {};
    res.pass    = pass;
    res.index   = index;
    res.type    = tname;
    res.bpw     = (double) bytes / weights;
    res.weights = weights;
    res.bytes   = bytes;
    res.reps    = reps;
    res.passes  = graphs * reps;
    res.seconds = secs;
    res.joules  = joules;
    res.nmse    = nmse;
    auto samples = smp.get(phase, std::min(2.0, o.seconds / 10));
    std::vector<double> pw;
    res.sm_min = res.mem_min = ~0u;
    for (auto & s : samples) {
        pw.push_back(s.power_mw / 1000.0);
        res.sm_min   = std::min(res.sm_min, s.sm_mhz);
        res.sm_max   = std::max(res.sm_max, s.sm_mhz);
        res.mem_min  = std::min(res.mem_min, s.mem_mhz);
        res.mem_max  = std::max(res.mem_max, s.mem_mhz);
        res.temp_max = std::max(res.temp_max, s.temp_c);
        res.reasons |= s.reasons;
    }
    if (samples.empty()) res.sm_min = res.mem_min = 0;
    res.p_med = median(pw);

    const double us   = secs / res.passes * 1e6;
    const double gbps = (double) bytes * res.passes / secs / 1e9;
    printf("[%s] %lld passes in %.2f s: %.2f us/pass, %.1f GB/s", tname, (long long) res.passes, secs, us, gbps);
    if (!std::isnan(joules)) {
        printf(", %.1f W avg, %.2f pJ/weight", joules / secs, joules / res.passes / weights * 1e12);
    }
    printf("\n");

    ggml_free(ctx_g);
    ggml_backend_buffer_free(buf_w);
    ggml_backend_buffer_free(buf_io);
    ggml_free(ctx_w);
    ggml_free(ctx_io);
    return true;
}

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    options o = parse_args(argc, argv);

    ggml_backend_load_all();

    ggml_backend_dev_t dev = nullptr;
    if (!o.device.empty()) {
        dev = ggml_backend_dev_by_name(o.device.c_str());
        if (!dev) { fprintf(stderr, "device %s not found\n", o.device.c_str()); return 1; }
    } else {
        for (size_t i = 0; i < ggml_backend_dev_count() && !dev; i++) {
            if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) dev = ggml_backend_dev_get(i);
        }
        if (!dev) {
            fprintf(stderr, "no GPU device found, using CPU\n");
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { fprintf(stderr, "failed to init %s\n", ggml_backend_dev_name(dev)); return 1; }

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    const bool is_gpu = props.type == GGML_BACKEND_DEVICE_TYPE_GPU;

    nvml_dev nv;
    if (is_gpu) nv.open(props.device_id);

    printf("device : %s (%s), pci %s\n", props.name, props.description, props.device_id ? props.device_id : "?");
    printf("nvml   : %s%s%s, power limit %.0f W\n", nv.ok() ? nv.name.c_str() : "unavailable", nv.ok() ? " via " : "",
           nv.how.c_str(), nv.power_limit_mw / 1000.0);
    printf("shape  : %d x (%lld x %lld), %s, n = 1\n", o.nmat, (long long) o.m, (long long) o.k,
           o.batched ? "batched 3-D tensor" : "separate tensors");

    sampler smp(nv, o.interval);

    const std::string summary_path = o.out + "_summary.csv";
    FILE *            fs           = fopen(summary_path.c_str(), "w");
    if (!fs) { perror(summary_path.c_str()); return 1; }
    fprintf(fs, "pass,index,type,device,layout,m,k,nmat,bytes_per_weight,weights_per_pass,bytes_per_pass,reps_per_graph,"
                "passes,seconds,us_per_pass,gb_per_s,energy_j,p_avg_w,p_med_w,p_idle_w,"
                "pj_per_weight_total,pj_per_weight_active,pj_per_byte_total,pj_per_byte_active,"
                "sm_mhz_min,sm_mhz_max,mem_mhz_min,mem_mhz_max,temp_max_c,clock_reasons,nmse\n");

    double p_idle = measure_idle(nv, smp, "idle:start", o.idle);

    std::vector<result> results;
    std::mt19937        rng(o.seed);
    for (int pass = 0; pass < o.repeat; pass++) {
        std::vector<ggml_type> order = o.types;
        if (pass > 0) {
            if (o.seed == 0) std::reverse(order.begin(), order.end());  // deterministic alternative order
            else             std::shuffle(order.begin(), order.end(), rng);
        }
        printf("\n== pass %d/%d:", pass + 1, o.repeat);
        for (auto t : order) printf(" %s", ggml_type_name(t));
        printf("\n");

        for (size_t i = 0; i < order.size(); i++) {
            if (o.cooldown > 0 && !(pass == 0 && i == 0)) {
                smp.phase("cooldown");
                std::this_thread::sleep_for(std::chrono::duration<double>(o.cooldown));
            }
            result r;
            if (!run_one(o, backend, nv, smp, order[i], pass, (int) i, r)) continue;
            results.push_back(r);

            // write rows as we go so that a killed job keeps its results;
            // the idle power is the start measurement here (the analysis may use the mean of start and end)
            const double pavg = r.joules / r.seconds;
            const double et   = r.joules / r.passes / r.weights * 1e12;
            const double ea   = (r.joules - p_idle * r.seconds) / r.passes / r.weights * 1e12;
            fprintf(fs, "%d,%d,%s,%s,%s,%lld,%lld,%d,%.6f,%lld,%lld,%d,%lld,%.4f,%.4f,%.2f,%.3f,%.2f,%.2f,%.2f,"
                        "%.4f,%.4f,%.4f,%.4f,%u,%u,%u,%u,%u,0x%llx,%.3e\n",
                    r.pass, r.index, r.type.c_str(), props.name, o.batched ? "batched" : "separate", (long long) o.m,
                    (long long) o.k, o.nmat, r.bpw, (long long) r.weights, (long long) r.bytes, r.reps,
                    (long long) r.passes, r.seconds, r.seconds / r.passes * 1e6,
                    (double) r.bytes * r.passes / r.seconds / 1e9, r.joules, pavg, r.p_med, p_idle, et, ea,
                    et / r.bpw, ea / r.bpw, r.sm_min, r.sm_max, r.mem_min, r.mem_max, r.temp_max, r.reasons, r.nmse);
            fflush(fs);
        }
    }

    double p_idle_end = measure_idle(nv, smp, "idle:end", o.idle);
    smp.stop();
    fclose(fs);
    if (nv.ok()) smp.write(o.out + "_samples.csv");

    // console table ----------------------------------------------------------
    printf("\nidle power: start %.1f W, end %.1f W\n", p_idle, p_idle_end);
    printf("%-4s %-8s %8s %10s %9s %8s %8s %10s %10s %10s\n", "pass", "type", "B/w", "us/pass", "GB/s", "P_avg", "P_med",
           "pJ/w tot", "pJ/w act", "pJ/B act");
    for (auto & r : results) {
        const double et = r.joules / r.passes / r.weights * 1e12;
        const double ea = (r.joules - p_idle * r.seconds) / r.passes / r.weights * 1e12;
        printf("%-4d %-8s %8.4f %10.2f %9.1f %8.1f %8.1f %10.3f %10.3f %10.3f\n", r.pass, r.type.c_str(), r.bpw,
               r.seconds / r.passes * 1e6, (double) r.bytes * r.passes / r.seconds / 1e9, r.joules / r.seconds, r.p_med,
               et, ea, ea / r.bpw);
    }
    printf("\nwrote %s%s\n", summary_path.c_str(), nv.ok() ? (", " + o.out + "_samples.csv").c_str() : "");

    ggml_backend_free(backend);
    return 0;
}
