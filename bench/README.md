# matvec_energy

Standalone benchmark for the energy measurement in `NOTES.md` ("Energy
Measurements: per-format matvec benchmark"). It replaces the patched
`test-backend-ops`. llama.cpp is not modified; the benchmark links against an
installed copy of ggml.

## What it measures

For each weight format (default `f16 q8_0 q4_0 q4_1 q4_K q5_K q6_K iq4_xs iq4_nl`):

1. **Weights.** It creates `--nmat` (default 4) matrices of 17408 × 5120. The
   values are random in [-1, 1] and are quantized on the host with
   `ggml_quantize_chunk`, as `test-backend-ops` does. Four matrices of Q4_K are
   about 200 MB, larger than the 50 MB L2, so the weights stream from DRAM.
2. **Graph.** It builds one ggml graph that computes `y_i = W_i · x` (`MUL_MAT`,
   n = 1, the decode case) for every matrix. The graph repeats this `reps`
   times, so one launch moves about 8 GB of weights and lasts tens of ms. A
   *pass* is one product over all matrices.
3. **Check.** It compares the output of matrix 0 against a host reference
   (dequantized weights, double accumulation) and reports the NMSE. Expect
   about 1e-5 for every format, because ggml quantizes `x` to Q8_1 on the GPU.
   A value above 5e-4 is flagged.
4. **Warmup.** It runs the graph for `--warmup` s without measuring.
5. **Measurement.** It runs the graph for `--seconds` s (default 20) and reads
   the NVML cumulative energy counter
   (`nvmlDeviceGetTotalEnergyConsumption`, mJ) before and after.
6. **Idle power.** It measures idle power for `--idle` s at the start and at
   the end, while the process holds its CUDA context.

A background thread samples power, SM/memory clocks, temperature and
clock-event (throttle) reasons every 20 ms. Each sample is tagged with the
current phase.

NVML is loaded with `dlopen`, so the benchmark builds without CUDA headers. The
GPU is matched to its ggml device by PCI bus id. Without NVML (e.g. on a CPU
node) it still runs and reports time and bandwidth only.

Layouts:
- default: `nmat` separate 2-D weight tensors sharing one `x`, as in a real
  decode step (gate/up).
- `--batched`: one 3-D tensor and one `MUL_MAT`, as in
  `test_mul_mat(..., bs={4,1})` from `NOTES.md`.

Repetitions: `-r 3` runs three passes over the format list. Pass 1 uses the
given order. Later passes are shuffled with `--seed N`, or reversed if no seed
is given. This exposes thermal and ordering effects.

## Outputs

`PREFIX_summary.csv`: one row per (pass, format). Columns:

| column | meaning |
|---|---|
| `bytes_per_weight` | actual `ggml_nbytes / weights` (e.g. Q4_K 0.5625) |
| `weights_per_pass`, `bytes_per_pass` | W = nmat·m·k and its size in bytes |
| `passes`, `seconds`, `us_per_pass` | timing of the measured window |
| `gb_per_s` | bytes_per_pass · passes / seconds (weights only) |
| `energy_j`, `p_avg_w` | NVML energy over the window and energy / time |
| `p_med_w` | median sampled power (first and last 2 s dropped) |
| `p_idle_w` | idle power measured at start |
| `pj_per_weight_total` | energy_j / (passes · W) · 1e12 |
| `pj_per_weight_active` | (energy_j − p_idle·seconds) / (passes · W) · 1e12 |
| `pj_per_byte_*` | the two above divided by bytes_per_weight |
| `sm_mhz_*`, `mem_mhz_*`, `temp_max_c`, `clock_reasons` | check for throttling (reasons OR-ed over the window) |
| `nmse` | correctness check of matrix 0 |

`PREFIX_samples.csv` is the full power trace: `t_s, phase, power_w, energy_j,
sm_mhz, mem_mhz, temp_c, clock_reasons`. The phases are `idle:start`, `setup`,
`warmup`, `run:<type>:<pass>`, `cooldown` and `idle:end`.

The idle power at the end is printed on the console. If it differs much from
the start value, recompute the active energy from the CSV with their mean.

## Build

### 1. llama.cpp (outside this repository, with CUDA, on a GPU cluster)

Only ggml is needed. The commit `a4d4f2fd4` matches `initial_benchmark.txt`.

```bash
module load gcc cmake cuda
git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_SRC"
cd "$LLAMA_SRC" && git checkout a4d4f2fd4
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DGGML_NATIVE=OFF \
      -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
      -DLLAMA_OPENSSL=OFF -DCMAKE_INSTALL_PREFIX="$LLAMA_PREFIX"
cmake --build build-cuda -j 16
cmake --install build-cuda
```

`CMAKE_CUDA_ARCHITECTURES=90` builds only for the H100 (sm_90), which keeps the
compile time short.

### 2. Benchmark (from this repository)

```bash
cmake -S bench -B build -DCMAKE_PREFIX_PATH="$LLAMA_PREFIX"
cmake --build build -j
```

The binary is `build/matvec_energy`. Its rpath points to `$LLAMA_PREFIX/lib`,
so no `LD_LIBRARY_PATH` is needed. `build/` is ignored by git.

## Run (inside a GPU allocation)

```bash
nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > results/gpu_state.txt
./build/matvec_energy -r 3 --seed 1 -o results/h100_separate
./build/matvec_energy -r 1 --batched -o results/h100_batched   # optional, NOTES.md layout
```

Duration: 2 × 10 s idle, plus, for each format and pass, about 25 s of warmup
and measurement, about 5 s of quantization and a 5 s cooldown. With 9 formats
and 3 passes that is about 17 min. Use `-s 10 -r 1` for a quick look.

Other shapes (see the NOTES next steps): `-m 5120 -k 17408`, `-m 10240 -k 5120`,
etc. A different set of formats: `-t q4_0,q4_K`. Full help: `--help`.

## Reading the results

- `gb_per_s` for F16 should be close to the H100 DRAM peak. F16 is the pure
  streaming reference.
- `pj_per_byte_active` about equal across formats means decode is
  DRAM/static-bound and software dequant is cheap.
- `pj_per_byte_active` higher for Q4_K / IQ4 than Q4_0 at equal bytes means
  the excess is the software dequant cost that a hardware decoder could remove.
- Lower `gb_per_s` for complex formats means dequant also costs time, and
  therefore static energy (visible in `pj_per_weight_total`).
- `sm_mhz_min` should equal `sm_mhz_max`, and `clock_reasons` should show no
  power or thermal bits, otherwise the run was throttled.
- The three passes should agree within a few %.

To confirm which kernel runs (`mul_mat_vec_q` / `mul_mat_vec_f`), wrap one
format with nsys:
`nsys profile -o mv ./build/matvec_energy -t q4_K -s 2 -r 1 --idle 0`.

## Caveats

- Random weights. Switching activity, and so power, may differ from real
  Qwen tensors.
- The Q8_1 quantization of `x` is a separate small kernel and is included in
  the time and energy.
- NVML reports energy for the whole board, including HBM.
- On Hopper, `power.draw` (the sampled power) is averaged by the driver. The
  energy counter is the primary measurement.
