 Option A: RTL datapath study (your idea), my top pick

 Build the dot-product unit inside a tensor core in SystemVerilog, synthesize it on an open PDK, and compare area and energy across variants using
 real Qwen weights and activations as stimulus.

 Variants to build:

 ┌─────┬──────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────┐
 │ #   │ Unit                                                                                         │ Models                                      │
 ├─────┼──────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ A0  │ int8×int8 DP-32 → int32, plus one fp32 scale multiply-add per block                          │ Today's path after software unpack (MMQ /   │
 │     │                                                                                              │ dp4a + scale)                               │
 ├─────┼──────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ A1  │ A0 + Q4_0 front end (nibble unpack, −8)                                                      │ Simplest GGUF support                       │
 ├─────┼──────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ A2  │ A0 + K-quant front end: decode packed 6-bit scales/mins, sub-scale s_j, min correction dmin· │ The main contribution                       │
 │     │ m_j·Σx, fp16 super-scales                                                                    │                                             │
 ├─────┼──────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ A3  │ A2 + 16-entry LUT (IQ4_NL/XS)                                                                │ Stretch goal                                │
 ├─────┼──────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ A2′ │ Avant-Garde-style flattening: premultiply q·s_j into ~10-bit elements before the multipliers │ Compare against A2                          │
 └─────┴──────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────┘

 A2 vs A2′ is a clean architecture question. Avant-Garde applies the sub-scale to every element: 32 multiplies, with wider multipliers. You can
 instead apply s_j to the int32 partial sum after each 32-element dot product, which costs 1 multiply per 32 MACs and keeps 4×8-bit multipliers. That
 should be noticeably cheaper in area and energy, and it's a concrete, testable finding.

 Toolchain (all open):
 - Yosys + OpenROAD / LibreLane, with OpenSTA for timing and power.
 - PDKs:
     - IHP SG13G2: you already have ~/work/IHP-Open-PDK. It's real silicon, 130 nm.
     - ASAP7: a predictive 7 nm process, closer to modern GPUs.
     - FreePDK45 / Nangate45: the same node as Avant-Garde, so results are directly comparable.
 - Power from switching activity: simulate with Verilator or Icarus, dump VCD/SAIF, then run OpenSTA power. Drive it with real Q4_K blocks extracted
   with the gguf Python package and real activations dumped from llama.cpp. Real data matters because switching activity depends on the values.
 - Correctness: bit-exact comparison against ggml's reference vec_dot_q4_K_q8_1.

 Weakness: it's component-level only. You need a system model to turn "pJ per MAC" into "J per token" (option B).

 Option B: Measured GPU energy + analytical system model (pairs with A)

 No simulator; real measurements plus arithmetic.

 1. Measure J/token per format on the H100 with the NVML energy counter. Possibly also on your Mac (Apple Silicon) with powermetrics.
 2. Measure the cost of software dequant directly:
     - Count instructions per kernel with NVBit.
     - Run energy microbenchmarks with NVML: a loop of only dp4a, only LOP3/shift (unpack), only FMA, only DRAM streaming. That gives pJ per
       instruction class and pJ per DRAM byte on your H100. This is the AccelWattch method, done by hand.
 3. Model: E_token = bytes_DRAM·e_DRAM + Σ instr_class·e_class + P_static·t.
 4. Plug in option A: replace the software dequant instructions with the RTL unit's energy (scaled to the GPU's process node, with that caveat
    stated) and estimate the energy saving per token.

 This replaces Accel-Sim's timing with measurement and RTL, which are easier to defend than AccelWattch on Hopper.

Reserach Question:
- can we reduce decode energy usage by hardware support for quantization?

First we measure gpu energy:
for a multiple quanzation gguf quanzation formats.
benchmark: 17408×5120.  
(1) measure J/token
-> single-kernel benchmarks (4) → pJ/weight, GB/s, instructions/weight per format [clean, isolated]
 real mixed model (1)+(3)       → per-format energy breakdown of UD-Q4_K_M        [realistic workload]
                                    = Σ_format bytes_format × cost_format  (check against measured total)
This gets us a baseline J/token for a full model, and also the breakdown by operation. 

We want to run this experiment:
 ┌──────────────────────┬──────────────┬─────────────────────────────────────┐
 │ Format               │ Bytes/weight │ Shows                               │
 ├──────────────────────┼──────────────┼─────────────────────────────────────┤
 │ F16                  │ 2.0          │ Pure streaming baseline, no dequant │
 ├──────────────────────┼──────────────┼─────────────────────────────────────┤
 │ Q8_0                 │ 1.06         │ Minimal dequant (one scale)         │
 ├──────────────────────┼──────────────┼─────────────────────────────────────┤
 │ Q4_0                 │ 0.56         │ Simple 4-bit                        │
 ├──────────────────────┼──────────────┼─────────────────────────────────────┤
 │ Q4_K                 │ 0.56         │ Same bytes as Q4_0, complex dequant │
 ├──────────────────────┼──────────────┼─────────────────────────────────────┤
 │ Q5_K / Q6_K / IQ4_XS │ 0.53–0.82    │ More format complexity              │
 └──────────────────────┴──────────────┴─────────────────────────────────────┘

- (a) Scaling equations: convert your PDK numbers to ~5 nm using Stillmaker & Baas, "Scaling equations for the accurate prediction of CMOS device
performance from 180 nm to 7 nm" (Integration, 2017), the standard citation for this. Synthesize in ASAP7 as well; it's closest to 4N, so it needs
the least extrapolation.

---

## Energy Measurements: per-format matvec benchmark (H100)

**Goal:** measure energy per weight (pJ/weight) and achieved bandwidth (GB/s) of the
decode matvec for each GGUF format, at the same shape. If pJ/byte rises with format
complexity at equal bytes (Q4_K vs Q4_0), the difference is the software dequant cost.

**Shape:** 17408 × 5120 (MLP gate/up; 64×/token, ~35% of decode time, ~63% of weights incl. down).
Use a batch of 4 distinct matrices (`bs={4,1}`) so the working set (~200 MB at 4.5 bpw)
exceeds the 50 MB L2 and the run streams from DRAM.

**Formats:** F16, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL.

> **Implemented** as a standalone benchmark in [`bench/`](bench/README.md) (`matvec_energy`):
> no llama.cpp patch is needed, NVML energy is read in-process, and the idle baseline,
> repetitions and results table are produced by the tool. Steps 1–6 below are kept
> as the original manual procedure.

### Step 1: Add the benchmark cases to llama.cpp

In `tests/test-backend-ops.cpp`, at the top of `make_test_cases_perf()` add:

```cpp
for (ggml_type t : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
                    GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                    GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL}) {
    // m = 17408 output rows, n = 1 (decode, batch 1), k = 5120, 4 distinct matrices
    test_cases.emplace_back(new test_mul_mat(t, GGML_TYPE_F32, 17408, 1, 5120, {4, 1}, {1, 1}));
}
```

Make the run long enough for stable power readings: in `eval_perf`, change
`while (total_time_us < 1000*1000)` (1 s) to `while (total_time_us < 20*1000*1000)` (20 s).

### Step 2: Build

```bash
module load gcc cmake cuda
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build -j --target test-backend-ops
```

### Step 3: Power logger

Run in a second terminal (or in the background), logging every 20 ms:

```bash
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,pstate \
           --format=csv,noheader,nounits -lms 20 > power_<format>.csv
```

(Optional, more precise: read the NVML energy counter before/after with
`pynvml.nvmlDeviceGetTotalEnergyConsumption(h)` (mJ).)

Record the setup once: `nvidia-smi -q -d CLOCK,POWER,PERFORMANCE > gpu_state.txt`.

### Step 4: Idle baseline

Logger on, GPU idle for 30 s. Idle power `P_idle` = median of the samples.

### Step 5: Run each format (one at a time, logger running)

```bash
for t in f16 q8_0 q4_0 q4_1 q4_K q5_K q6_K iq4_xs iq4_nl; do
  nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu \
             --format=csv,noheader,nounits -lms 20 > power_$t.csv &
  LOG=$!
  sleep 5                                   # idle lead-in
  ./build/bin/test-backend-ops perf -o MUL_MAT -b CUDA0 \
      -p "type_a=$t,type_b=f32,m=17408,n=1,k=5120" | tee perf_$t.txt
  sleep 5                                   # idle tail
  kill $LOG
done
```

`perf_$t.txt` gives **us/run** and **GB/s**. `power_$t.csv` gives the power trace.
Repeat the full loop 3× and in a different order to catch thermal and ordering effects.

### Step 6: Compute

For each format:

- `P_run` = median power over the plateau (drop the first and last 2 s of the run).
- `t_run` = us/run from test-backend-ops (one run = all 4 matrices).
- `W` = weights per run = 4 · 17408 · 5120 = 356,515,840.
- `B` = bytes per run = W · bytes_per_weight
  (F16 2.0, Q8_0 1.0625, Q4_0 0.5625, Q4_1 0.625, Q4_K 0.5625, Q5_K 0.6875,
  Q6_K 0.8203, IQ4_XS 0.53125, IQ4_NL 0.5625).

```
E_total  [pJ/weight] = P_run           · t_run / W · 1e12
E_active [pJ/weight] = (P_run − P_idle) · t_run / W · 1e12
pJ/byte             = E / bytes_per_weight
BW_eff   [GB/s]     = B / t_run       (compare with test-backend-ops GB/s and the H100 peak)
```

### Step 7: Sanity checks

- BW_eff for F16 should be close to the H100 peak (it's the pure streaming reference).
- SM and memory clocks stay constant across runs, with no throttling (check the CSV).
- Results across the 3 repetitions agree within a few %.
- Check that the kernel is `mul_mat_vec_q` (or `mul_mat_vec_f` for F16) with an nsys run of one format.

### Step 8: Results table

| Format | bytes/w | us/run | GB/s | P_run (W) | pJ/weight (total) | pJ/weight (active) | pJ/byte (active) |
|--------|---------|--------|------|-----------|-------------------|--------------------|------------------|
| F16    | 2.0     |        |      |           |                   |                    |                  |
| Q8_0   | 1.0625  |        |      |           |                   |                    |                  |
| Q4_0   | 0.5625  |        |      |           |                   |                    |                  |
| Q4_1   | 0.625   |        |      |           |                   |                    |                  |
| Q4_K   | 0.5625  |        |      |           |                   |                    |                  |
| Q5_K   | 0.6875  |        |      |           |                   |                    |                  |
| Q6_K   | 0.8203  |        |      |           |                   |                    |                  |
| IQ4_XS | 0.53125 |        |      |           |                   |                    |                  |
| IQ4_NL | 0.5625  |        |      |           |                   |                    |                  |

**How to read it:**
- pJ/byte roughly the same for all formats → decode is DRAM/static-bound, software dequant is cheap.
- pJ/byte higher for complex formats at equal bytes (Q4_K > Q4_0, IQ4 > Q4_0) → the excess is the
  **software dequant cost**, which a hardware decoder could remove.
- Lower GB/s for complex formats → dequant also costs time, and therefore static energy.

### Notes / caveats

- test-backend-ops fills the weights with random data. Switching activity (and so power) can
  differ slightly from real weights; good enough for a first pass. Later, re-run with real Qwen tensors.
- The activation is quantized to Q8_1 in a separate small kernel, included in t_run (small).
- Next steps: add the other shapes (5120×17408, 10240×5120, 1024×5120, 248320×5120) and run NVBit
  instruction counts for the same kernels (instructions/weight per format).
