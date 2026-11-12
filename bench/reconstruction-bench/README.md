# Controlled reconstruction experiments

This suite tests Q4_0, Q4_K, and IQ4_XS with 256 threads per CTA throughout.
It is a diagnostic implementation, not a replacement for llama.cpp's optimized
integer-dot kernels. Use `profile_existing.sh` for per-PC counts on the original
llama.cpp matvec benchmark as a separate experiment.

## Build and run on Kuma

Requires the gcc/cmake/cuda modules, CUDA 12.4+, installed ggml from the same
llama.cpp revision as before, and Python 3. CMake 3.29+ is needed for NVBit.
The default ggml prefix is `/scratch/mindreia/qwentization/ggml-nvbit-install`.
Override with `GGML_PREFIX=/path/to/install`.

Run `bash run.sh` for CPU-reference validation and uninstrumented timing.
Only after it succeeds, run `bash profile_pc.sh` for per-PC counts on the new
kernels. Independently, `bash profile_existing.sh` collects per-PC counts on
the existing, graph-disabled `build-nvbit/matvec_energy` binary.

## Experiments

1. **write_fp16:** reconstruct every weight and store it to FP16 output. This
   explicitly includes conversion and output memory traffic; it is not just
   unpacking time.
2. **checksum:** reconstruct each weight, accumulate it into a checksum, and
   store one sum per CTA. This avoids a full expanded output buffer, but adds
   reduction work and dependencies. Values feed externally visible outputs;
   inspect generated assembly/per-PC counts to confirm the chosen compiler's
   treatment rather than assuming every scalar intermediate survives.
3. **read_bytes_control:** load every packed input byte and accumulate a checksum.
   Same input allocation and contents as that format's decoder, but different
   load instructions and reduction work. Never subtract this time and label
   the difference exact reconstruction time.
4. **matvec:** reconstruct values in registers, multiply by FP32 activations,
   and reduce with a shared implementation. One CTA per row, 256 threads per
   CTA for every format and ablation. This uses FP32 scalar arithmetic rather
   than llama.cpp's packed integer dot path, isolating this implementation's
   sensitivity to the decoder rather than predicting native hardware speedup.

Checksum and matvec have these variants:

- **full:** correct interpretation of weights and metadata.
- **no_metadata_DIAGNOSTIC:** keep code extraction/mapping, replace scales with
  one and offsets with zero. Removes metadata loads and arithmetic.
- **no_mapping_DIAGNOSTIC:** IQ replaces nonlinear lookup with a centered
  nibble; Q4_0 omits centering. No Q4_K version because its codes are already
  unsigned linear codes. Metadata remains in use.
- **codes_only_DIAGNOSTIC:** omit metadata and mapping. These variants produce
  deliberately different numerical answers. Do not treat them as valid models.

IQ lookup uses a runtime-initialized constant-memory table. This makes its
mapping explicit, but differs from llama.cpp's PRMT-based implementation.
Divergent constant-table accesses may affect performance. The native per-PC
experiment is necessary to understand the production kernel.

The ablations share source geometry and reduction logic. They do NOT guarantee
identical executed loads, register allocations, dependencies, occupancy, or
cache behavior. Differences are sensitivity measurements, not exact attribution.
An architecture simulation remains necessary for a proposed hardware speedup.

## Shapes, measurement, and correctness

- Small: 256 × 1024, one matrix; about 136–144 KiB packed input and 512 KiB output.
  It is intended to fit L2; no counter measurement proves a particular hit rate.
- Streaming: four 17408 × 5120 matrices; about 190–200 MiB packed input.
  The working set exceeds H100 L2. That alone does not establish identical HBM
  transactions across variants.
- These are equal-weight comparisons. IQ4_XS has fewer bytes/weight than Q4_0
  and Q4_K; Q4_0 and Q4_K both have 0.5625 bytes/weight.
- Each row's original random values are identical across formats and pass order.
  The runner randomizes format order across three outer passes with a fixed seed.
  Variant order within a process is fixed; expand/randomize it for final energy
  or thermally sensitive studies.
- Normal runs gate timing on full-decoder and full-matvec checks against ggml's
  CPU `to_float` reference, sampling first/last rows and a middle row.
- Graph batches of 32 launches amortize host overhead for timing. CUDA event
  times still include launch scheduling and possible idle gaps between graphs.
- Reports timing, registers per thread, and local bytes per thread. Local bytes
  include compiler local allocation and are not a direct count of spill accesses.
  Retain `build.log` for compiler register/spill reports.
- No energy measurement is implemented here. Use uninstrumented, sustained
  runs with board energy counters in a later extension. Never use NVBit timing
  or energy values as physical performance measurements.
- The executable supports `--m`, `--k` (multiple of 256), `--nmat`, `--format`,
  `--seconds`, `--repeat`, `--seed`, `--out`, `--validate-only`, and `--profile`.
  Additional matrix shapes must come from actual model metadata before claiming
  model-wide representativeness. The streaming shape matches the earlier bench.

## NVBit per-address tool

`prepare_pc_tool.py` copies the existing official opcode_hist tool into
`tools/pc_hist`, preserving the original. It replaces each histogram key with
function name, original byte offset, and opcode. That distinguishes two uses of
the same opcode. It does not automatically infer semantic roles.

`parse_pc.py` verifies each launch's histogram sum and writes
`instruction_addresses.csv`. Match its decimal PCs against the hexadecimal
offsets in the log's static disassembly. Keep ambiguous PCs unclassified.
Neither script assigns all shifts to reconstruction or all IMADs to addressing.

New-kernel profiles launch each case exactly once, without graphs, validation,
calibration, or warmups inside the instrumented process. CPU-reference checks
run separately first. Original-kernel profiles preserve the original workload:
eight matvec launches and eight Q8_1 activation launches per quantized format.

Decode definitions checked against:
https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
and `ggml-common.h`. The installed ggml CPU check is the revision-specific
compatibility gate. Run it before drawing conclusions.

## Validation performed before delivery

Host-compiled decoder tests exercise all 256 positions of randomized blocks,
including every packed K scale/min field and IQ code, against independent
reference loops. Shell and Python syntax and per-PC parser integrity are tested.
CUDA compilation, GPU reference checks, timing, and profiler integration must
still be run on Kuma; the preparation machine has no CUDA toolkit or GPU.
