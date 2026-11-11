# Unfused Qwen profiling on Kuma

Requires EPFL network/VPN, working SSH access as `mindreia`, and the existing
model/build under `/scratch/mindreia/qwen-bench`. No compilation or download is
performed. Account `cs471` is explicitly selected.

From your Mac:

```bash
cd ~/Documents/qwentization/scripts
bash request-h100.sh
```

This uploads `profile-unfused.sh`, requests one H100 for one hour, and runs
prefill (`pp512`) and decode (`tg128`) profiles with operation fusion disabled.
Each phase uses three repetitions and the default warmup. When the command
finishes, the allocation is released. Results are written in a unique directory,
so earlier reports are preserved.

At completion, copy and run the printed command, for example:

```bash
bash fetch-profiles.sh unfused-YYYYMMDD-HHMMSS-JOBID
```

Reports and text logs are downloaded into `~/Documents/qwentization/results/`.
Open the `.nsys-rep` files using NVIDIA Nsight Systems on your Mac.

If you already hold a GPU allocation, upload the script and run it there with
`bash -l /scratch/mindreia/qwen-bench/scripts/profile-unfused.sh` instead of
requesting another GPU. Do not run profiling on a login node.

The script checks that the source contains the fusion-disable switch. Inspect
the resulting kernel names to verify its effect on the built executable.
Unpacking, scales, and quantization corrections remain inside matrix kernels;
this disables surrounding operation fusion, not quantized arithmetic itself.
These captures include warmup and are for operation attribution. Keep the
original fused reports for normal end-to-end performance comparisons.

Scratch storage is temporary. Download or archive results promptly into your
group's durable project directory.
