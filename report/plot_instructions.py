"""Figure 2 of the status report: instructions per byte vs. active energy per byte.

Instruction counts: results/nvbit_20261112_182338 (llama.cpp mul_mat_vec_q / mul_mat_vec_f, thread-level, one
launch of a 17408x5120 matrix). Energy: results/h100_all_formats_20261112_173329_summary.csv.
"""
import matplotlib.pyplot as plt
import numpy as np

# format, group, instructions per weight, bytes per weight, active pJ/byte
DATA = [
    ("F16", "Uncompressed", 10.09, 2.0, 190),
    ("Q8_0", "Single scale", 6.34, 1.0625, 222),
    ("Q4_1", "Single scale", 4.31, 0.625, 229),
    ("Q4_0", "Single scale", 4.83, 0.5625, 260),
    ("IQ4_NL", "Lookup table", 6.51, 0.5625, 300),
    ("IQ4_XS", "Lookup table", 6.17, 0.53125, 293),
    ("Q6_K", "Two-level", 7.49, 0.8203125, 281),
    ("Q5_K", "Two-level", 8.01, 0.6875, 309),
    ("Q4_K", "Two-level", 7.16, 0.5625, 315),
]
COLORS = {
    "Uncompressed": "#8c8c8c",
    "Single scale": "#4c72b0",
    "Lookup table": "#e8a33d",
    "Two-level": "#d65f5f",
}
OFFSETS = {"IQ4_NL": (-34, -3), "IQ4_XS": (5, -8), "Q5_K": (-14, 6), "Q4_K": (-4, -12), "Q6_K": (-28, 2)}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                     "font.size": 8})
x = np.array([d[2] / d[3] for d in DATA])
y = np.array([d[4] for d in DATA])
slope, intercept = np.polyfit(x, y, 1)

fig, ax = plt.subplots(figsize=(3.4, 2.3))
xs = np.array([4, 13.5])
ax.plot(xs, intercept + slope * xs, color="black", ls="--", lw=0.8, zorder=1)
for (name, group, *_), xi, yi in zip(DATA, x, y):
    ax.scatter(xi, yi, color=COLORS[group], s=22, zorder=2)
    ax.annotate(name, (xi, yi), textcoords="offset points", xytext=OFFSETS.get(name, (4, -9)), fontsize=7)
ax.set_xlabel("Instructions executed per byte of weights")
ax.set_ylabel("Energy per byte (pJ)")
ax.set_xlim(4, 13.5)
ax.set_ylim(170, 335)
ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=4) for c in COLORS.values()]
ax.legend(handles, COLORS.keys(), loc="lower right", frameon=False, handletextpad=0.2, fontsize=7)
fig.tight_layout()
fig.savefig("instructions.pdf")
print(f"fit: {intercept:.0f} + {slope:.1f} * x, r = {np.corrcoef(x, y)[0, 1]:.3f}")
