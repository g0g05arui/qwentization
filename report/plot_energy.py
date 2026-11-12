"""Figure 1 of the status report: energy per byte and processing speed per weight format.

Data: H100 matrix-vector benchmark (bench/matvec_energy.cpp), Table 2 of the report.
"""
import matplotlib.pyplot as plt

# format, group, active pJ/byte, GB/s
DATA = [
    ("F16", "Uncompressed", 190, 2191),
    ("Q8_0", "Single scale", 222, 2079),
    ("Q4_1", "Single scale", 229, 1952),
    ("Q4_0", "Single scale", 260, 1771),
    ("IQ4_NL", "Lookup table", 300, 1825),
    ("IQ4_XS", "Lookup table", 293, 1654),
    ("Q6_K", "Two-level", 281, 1860),
    ("Q5_K", "Two-level", 309, 1802),
    ("Q4_K", "Two-level", 315, 1524),
]
COLORS = {
    "Uncompressed": "#8c8c8c",
    "Single scale": "#4c72b0",
    "Lookup table": "#e8a33d",
    "Two-level": "#d65f5f",
}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "font.size": 8})
names = [d[0] for d in DATA]
colors = [COLORS[d[1]] for d in DATA]
x = range(len(DATA))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 3.3), sharex=True)

ax1.bar(x, [d[2] for d in DATA], color=colors, width=0.7)
ax1.axhline(DATA[0][2], color="black", ls="--", lw=0.8)
ax1.set_ylabel("Energy per byte (pJ)")
ax1.set_ylim(0, 350)

ax2.bar(x, [d[3] / 1000 for d in DATA], color=colors, width=0.7)
ax2.axhline(DATA[0][3] / 1000, color="black", ls="--", lw=0.8)
ax2.set_ylabel("Speed (TB/s)")
ax2.set_ylim(0, 2.5)
ax2.set_xticks(list(x))
ax2.set_xticklabels(names, rotation=35, ha="right")

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
fig.legend(handles, COLORS.keys(), loc="upper center", ncol=2, frameon=False,
           handlelength=1, columnspacing=1, bbox_to_anchor=(0.55, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=0.6)
fig.savefig("energy.pdf")
