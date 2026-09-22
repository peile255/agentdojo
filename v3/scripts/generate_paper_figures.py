from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("ccfa_v3/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Figure 2: Layered safety observability
# ---------------------------------------------------------
labels = [
    "Source\nexposure",
    "Source\nunsafe",
    "Intermediate\npropagation",
    "Final response\nunsafe",
]

clean = np.array([0, 0, 0, 0])
compromised = np.array([88, 88, 0, 0])

x = np.arange(len(labels))
w = 0.36

fig, ax = plt.subplots(figsize=(7.2, 4.4))

b1 = ax.bar(x - w/2, clean, w, label="Clean")
b2 = ax.bar(x + w/2, compromised, w, label="Compromised")

ax.set_ylabel("Number of runs")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 100)
ax.legend(frameon=False)

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{int(h)}/88",
            (bar.get_x() + bar.get_width()/2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT / "fig2_layered_safety.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig2_layered_safety.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------
# Figure 3: Legacy vs execution-aware evaluator
# ---------------------------------------------------------
names = ["Legacy\nfinal-text", "Execution-aware\nutility-v2"]
values = [120, 77]

fig, ax = plt.subplots(figsize=(5.6, 4.4))

bars = ax.bar(names, values, width=0.55)

ax.set_ylabel("Utility passes")
ax.set_ylim(0, 140)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        value + 3,
        f"{value}/176",
        ha="center",
        va="bottom",
        fontsize=10,
    )

ax.text(
    0.5,
    0.93,
    r"Agreement = 58.5%, $\kappa=0.207$",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
)

ax.text(
    0.5,
    0.86,
    r"Exact McNemar $p=4.09\times10^{-7}$",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT / "fig3_evaluator_passes.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig3_evaluator_passes.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------
# Figure 4: Topology summaries by evaluator
# ---------------------------------------------------------
topologies = [
    "Sequential",
    "Peer debate",
    "Hierarchical",
    "Safety monitor",
]

legacy = np.array([29, 31, 38, 22])
utility_v2 = np.array([29, 16, 18, 14])

x = np.arange(len(topologies))
w = 0.36

fig, ax = plt.subplots(figsize=(7.4, 4.6))

b1 = ax.bar(x - w/2, legacy, w, label="Legacy")
b2 = ax.bar(x + w/2, utility_v2, w, label="Utility-v2")

ax.set_ylabel("Utility passes out of 44")
ax.set_xticks(x)
ax.set_xticklabels(topologies)
ax.set_ylim(0, 44)
ax.legend(frameon=False)

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{int(h)}",
            (bar.get_x() + bar.get_width()/2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT / "fig4_topology_by_evaluator.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig4_topology_by_evaluator.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------
# Figure 5: Utility-v2 heatmap
# ---------------------------------------------------------
data = np.array([
    [90.9, 63.6, 54.5, 54.5],
    [45.5, 36.4, 36.4, 27.3],
    [45.5, 45.5, 36.4, 36.4],
    [27.3, 36.4, 36.4, 27.3],
])

rows = [
    "Sequential",
    "Peer debate",
    "Hierarchical",
    "Safety monitor",
]

cols = [
    "Clean /\nstandard",
    "Clean /\ndefended",
    "Compromised /\nstandard",
    "Compromised /\ndefended",
]

fig, ax = plt.subplots(figsize=(8.2, 4.8))

im = ax.imshow(data, aspect="auto")

ax.set_xticks(np.arange(len(cols)))
ax.set_xticklabels(cols)
ax.set_yticks(np.arange(len(rows)))
ax.set_yticklabels(rows)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(
            j,
            i,
            f"{data[i, j]:.1f}%",
            ha="center",
            va="center",
        )

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Execution-aware utility (%)")

fig.tight_layout()
fig.savefig(OUT / "fig5_utility_heatmap.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig5_utility_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved figures to: {OUT.resolve()}")
