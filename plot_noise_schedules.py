import math
import numpy as np
import matplotlib.pyplot as plt

# ── parameters ──────────────────────────────────────────────────────────────
SIGMA_MIN = 0.01
SIGMA_MAX = 1.0
# ────────────────────────────────────────────────────────────────────────────

t = np.linspace(0.0, 1.0, 500)

# Mirrors SigmaSchedule.sigma() in src/utils/WIP_SDE.py
linear      = SIGMA_MIN + t * (SIGMA_MAX - SIGMA_MIN)
exponential = SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** t
cosine      = SIGMA_MIN + 0.5 * (SIGMA_MAX - SIGMA_MIN) * (1.0 - np.cos(math.pi * t))

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(t, linear,      color="#1f77b4", linewidth=2.2, label="linear")
ax.plot(t, exponential, color="#ff7f0e", linewidth=2.2, label="exponential")
ax.plot(t, cosine,      color="#2ca02c", linewidth=2.2, label="cosine")

ax.set_xlabel("t", fontsize=13)
ax.set_ylabel("σ(t)", fontsize=13)

ax.legend(fontsize=11, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
plt.tight_layout()
plt.savefig("../final_picture/noise_schedules.png", dpi=150)
plt.show()
print("Saved → noise_schedules.png")
