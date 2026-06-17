"""
Plot EDM noise-level distributions for a set of (P_mean, P_std) pairs.

ln(sigma) ~ N(P_mean, P_std^2), so we plot the Gaussian density over ln(sigma).
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Edit this list to add / remove curves ────────────────────────────────────
PARAMS = [
    (-1.2, 1.2),
    (-1.4, 1.8),
]
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def plot_edm_noise(params, out_path="../final_picture/edm_noise_distribution.png"):
    means = [m for m, _ in params]
    stds  = [s for _, s in params]

    x_min = min(means) - 3.5 * max(stds)
    x_max = max(means) + 3.5 * max(stds)
    ln_sigma = np.linspace(x_min, x_max, 2000)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Match the default matplotlib color cycle (same palette as the EDM preconditioning plot)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [default_colors[i % len(default_colors)] for i in range(len(params))]

    for (P_mean, P_std), color in zip(params, colors):
        pdf = gaussian_pdf(ln_sigma, P_mean, P_std)
        ax.plot(
            ln_sigma, pdf,
            color=color,
            linewidth=2.2,
            label=rf"$P_{{mean}}={P_mean},\; P_{{std}}={P_std}$",
        )

    ax.set_xlabel(r"$\ln(\sigma)$", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_edm_noise(PARAMS)
