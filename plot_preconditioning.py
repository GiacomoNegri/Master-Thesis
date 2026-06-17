import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_preconditioning(sigma_min: float, sigma_max: float,
                         sigma_data: float, out_dir: str) -> None:
    """
    c_skip(σ), c_out(σ), c_in(σ) vs σ on a log scale.
    Pure math — no model evaluation needed.
    """
    os.makedirs(out_dir, exist_ok=True)
    sigmas = np.logspace(np.log10(sigma_min * 0.5), np.log10(sigma_max * 2.0), 500)

    c_skip = sigma_data ** 2 / (sigmas ** 2 + sigma_data ** 2)
    c_out  = sigmas * sigma_data / np.sqrt(sigmas ** 2 + sigma_data ** 2)
    c_in   = 1.0 / np.sqrt(sigmas ** 2 + sigma_data ** 2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sigmas, c_skip, lw=2.2, label=r"$c_\mathrm{skip}(\sigma)$")
    ax.plot(sigmas, c_out,  lw=2.2, label=r"$c_\mathrm{out}(\sigma)$")
    ax.plot(sigmas, c_in,   lw=2.2, label=r"$c_\mathrm{in}(\sigma)$")
    ax.axvline(sigma_data, color="dimgray", ls="--", lw=1.4,
               label=rf"$\sigma_{{\mathrm{{data}}}} = {sigma_data}$  (crossover)")
    ax.axhline(0.5, color="lightgray", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlim(sigmas[0], sigmas[-1])
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel(r"Noise level $\sigma$")
    ax.set_ylabel("Coefficient value")
    ax.set_title("EDM preconditioning coefficients")
    ax.legend(loc="center left")
    _save(fig, os.path.join(out_dir, "preconditioning.png"))


if __name__ == "__main__":
    plot_preconditioning(
        sigma_min=0.002,
        sigma_max=8.0,
        sigma_data=1.0,
        out_dir="figures/edm",
    )
