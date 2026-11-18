"""Run TVC on synthetic daily time series without external climate data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tvc import TVC, spin_up
from tvc.synthetic import generate_ar1


def main() -> None:
    n_train = 4000
    n_validation = 4000
    scales = [90, 45, 22, 11, 5, 2]
    observations = generate_ar1(n_train, 0.75, 10)
    model_training = generate_ar1(n_train, 0.45, 20)
    model_validation = generate_ar1(n_validation, 0.45, 30)

    tvc = TVC(observations, model_training, model_validation, scales)
    corrected_components = tvc.TVC_postprocess()
    corrected_series = np.real_if_close(corrected_components.sum(axis=1))

    raw_aligned = model_validation[spin_up(scales) :]

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "tvc_synthetic_demo.png"

    n_plot = min(500, len(corrected_series))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(raw_aligned[:n_plot], label="Raw model", alpha=0.8)
    axes[0].plot(corrected_series[:n_plot], label="TVC corrected", alpha=0.8)
    axes[0].set_title("Synthetic validation series")
    axes[0].set_xlabel("Daily time step")
    axes[0].set_ylabel("Value")
    axes[0].legend()

    axes[1].scatter(raw_aligned, corrected_series, s=8, alpha=0.25)
    axes[1].set_title("Corrected versus raw validation values")
    axes[1].set_xlabel("Raw model")
    axes[1].set_ylabel("TVC corrected")
    axes[1].axline((0, 0), slope=1, linestyle="--", linewidth=1)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved demonstration figure to: {output_path}")


if __name__ == "__main__":
    main()
