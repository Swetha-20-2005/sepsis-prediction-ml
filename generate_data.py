"""Generate a clearly synthetic dataset for a machine-learning demo.
This dataset is not real patient data and is not suitable for clinical use.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 1200
OUT = Path("data/sepsis_synthetic.csv")

def main():
    rng = np.random.default_rng(SEED)
    heart_rate = rng.normal(88, 16, N_ROWS).clip(45, 170)
    temperature = rng.normal(37.1, 0.8, N_ROWS).clip(34.5, 41.5)
    respiratory_rate = rng.normal(19, 5, N_ROWS).clip(6, 45)
    lactate = rng.lognormal(mean=0.35, sigma=0.45, size=N_ROWS).clip(0.4, 8.0)
    wbc = rng.normal(10, 4, N_ROWS).clip(1, 35)

    # Synthetic risk score used only to create demonstration labels.
    score = (
        0.025 * (heart_rate - 90)
        + 0.8 * np.maximum(temperature - 38.0, 0)
        + 0.035 * (respiratory_rate - 20)
        + 0.65 * (lactate - 2.0)
        + 0.045 * (wbc - 11)
        + rng.normal(0, 0.9, N_ROWS)
    )
    probability = 1 / (1 + np.exp(-score))
    sepsis_label = rng.binomial(1, probability)

    df = pd.DataFrame({
        "HR": heart_rate.round(1),
        "Temp": temperature.round(2),
        "Resp": respiratory_rate.round(1),
        "Lactate": lactate.round(2),
        "WBC": wbc.round(2),
        "SepsisLabel": sepsis_label.astype(int),
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Created {OUT} with {len(df)} synthetic rows.")
    print("Label distribution:")
    print(df["SepsisLabel"].value_counts().sort_index().to_string())

if __name__ == "__main__":
    main()
