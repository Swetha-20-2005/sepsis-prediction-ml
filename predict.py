"""Run a demo prediction with the trained model. Not for medical decisions."""
import argparse
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("models/sepsis_model.joblib")
FEATURES = ["HR", "Temp", "Resp", "Lactate", "WBC"]

def main():
    parser = argparse.ArgumentParser(description="Synthetic sepsis model demo")
    parser.add_argument("--hr", type=float, required=True, help="Heart rate")
    parser.add_argument("--temp", type=float, required=True, help="Temperature")
    parser.add_argument("--resp", type=float, required=True, help="Respiratory rate")
    parser.add_argument("--lactate", type=float, required=True, help="Lactate")
    parser.add_argument("--wbc", type=float, required=True, help="White blood cell count")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python generate_data.py` then `python train.py`.")
    row = pd.DataFrame([{
        "HR": args.hr, "Temp": args.temp, "Resp": args.resp,
        "Lactate": args.lactate, "WBC": args.wbc
    }], columns=FEATURES)
    model = joblib.load(MODEL_PATH)
    pred = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    print(f"Demo model output: class={pred}, estimated model score={probability:.1%}")
    print("This is not a diagnosis or clinically validated risk estimate.")

if __name__ == "__main__":
    main()
