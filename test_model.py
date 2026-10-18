"""Basic automated tests for the training pipeline."""
import subprocess
import sys
import unittest
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent

class SepsisProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([sys.executable, "generate_data.py"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "train.py"], cwd=ROOT, check=True)

    def test_dataset_has_required_columns_and_binary_target(self):
        df = pd.read_csv(ROOT / "data" / "sepsis_synthetic.csv")
        self.assertTrue({"HR", "Temp", "Resp", "Lactate", "WBC", "SepsisLabel"}.issubset(df.columns))
        self.assertTrue(set(df["SepsisLabel"].unique()).issubset({0, 1}))
        self.assertGreater(len(df), 0)

    def test_saved_model_predicts_one_row(self):
        model = joblib.load(ROOT / "models" / "sepsis_model.joblib")
        row = pd.DataFrame([{"HR": 90, "Temp": 37.0, "Resp": 18, "Lactate": 1.2, "WBC": 9.0}])
        result = model.predict(row)
        self.assertEqual(len(result), 1)
        self.assertIn(int(result[0]), [0, 1])

if __name__ == "__main__":
    unittest.main(verbosity=2)
