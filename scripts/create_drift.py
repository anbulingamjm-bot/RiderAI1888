import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_data import generate, ROOT
df = generate(1000, seed=7, drift_from=700)  # samples 1-700 healthy, 701-1000 gradual drift
df.to_csv(os.path.join(ROOT, "data/raider_drift_data.csv"), index=False)
print("Saved data/raider_drift_data.csv"); print(df.label.value_counts())
