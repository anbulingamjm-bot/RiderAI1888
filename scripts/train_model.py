import os, sys, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv = os.path.join(ROOT, "data/raider_healthy_data.csv")
if not os.path.exists(csv): sys.exit("Missing healthy CSV. Run: python scripts/generate_data.py")
df = pd.read_csv(csv).dropna(); print("Loaded", len(df), "healthy samples")
tr, te = train_test_split(df, test_size=0.2, random_state=1)
# (rpm, speed) -> expected temperature, battery, fuel.  speed -> expected rpm.
m1 = RandomForestRegressor(200, min_samples_leaf=5, random_state=1).fit(tr[["rpm", "speed"]], tr[["temperature", "battery_voltage", "fuel_consumption"]])
m2 = RandomForestRegressor(200, min_samples_leaf=5, random_state=1).fit(tr[["speed"]], tr["rpm"])
r1 = te[["temperature", "battery_voltage", "fuel_consumption"]].values - m1.predict(te[["rpm", "speed"]])
r2 = te["rpm"].values - m2.predict(te[["speed"]])
std = dict(temperature=r1[:, 0].std(), battery_voltage=r1[:, 1].std(), fuel_consumption=r1[:, 2].std(), rpm=r2.std())
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
joblib.dump(dict(m1=m1, m2=m2, std=std), os.path.join(ROOT, "models/healthy_digital_twin.pkl"))
print("Residual std (healthy):", {k: round(float(v), 3) for k, v in std.items()}); print("Model saved: models/healthy_digital_twin.pkl")
