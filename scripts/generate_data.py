import sys, os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
from physics import *

def generate(n=1000, seed=42, drift_from=None):
    r = np.random.default_rng(seed); sp = 40.0; T = None; rows = []
    t0 = pd.Timestamp("2026-01-01 08:00:00")
    for i in range(n):
        sp = float(np.clip(sp + r.normal(0, 3), 10, 70))
        rpm = rpm_f(sp) + r.normal(0, 60)
        tt = temp_f(rpm, sp); T = tt if T is None else T + 0.15 * (tt - T) + r.normal(0, 0.3)
        row = dict(timestamp=t0 + pd.Timedelta(seconds=i), rpm=rpm, speed=sp, temperature=T + r.normal(0, .3),
                   battery_voltage=batt_f(rpm) + r.normal(0, .05), fuel_consumption=fuel_f(rpm, sp) + r.normal(0, .03), label="NORMAL")
        if drift_from is not None and i >= drift_from:
            k = (i - drift_from) / (n - drift_from)
            row["temperature"] += 16 * k; row["rpm"] += 450 * k
            row["battery_voltage"] += -0.8 * k + r.normal(0, .15 * k); row["fuel_consumption"] += 0.4 * k
            row["label"] = "EARLY DRIFT" if k < .35 else "PRE-FAULT" if k < .7 else "CRITICAL"
        rows.append(row)
    return pd.DataFrame(rows).round(3)

if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    df = generate(); df.drop(columns="label").to_csv(os.path.join(ROOT, "data/raider_healthy_data.csv"), index=False)
    print("Saved data/raider_healthy_data.csv", len(df), "rows")
