import warnings; warnings.filterwarnings("ignore")
import os, sqlite3, io, csv, joblib, numpy as np
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from backend.physics import *

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "database", "raider_ai.db"); MODEL = os.path.join(ROOT, "models", "healthy_digital_twin.pkl")
app = Flask(__name__); CORS(app)
LIMITS = dict(rpm=(0, 10000, "RPM"), speed=(0, 120, "speed"), temperature=(20, 130, "engine temperature"),
              battery_voltage=(8, 16, "battery voltage"), fuel_consumption=(0, 15, "fuel consumption"))
W = dict(temperature=.35, rpm=.2, battery_voltage=.25, fuel_consumption=.2)
WORDS = dict(temperature="Potential overheating trend", rpm="RPM behaviour deviating from expected for this speed",
             battery_voltage="Possible charging/battery behaviour change", fuel_consumption="Fuel consumption deviating from expected")

def db():
    os.makedirs(os.path.dirname(DB), exist_ok=True); c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c
def init_db():
    c = db(); c.executescript("""
    CREATE TABLE IF NOT EXISTS sensor_data(id INTEGER PRIMARY KEY, ts TEXT, source TEXT, rpm REAL, speed REAL, temperature REAL, battery_voltage REAL, fuel_consumption REAL);
    CREATE TABLE IF NOT EXISTS health_history(id INTEGER PRIMARY KEY, ts TEXT, sensor_id INT, health REAL, drift REAL, status TEXT, exp_temperature REAL, exp_rpm REAL);
    CREATE TABLE IF NOT EXISTS anomaly_events(id INTEGER PRIMARY KEY, ts TEXT, status TEXT, primary_contributor TEXT, message TEXT);
    CREATE TABLE IF NOT EXISTS recovery_events(id INTEGER PRIMARY KEY, ts TEXT, before_health REAL, after_health REAL, result TEXT, detail TEXT);
    CREATE TABLE IF NOT EXISTS simulation_events(id INTEGER PRIMARY KEY, ts TEXT, event TEXT);""")
    c.commit(); c.close()
def now(): return datetime.now().isoformat(timespec="seconds")
def log_sim(ev):
    try: c = db(); c.execute("INSERT INTO simulation_events(ts,event) VALUES(?,?)", (now(), ev)); c.commit(); c.close()
    except Exception as e: print("DB error", e)
def log_sample(s, a, source):
    try:
        c = db(); cur = c.execute("INSERT INTO sensor_data(ts,source,rpm,speed,temperature,battery_voltage,fuel_consumption) VALUES(?,?,?,?,?,?,?)",
            (now(), source, s["rpm"], s["speed"], s["temperature"], s["battery_voltage"], s["fuel_consumption"]))
        c.execute("INSERT INTO health_history(ts,sensor_id,health,drift,status,exp_temperature,exp_rpm) VALUES(?,?,?,?,?,?,?)",
            (now(), cur.lastrowid, a["health"], a["drift"], a["status"], a["params"]["temperature"]["expected"], a["params"]["rpm"]["expected"]))
        last = c.execute("SELECT status FROM anomaly_events ORDER BY id DESC LIMIT 1").fetchone()
        if a["status"] != "NORMAL" and (not last or last["status"] != a["status"]):
            c.execute("INSERT INTO anomaly_events(ts,status,primary_contributor,message) VALUES(?,?,?,?)", (now(), a["status"], a["primary"], a["explanation"]))
        elif a["status"] == "NORMAL" and last and last["status"] != "NORMAL":
            c.execute("INSERT INTO anomaly_events(ts,status,primary_contributor,message) VALUES(?,?,?,?)", (now(), "NORMAL", "-", "Behaviour returned to healthy Digital Twin range."))
        c.commit(); c.close()
    except Exception as e: print("DB error", e)

class Twin:
    """Healthy-state representation: trained ML model + residual statistics -> expected healthy behaviour."""
    def __init__(s): s.m = None; s.err = None; s.load()
    def load(s):
        try: s.m = joblib.load(MODEL); s.err = None
        except Exception as e:
            s.m = None; s.err = "Digital Twin model not found. Run: python scripts/generate_data.py then python scripts/train_model.py" if not os.path.exists(MODEL) else f"Model load error: {e}"
    def expected(s, x):
        if s.m is None: s.load()
        if s.m is None: raise RuntimeError(s.err)
        e = s.m["m1"].predict([[x["rpm"], x["speed"]]])[0]; r = s.m["m2"].predict([[x["speed"]]])[0]
        return dict(rpm=float(r), speed=float(x["speed"]), temperature=float(e[0]), battery_voltage=float(e[1]), fuel_consumption=float(e[2]))
    def sigma(s, p): return max(2.5 * float(s.m["std"][p]), {"temperature": 1.0, "rpm": 60, "battery_voltage": .08, "fuel_consumption": .05}[p])
twin = Twin()

def analyze(x, hist=()):
    """Single core engine used by manual, simulated, what-if, recovery and ESP32 input."""
    e = twin.expected(x); params = {}; zs = {}
    for p in ["rpm", "temperature", "battery_voltage", "fuel_consumption"]:
        dev = x[p] - e[p]; z = dev / twin.sigma(p); zs[p] = z
        params[p] = dict(actual=round(x[p], 2), expected=round(e[p], 2), deviation=round(dev, 2), pct=round(dev / e[p] * 100, 1) if e[p] else 0,
                         z=round(z, 2), score=round(100 / (1 + (abs(z) / 2.5) ** 2), 1), level="HIGH" if abs(z) >= 2.5 else "MEDIUM" if abs(z) >= 1.2 else "LOW")
    params["speed"] = dict(actual=round(x["speed"], 2), expected=round(x["speed"], 2), deviation=0, pct=0, z=0, score=100, level="LOW")
    cur = max(abs(v) for v in zs.values()); series = list(hist)[-14:] + [cur]
    mean = float(np.mean(series)); persist = float(np.mean([v > 1.2 for v in series]))
    slope = float(np.polyfit(range(len(series)), series, 1)[0]) if len(series) >= 5 else 0.0
    drift = 0.6 * mean + 0.4 * cur
    if slope > 0.05 and persist > .6: drift *= 1.15
    status = "NORMAL" if drift < 1.2 else "EARLY DRIFT" if drift < 2.5 else "PRE-FAULT" if drift < 4 else "CRITICAL"
    wsum = sum(W[p] * params[p]["score"] for p in W); dh = 100 / (1 + (drift / 2.5) ** 2)
    health = round(0.8 * wsum + 0.2 * dh, 1)
    label = "Excellent" if health >= 95 else "Healthy" if health >= 85 else "Monitor" if health >= 70 else "Warning" if health >= 50 else "Critical"
    tot = sum(abs(zs[p]) * W[p] for p in W) or 1
    for p in W: params[p]["contribution"] = round(abs(zs[p]) * W[p] / tot * 100, 1)
    pr = max(W, key=lambda p: abs(zs[p]) * W[p]); d = params[pr]["deviation"]
    if status == "NORMAL": expl = "Actual behaviour is consistent with the healthy Digital Twin for the current operating conditions."
    else:
        head = "Temperature running below expected" if (pr == "temperature" and d < 0) else WORDS[pr]
        expl = f"{head} ({d:+.2f} vs expected). " + ("The deviation has persisted over recent observations." if persist > .5 and len(series) > 3 else "Based on the current reading only.")
    fix = dict(temperature="cooling system", rpm="throttle/ignition behaviour", battery_voltage="battery/charging system", fuel_consumption="fuel system and air filter")[pr]
    rec = {"NORMAL": "Continue normal operation and routine monitoring.",
           "EARLY DRIFT": f"Continue monitoring. If the trend persists, inspect the {fix}.",
           "PRE-FAULT": f"Reduce load and plan a maintenance inspection soon (check {fix}); deviation is increasing consistently.",
           "CRITICAL": "Safe stop and cooling recommended, then a maintenance inspection." if pr == "temperature" else f"Stop the vehicle safely and perform a maintenance inspection (check {fix})."}[status]
    return dict(status=status, health=health, label=label, drift=round(drift, 2), confidence=round(min(.95, .5 + .3 * persist + .05 * min(len(series), 3)), 2), slope=round(slope, 3),
                params=params, primary=pr, explanation=expl, recommendation=rec, cur_z=cur)

class Sim:
    SC = dict(temperature=5, rpm=200, battery_voltage=-.25, fuel_consumption=.12)
    RATE = dict(temperature=.3, rpm=8, battery_voltage=.02, fuel_consumption=.01)
    def __init__(s): s.rng = np.random.default_rng(); s.last_recovery = None; s.reset()
    def reset(s):
        s.speed = 40.; s.T = None; s.mode = "NORMAL"; s.off = {p: 0. for p in s.SC}; s.tgt = {p: 0. for p in s.SC}; s.hist = []; s.series = []; s.last = None; s.lasta = None
    def set_mode(s, m): k = dict(NORMAL=0, EARLY=1, PRE=2, CRITICAL=4)[m]; s.tgt = {p: v * k for p, v in s.SC.items()}; s.mode = m
    def inject(s, p): s.tgt[p] = s.SC[p] * 3; s.mode = "INJECT " + p
    def tick(s):
        r = s.rng; s.speed = float(np.clip(s.speed + r.normal(0, 3), 10, 70)); rpm = rpm_f(s.speed) + r.normal(0, 60)
        tt = temp_f(rpm, s.speed); s.T = tt if s.T is None else s.T + .15 * (tt - s.T) + r.normal(0, .3)
        for p in s.off: s.off[p] += float(np.clip(s.tgt[p] - s.off[p], -s.RATE[p], s.RATE[p]))
        x = dict(rpm=rpm + s.off["rpm"], speed=s.speed, temperature=s.T + s.off["temperature"] + r.normal(0, .3),
                 battery_voltage=batt_f(rpm) + s.off["battery_voltage"] + r.normal(0, .05), fuel_consumption=fuel_f(rpm, s.speed) + s.off["fuel_consumption"] + r.normal(0, .03))
        a = analyze(x, s.hist); s.hist = (s.hist + [a["cur_z"]])[-30:]; s.last = x; s.lasta = a
        pt = dict(t=datetime.now().strftime("%H:%M:%S"), drift=a["drift"], health=a["health"], status=a["status"], speed=round(x["speed"], 1))
        for p in ["rpm", "temperature", "battery_voltage", "fuel_consumption"]: pt[p] = a["params"][p]["actual"]; pt[p + "_e"] = a["params"][p]["expected"]
        s.series = (s.series + [pt])[-60:]; log_sample(x, a, "simulation"); return x, a
sim = Sim()

def snap(): return dict(mode=sim.mode, analysis=sim.lasta, series=sim.series, real=sim.last)
def err(msg, code=400): return jsonify(error=msg), code
def guard(f):
    def w(*a, **k):
        try: return f(*a, **k)
        except RuntimeError as e: return err(str(e), 503)
        except Exception as e: return err(f"Server error: {e}", 500)
    w.__name__ = f.__name__; return w
def validate(d):
    out = {}
    for k, (lo, hi, name) in LIMITS.items():
        v = (d or {}).get(k)
        if v is None or str(v).strip() == "": raise ValueError(f"Please enter {name}.")
        try: v = float(v)
        except Exception: raise ValueError(f"Please enter a numeric value for {name}.")
        if not (lo <= v <= hi): raise ValueError(f"Please enter a realistic {name} ({lo}-{hi}).")
        out[k] = v
    return out

@app.get("/api/current")
@guard
def current():
    if request.args.get("peek") != "1" or sim.lasta is None: sim.tick()
    return jsonify(snap())
@app.get("/api/health")
@guard
def health():
    if sim.lasta is None: sim.tick()
    a = sim.lasta; return jsonify(health=a["health"], label=a["label"], status=a["status"], params={p: v["score"] for p, v in a["params"].items()})
@app.get("/api/digital-twin")
@guard
def dtwin():
    if sim.lasta is None: sim.tick()
    return jsonify(real=sim.last, params=sim.lasta["params"], status=sim.lasta["status"], model="RandomForest baseline (LSTM-ready)")
@app.get("/api/anomaly")
@guard
def anomaly():
    if sim.lasta is None: sim.tick()
    c = db(); ev = [dict(r) for r in c.execute("SELECT * FROM anomaly_events ORDER BY id DESC LIMIT 20")]; c.close()
    return jsonify(analysis=sim.lasta, events=ev)
@app.post("/api/manual-analysis")
@guard
def manual():
    try: x = validate(request.get_json(silent=True))
    except ValueError as e: return err(str(e))
    a = analyze(x); log_sample(x, a, "manual"); return jsonify(analysis=a)
@app.post("/api/simulate")
@guard
def simulate():
    d = request.get_json(silent=True) or {}
    if d.get("mode"):
        if d["mode"] not in ("NORMAL", "EARLY", "PRE", "CRITICAL"): return err("Unknown mode")
        sim.set_mode(d["mode"]); log_sim("mode " + d["mode"])
    elif d.get("inject") in Sim.SC: sim.inject(d["inject"]); log_sim("inject " + d["inject"])
    else: return err("Provide 'mode' or 'inject'")
    return jsonify(mode=sim.mode)
@app.post("/api/reset")
@guard
def reset(): sim.reset(); log_sim("reset"); return jsonify(ok=True)
@app.post("/api/what-if")
@guard
def what_if():
    d = request.get_json(silent=True) or {}
    try: x = validate(d); dur = int(d.get("duration", 20))
    except ValueError as e: return err(str(e))
    if not 1 <= dur <= 60: return err("Simulation duration must be 1-60 minutes.")
    e = twin.expected(x); T0 = x["temperature"]; dv = T0 - e["temperature"]; pts = []
    for m in range(dur + 1):
        hl = e["temperature"] + dv * np.exp(-m / 8)
        rk = e["temperature"] + dv * np.exp(m / 25) if dv > 0 else hl + 0.05 * m * (x["rpm"] / 6000)
        pts.append(dict(minute=m, healthy=round(float(hl), 1), risk=round(float(rk), 1)))
    z = (pts[-1]["risk"] - e["temperature"]) / twin.sigma("temperature")
    lvl = "LOW" if z < 1.2 else "MODERATE" if z < 2.5 else "HIGH" if z < 4 else "SEVERE"
    return jsonify(expected_temperature=round(e["temperature"], 1), current=T0, points=pts, risk_level=lvl,
                   note="Model-based simulation using the healthy Digital Twin; not a guaranteed future outcome.")
@app.post("/api/recovery")
@guard
def recovery():
    d = request.get_json(silent=True) or {}
    if sim.last is None: sim.tick()
    x0 = dict(sim.last); before = analyze(x0, sim.hist); e = twin.expected(x0)
    eff = float(d.get("effectiveness", .95 if before["status"] != "CRITICAL" else .75)); steps = []
    for f in (0, .35, .65, .85, 1.0):
        x = {p: (x0[p] + (e[p] - x0[p]) * f * eff if p in Sim.SC else x0[p]) for p in x0}; a = analyze(x)
        steps.append(dict(temperature=a["params"]["temperature"]["actual"], health=a["health"], status=a["status"]))
    after = a
    res = "RECOVERY VERIFIED" if after["status"] == "NORMAL" and after["health"] >= 85 else "PARTIAL RECOVERY" if after["health"] > before["health"] + 10 else "NOT RECOVERED"
    rec = dict(result=res, before=dict(health=before["health"], status=before["status"], temperature=before["params"]["temperature"]["actual"]),
               after=dict(health=after["health"], status=after["status"], temperature=after["params"]["temperature"]["actual"]),
               expected_temperature=round(e["temperature"], 1), steps=steps, action=before["recommendation"],
               advice="Behaviour is back near the healthy Digital Twin." if res.startswith("RECOVERY") else "Further inspection is recommended.")
    if res != "NOT RECOVERED":
        sim.tgt = {p: 0. for p in sim.SC}; sim.off = {p: v * (1 - eff) for p, v in sim.off.items()}; sim.hist = []; sim.mode = "NORMAL"
    try: c = db(); c.execute("INSERT INTO recovery_events(ts,before_health,after_health,result,detail) VALUES(?,?,?,?,?)", (now(), before["health"], after["health"], res, rec["advice"])); c.commit(); c.close()
    except Exception as ex: print("DB error", ex)
    sim.last_recovery = rec; return jsonify(rec)
@app.get("/api/recovery-status")
@guard
def rstatus(): return jsonify(last=sim.last_recovery, current_status=sim.lasta["status"] if sim.lasta else None)
@app.get("/api/history")
@guard
def history():
    q = request.args
    if q.get("start"): s0, e0 = q["start"], q.get("end") or datetime.now().isoformat()
    else: s0, e0 = (datetime.now() - timedelta(days=int(q.get("days", 1)))).isoformat(), datetime.now().isoformat()
    c = db(); ph = (s0, e0)
    rows = [dict(r) for r in c.execute("SELECT h.ts,s.source,s.rpm,s.speed,s.temperature,s.battery_voltage,s.fuel_consumption,h.health,h.drift,h.status FROM health_history h JOIN sensor_data s ON s.id=h.sensor_id WHERE h.ts BETWEEN ? AND ? ORDER BY h.id", ph)]
    an = [dict(r) for r in c.execute("SELECT * FROM anomaly_events WHERE ts BETWEEN ? AND ? ORDER BY id DESC", ph)]
    rc = [dict(r) for r in c.execute("SELECT * FROM recovery_events WHERE ts BETWEEN ? AND ? ORDER BY id DESC", ph)]; c.close()
    if q.get("format") == "csv":
        o = io.StringIO(); w = csv.DictWriter(o, fieldnames=list(rows[0].keys()) if rows else ["ts"]); w.writeheader(); w.writerows(rows)
        return Response(o.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=raider_history.csv"})
    step = max(1, len(rows) // 300); return jsonify(records=rows[::step], anomalies=an, recoveries=rc, total=len(rows))
@app.post("/api/ingest")  # future ESP32 entry point -> same analysis engine
@guard
def ingest():
    try: x = validate(request.get_json(silent=True))
    except ValueError as e: return err(str(e))
    a = analyze(x, sim.hist); sim.hist = (sim.hist + [a["cur_z"]])[-30:]; log_sample(x, a, "esp32"); return jsonify(analysis=a)

init_db()
if __name__ == "__main__":
    if twin.m is None: print("WARNING:", twin.err)
    app.run(port=5000, debug=False)
