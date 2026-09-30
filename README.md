# RAIDER AI — Digital Twin Motorcycle Health Intelligence (TVS Raider 125)
## Setup (Windows, VS Code) — two terminals
Terminal 1 (backend):
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python scripts/generate_data.py
    python scripts/create_drift.py
    python scripts/train_model.py
    python backend/app.py
Terminal 2 (frontend):
    cd frontend
    npm install
    npm run dev
Open http://localhost:5173  (login: admin / raider123). Backend runs on port 5000.
## Demo flow
Dashboard (NORMAL) -> Simulation Control: INJECT TEMPERATURE DRIFT -> watch Live Monitoring lines separate (EARLY DRIFT -> PRE-FAULT) -> Drift & Explainability -> What-If -> Safe Response: SIMULATE RECOVERY -> RECOVERY VERIFIED.
## Modules
backend/app.py: Flask API, Digital Twin, drift engine (z-score, persistence, trend), health score, what-if, recovery, SQLite. physics.py: healthy relationships. scripts/: data, drift data, training.
## Troubleshooting
"Model not found": run generate_data.py then train_model.py. "Backend unavailable": start backend first. Port 5000 busy: change port in app.py and vite.config.js.
## Future ESP32
POST JSON {rpm,speed,temperature,battery_voltage,fuel_consumption} to /api/ingest over Wi-Fi; it uses the same analysis engine.
