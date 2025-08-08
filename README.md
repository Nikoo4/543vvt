# Roulette Prediction Server

FastAPI service that predicts the roulette pocket using three ball crossings.
* Returns **prediction** immediately.
* After 60 valid rows with good accuracy adds `periskok` – 4 pockets adjusted for jump bias.
* Stores only valid rounds, keeps dataset under 600 rows and self‑prunes when mean error grows.

## Quick start
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```
