"""
Roulette prediction server

"""

from __future__ import annotations
import math, csv, os, time, uuid, errno
from collections import deque
from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────  roulette layout  ────────────────────────────
WSEQ = [
    0, 32, 15, 19,  4, 21,  2, 25, 17, 34,  6, 27,
   13, 36, 11, 30,  8, 23, 10,  5, 24, 16, 33,  1,
   20, 14, 31,  9, 22, 18, 29,  7, 28, 12, 35,  3, 26
]
POS  = {n: i for i, n in enumerate(WSEQ)}
IDX_MAX = len(WSEQ)

def idx_wrap(i: int) -> int: 
   return i % IDX_MAX

def ang_wrap(a: float) -> float: 
   return (a + 2*math.pi) % (2*math.pi)

# ────────────────────────  request models  ──────────────────────────────
class Crossing(BaseModel):
   idx: int
   t: float
   theta: float
   slot: Optional[int] = None
   phi: float

class PredictRequest(BaseModel):
   crossings: List[Crossing]
   direction: str                # "cw" / "ccw"
   theta_zero: float
   ts_start: Optional[int] = None

class LogWinnerRequest(BaseModel):
   round_id: str
   winning_number: int
   timestamp: Optional[int] = None
   predicted_number: Optional[int] = None

# ────────────────────────────  config  ──────────────────────────────────
# Physics constants
G = 9.81  # gravity m/s²
R_WHEEL = 0.41  # wheel radius in meters (820mm / 2)
R_TRACK = 0.48  # track radius
R_DEFLECTOR = 0.38  # deflector radius
TABLE_INCLINE = math.radians(2.5)  # table incline angle
MU_ROLLING = 0.005  # rolling friction coefficient
DT_INT = 0.001  # integration time step

# File and limits
MAX_ROWS = int(os.getenv("ROULETTE_MAX_ROWS", "600"))
MAX_ERR = int(os.getenv("ROULETTE_MAX_ERR", "50"))

MIN_SPINS_FOR_MODE = int(os.getenv("ROULETTE_MIN_SPINS_FOR_MODE", "100"))
GOOD_MEAN_ERROR = float(os.getenv("ROULETTE_GOOD_MEAN_ERROR", "8.0"))

IMPACT_DT = float(os.getenv("ROULETTE_IMPACT_DT", "0.60"))

CSV_HEADER = [
   "round_id", "ts_start", "direction", "theta_zero",
   "ball_t1", "ball_theta1", "ball_phi1",
   "ball_t2", "ball_theta2", "ball_phi2",
   "ball_t3", "ball_theta3", "ball_phi3",
   "omega_ball", "alpha_ball", "omega_wheel", "alpha_wheel",
   "predicted_number", "periskok",
   "ts_predict",
   "winning_number", "ts_winner",
   "error_slots", "latency_ms",
]

# ---- robust CSV path picking + safe IO ----
def _as_file_path(p: str) -> str:
    """Если p — каталог (существующий/оканчивается на / или последний сегмент без точки) → дополним именем файла."""
    if not p:
        return ""
    p = p.strip().rstrip("/\\")
    base = os.path.basename(p)
    if os.path.isdir(p) or p.endswith(os.sep) or ("." not in base and os.path.isabs(p)):
        return os.path.join(p, "dataset.csv")
    return p

def _try_path(candidates: List[str]) -> str:
    for c in candidates:
        if not c:
            continue
        c = _as_file_path(c)
        d = os.path.dirname(c) or "."
        try:
            os.makedirs(d, exist_ok=True)
            # проверяем запись
            with open(c, "a", encoding="utf-8") as f:
                pass
            # если файл новый — пишем заголовок
            if os.path.getsize(c) == 0:
                with open(c, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(CSV_HEADER)
            return c
        except Exception as e:
            print(f"[init] skip '{c}': {e}")
            continue
    raise RuntimeError("No writable location for dataset.csv")

def pick_csv_path() -> str:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    env_p  = os.getenv("ROULETTE_CSV", "").strip()
    state  = os.getenv("STATE_DIRECTORY", "").split(":")[0]  # systemd StateDirectory
    xdg    = os.getenv("XDG_STATE_HOME", os.path.join(os.path.expanduser("~"), ".local", "state"))

    candidates = [
        env_p,
        os.path.join(state, "dataset.csv") if state else "",
        os.path.join(xdg, "roulette", "dataset.csv"),
        os.path.join(os.path.expanduser("~"), "roulette", "dataset.csv"),
        os.path.join(app_dir, "data", "dataset.csv"),
        "/var/tmp/roulette_dataset.csv",  # последний шанс (обычно переживает перезагрузку)
    ]
    return _try_path(candidates)

CSV_PATH = pick_csv_path()
print(f"[init] Using dataset: {CSV_PATH}")

# ───────────────────────  math utils  ───────────────────────────────────
def _savgol_5_2(arr):
   """Simple Savitzky-Golay filter (window=5, poly=2)"""
   if len(arr) < 5:
       return arr
   coeff = [-3, 12, 17, 12, -3]
   out = []
   for i in range(len(arr)):
       if i < 2 or i > len(arr) - 3:
           out.append(arr[i])
       else:
           s = 0
           for k, c in enumerate(coeff):
               s += c * arr[i - 2 + k]
           out.append(s / 35)
   return out

def is_strictly_inc(arr: List[float]) -> bool:
   return all(arr[i] < arr[i+1] for i in range(len(arr)-1))

def fit_theta_poly(ts: List[float], thetas: List[float]):
   """
   Quadratic fit θ(t)=a0+a1 t+a2 t².  Return (a0,a1,a2), ω=a1, α=2a2
   """
   t0 = ts[0]
   xs = [t - t0 for t in ts]
   n = len(xs)
   
   # Simple closed‑form normal equations for 3 points
   Sx = sum(xs)
   Sx2 = sum(x*x for x in xs)
   Sx3 = sum(x**3 for x in xs)
   Sx4 = sum(x**4 for x in xs)
   Sy = sum(thetas)
   Sxy = sum(x*y for x, y in zip(xs, thetas))
   Sx2y = sum(x*x*y for x, y in zip(xs, thetas))

   det = (n*Sx2*Sx4 + 2*Sx*Sx2*Sx3 - Sx2*Sx2*Sx2 - n*Sx3*Sx3 - Sx*Sx*Sx4)
   if abs(det) < 1e-9: 
       raise ValueError("singular matrix")

   a0 = (Sy*Sx2*Sx4 + Sx*Sx3*Sx2y + Sx2*Sx3*Sxy -
         Sx2*Sy*Sx2 - Sx*Sx2y*Sx4 - Sx3*Sx3*Sxy) / det
   a1 = (n*Sx3*Sx2y + Sy*Sx*Sx4 + Sx2*Sx2*Sxy -
         Sx2*Sy*Sx3 - n*Sx4*Sxy - Sx*Sx*Sx2y) / det
   a2 = (n*Sx2*Sx2y + Sx*Sx2*Sxy + Sy*Sx*Sx3 -
         Sx2*Sy*Sx2 - Sx*Sx*Sx2y - n*Sx3*Sxy) / det
   
   return (a0, a1, a2), a1, 2*a2

def compute_ball_trajectory(omega0, alpha):
   """
   Compute time for ball to reach deflectors using physics model from PDF
   """
   # Critical angular velocity (equation 1 from PDF)
   omega_critical = math.sqrt(G * math.tan(TABLE_INCLINE) / R_TRACK)
   
   # Time on rim
   if omega0 > omega_critical:
       t_rim = (omega0 - omega_critical) / abs(alpha)
   else:
       t_rim = 0
   
   # Time sliding on stator
   r = R_TRACK
   v_r = 0  # radial velocity
   t = 0
   omega = omega_critical if t_rim > 0 else omega0
   
   while r > R_DEFLECTOR and t < 5.0:
       # Radial acceleration (equation 3 from PDF)
       a_r = r * omega * omega * math.cos(TABLE_INCLINE) - G * math.sin(TABLE_INCLINE)
       
       # Update velocity and position
       v_r += a_r * DT_INT
       r += v_r * DT_INT
       
       # Angular velocity continues to decrease
       omega += alpha * DT_INT
       t += DT_INT
       
       # Prevent infinite loop
       if omega <= 0:
           break
   
   return t_rim + t

def map_angle_to_number(theta: float) -> int:
   idx = int(round((theta / (2*math.pi)) * IDX_MAX)) % IDX_MAX
   return WSEQ[idx]

def slots_delta(pred: int, win: int, direction: str = "cw") -> int:
   ip, iw = POS[pred], POS[win]
   if direction.lower() == "cw":
       return idx_wrap(iw - ip)
   else:
       return idx_wrap(ip - iw)

# ─────────────────────  CSV helpers  ────────────────────────────────────
def ensure_csv():
    # уже сделано в pick_csv_path(); оставляем для совместимости
    if not os.path.exists(CSV_PATH):
        _try_path([CSV_PATH])

def read_rows():
   if not os.path.isfile(CSV_PATH):
       return deque(maxlen=800)
   try:
       with open(CSV_PATH, newline="", encoding="utf-8") as f:
           rows = list(csv.DictReader(f))
       return deque(rows, maxlen=800)
   except Exception as e:
       print(f"Error reading CSV file: {e}")
       return deque(maxlen=800)

def write_row(row: Dict[str, str]):
    ensure_csv()
    try:
        import fcntl
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            try: 
                fcntl.flock(f, fcntl.LOCK_EX)
            except Exception: 
                pass  # Windows не поддерживает fcntl
            csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)
            f.flush()
            os.fsync(f.fileno())
            try: 
                fcntl.flock(f, fcntl.LOCK_UN)
            except Exception: 
                pass
    except Exception as e:
        print(f"Error writing to CSV: {e}")
        raise

def tail_rows(n: int) -> List[Dict[str, str]]:
   rows = read_rows()
   return list(rows)[-n:] if len(rows) > n else list(rows)

def enough_stats() -> bool:
   return len(tail_rows(MIN_SPINS_FOR_MODE)) >= MIN_SPINS_FOR_MODE

def mean_error_good() -> bool:
   """Check if mean error is good enough"""
   recent = tail_rows(50)
   if len(recent) < 10:
       return False
   
   errors = []
   for r in recent:
       if r.get("winning_number") and r.get("predicted_number"):
           try:
               err = abs(slots_delta(
                   int(r["predicted_number"]),
                   int(r["winning_number"]),
                   r.get("direction", "cw")
               ))
               errors.append(err)
           except:
               pass
   
   if not errors:
       return False
   
   mean_err = sum(errors) / len(errors)
   return mean_err <= GOOD_MEAN_ERROR

def offset_mode(direction: str) -> int:
   hist: Dict[int, int] = {}
   for r in tail_rows(MAX_ROWS):
       if not r.get("winning_number"): 
           continue
       if r["direction"].lower() != direction.lower(): 
           continue
       try:
           d = slots_delta(
               int(r["predicted_number"]),
               int(r["winning_number"]), 
               direction
           )
           if abs(d) <= 12:  # Only consider reasonable offsets
               hist[d] = hist.get(d, 0) + 1
       except: 
           pass
   
   if not hist: 
       return 0
   
   # Return most common offset
   return max(hist.items(), key=lambda kv: kv[1])[0]

def build_periskok(pred: int, direction: str) -> List[int]:
   """Build list of 4 neighboring pockets based on learned offset"""
   if not enough_stats() or not mean_error_good(): 
       return []
   
   off = offset_mode(direction)
   if off == 0:  # No clear offset pattern
       return []
   
   # Get index of predicted + offset
   i_center = idx_wrap(POS[pred] + off)
   center_num = WSEQ[i_center]
   
   # Get 2 neighbors on each side
   neighbors = []
   for delta in [1, -1, 2, -2]:
       idx = idx_wrap(i_center + delta)
       neighbors.append(WSEQ[idx])
   
   # Return center + best 3 neighbors (total 4)
   return [center_num] + neighbors[:3]

def valid_payload(ts: List[float], thetas: List[float], phis: List[float], direction: str) -> bool:
   """Validate incoming data"""
   # Basic length check only
   if len(ts) == 3 and len(thetas) == 3 and len(phis) == 3:
       return True
   return False
   
   # # More strict validation - COMMENTED OUT FOR DATA COLLECTION
   # if not is_strictly_inc(ts):
   #     return False
   
   # try:
   #     _, omega, alpha = fit_theta_poly(ts, thetas)
   #     if not (0.1 <= abs(omega) <= 50):
   #         return False
   #     if not (-10 <= alpha <= 0.1):
   #         return False
   # except:
   #     return False
   
   # return True

def prune_csv():
   rows = read_rows()
   if len(rows) <= MAX_ROWS: 
       return
   
   # Compute recent MAE
   recent = list(rows)[-100:]
   errs = []
   for r in recent:
       if r.get("winning_number") and r.get("predicted_number"):
           try:
               err = abs(slots_delta(
                   int(r["predicted_number"]),
                   int(r["winning_number"]),
                   r.get("direction", "cw")
               ))
               errs.append(err)
           except: 
               pass
   
   mean_err = sum(errs) / len(errs) if errs else GOOD_MEAN_ERROR + 1
   
   # Drop old rows if error is bad
   if mean_err > GOOD_MEAN_ERROR * 1.5:
       rows = deque(list(rows)[100:], maxlen=800)
   
   # Enforce max rows
   rows = deque(list(rows)[-MAX_ROWS:], maxlen=800)
   
   # Write back
   tmp = CSV_PATH + ".tmp"
   try:
       with open(tmp, "w", newline="", encoding="utf-8") as f:
           w = csv.DictWriter(f, fieldnames=CSV_HEADER)
           w.writeheader()
           w.writerows(rows)
       os.replace(tmp, CSV_PATH)
   except Exception as e:
       print(f"Error pruning CSV: {e}")

# ─────────────────────────  FastAPI  ────────────────────────────────────
app = FastAPI(title="Roulette Server")

app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"], 
   allow_methods=["*"],
   allow_headers=["*"], 
   allow_credentials=True
)

PENDING: Dict[str, Dict[str, Any]] = {}

@app.on_event("startup")
def startup_event():
    """Initialize CSV file on server startup"""
    try:
        ensure_csv()
        print(f"CSV file initialized at: {CSV_PATH}")
    except Exception as e:
        print(f"Warning: Could not initialize CSV on startup: {e}")

@app.get("/")
def root():
   try:
       ensure_csv()
       rows_count = len(read_rows())
   except Exception as e:
       print(f"Error in root endpoint: {e}")
       rows_count = 0
   return {"status": "Roulette prediction server running", "dataset_rows": rows_count, "csv_path": CSV_PATH}

@app.get("/health")
def health():
   try:
       rows_count = len(read_rows())
   except:
       rows_count = 0
   return {"ok": True, "rows": rows_count, "pending": len(PENDING)}

# ---------- /predict ----------
@app.post("/predict")
def predict(req: PredictRequest):
   try:
       # Extract data
       ts = [c.t for c in req.crossings]
       thetas = [c.theta for c in req.crossings]
       phis = [c.phi for c in req.crossings]
       
       # Validate - SIMPLIFIED
       if not valid_payload(ts, thetas, phis, req.direction):
           return {"ok": False, "error": "Invalid payload"}
       
       # Apply smoothing
       thetas_smooth = _savgol_5_2(thetas)
       phis_smooth = _savgol_5_2(phis)
       
       # Fit ball motion
       try:
           (a0, a1, a2), omega_ball, alpha_ball = fit_theta_poly(ts, thetas_smooth)
       except Exception as e:
           # Use fallback values
           omega_ball = (thetas[-1] - thetas[0]) / (ts[-1] - ts[0])
           alpha_ball = -0.5
           a0 = thetas[0]
           a1 = omega_ball
           a2 = alpha_ball / 2
           # return {"ok": False, "error": f"Ball fit failed: {str(e)}"}
       
       # Fit wheel motion
       try:
           (b0, b1, b2), omega_wheel, alpha_wheel = fit_theta_poly(ts, phis_smooth)
       except Exception as e:
           # Wheel might be stationary
           omega_wheel = 0
           alpha_wheel = 0
           b0 = phis_smooth[0] if phis_smooth else 0
           b1 = b2 = 0
       
       # Compute time to deflectors
       try:
           dt_to_deflector = compute_ball_trajectory(abs(omega_ball), alpha_ball)
           # Sanity check
           if dt_to_deflector < 0.1 or dt_to_deflector > 10:
               dt_to_deflector = IMPACT_DT
       except Exception as e:
           print(f"Trajectory computation failed: {e}")
           dt_to_deflector = IMPACT_DT
       
       # Time from start of measurements
       t_impact = ts[-1] + dt_to_deflector
       dt_from_start = t_impact - ts[0]
       
       # Positions at impact
       theta_ball_impact = a0 + a1 * dt_from_start + a2 * dt_from_start * dt_from_start
       phi_wheel_impact = b0 + b1 * dt_from_start + b2 * dt_from_start * dt_from_start
       
       # Relative angle
       theta_relative = ang_wrap(theta_ball_impact - phi_wheel_impact)
       
       # Convert to number
       predicted_number = map_angle_to_number(theta_relative)
       
       # Check for periskok
       periskok = build_periskok(predicted_number, req.direction)
       
       # Generate round ID
       round_id = str(uuid.uuid4())
       now = int(time.time() * 1000)
       
       # Store for validation
       PENDING[round_id] = {
           "ts_start": req.ts_start or now,
           "direction": req.direction.lower(),
           "theta_zero": req.theta_zero,
           "ts": ts,
           "thetas": thetas,
           "phis": phis,
           "omega_ball": omega_ball,
           "alpha_ball": alpha_ball,
           "omega_wheel": omega_wheel,
           "alpha_wheel": alpha_wheel,
           "predicted": predicted_number,
           "periskok": periskok,
           "ts_predict": now
       }
       
       # Build response
       response = {
           "ok": True,
           "round_id": round_id,
           "prediction": int(predicted_number),
           "omega_ball": round(omega_ball, 6),
           "alpha_ball": round(alpha_ball, 6),
           "dataset_rows": len(read_rows())
       }
       
       if periskok:
           response["periskok"] = periskok
       
       print(f"Prediction: {predicted_number}, Periskok: {periskok}, Rows: {response['dataset_rows']}")
       
       return response
       
   except Exception as e:
       print(f"Predict error: {e}")
       import traceback
       traceback.print_exc()
       return {"ok": False, "error": str(e)}

# ---------- /log_winner ----------
@app.post("/log_winner")
def log_winner(req: LogWinnerRequest):
   try:
       now = int(time.time() * 1000)
       entry = PENDING.pop(req.round_id, None)
       
       if entry is None:
           return {"ok": True, "ignored": True, "reason": "no_matching_round"}
       
       # Get data from entry
       ts = entry["ts"]
       thetas = entry["thetas"]
       phis = entry["phis"]
       direction = entry["direction"]
       
       # # VALIDATION REMOVED FOR DATA COLLECTION
       # if len(ts) != 3 or len(thetas) != 3 or len(phis) != 3:
       #     return {"ok": True, "ignored": True, "reason": "invalid_data_length"}
       
       # Calculate error
       err = slots_delta(entry["predicted"], req.winning_number, direction)
       
       # # ERROR FILTER REMOVED FOR DATA COLLECTION
       # if abs(err) > MAX_ERR:
       #     return {"ok": True, "ignored": True, "reason": "error_too_large", "error_slots": err}
       
       # Build CSV row
       row = {
           "round_id": req.round_id,
           "ts_start": str(entry["ts_start"]),
           "direction": direction,
           "theta_zero": f"{entry['theta_zero']:.6f}",
           "ball_t1": f"{ts[0]:.6f}", 
           "ball_theta1": f"{thetas[0]:.6f}",
           "ball_phi1": f"{phis[0]:.6f}",
           "ball_t2": f"{ts[1]:.6f}", 
           "ball_theta2": f"{thetas[1]:.6f}",
           "ball_phi2": f"{phis[1]:.6f}",
           "ball_t3": f"{ts[2]:.6f}", 
           "ball_theta3": f"{thetas[2]:.6f}",
           "ball_phi3": f"{phis[2]:.6f}",
           "omega_ball": f"{entry['omega_ball']:.6f}",
           "alpha_ball": f"{entry['alpha_ball']:.6f}",
           "omega_wheel": f"{entry['omega_wheel']:.6f}",
           "alpha_wheel": f"{entry['alpha_wheel']:.6f}",
           "predicted_number": str(entry["predicted"]),
           "periskok": ",".join(map(str, entry["periskok"])) if entry["periskok"] else "",
           "ts_predict": str(entry["ts_predict"]),
           "winning_number": str(req.winning_number),
           "ts_winner": str(req.timestamp or now),
           "error_slots": str(err),
           "latency_ms": str((req.timestamp or now) - entry["ts_predict"]),
       }
       
       write_row(row)
       prune_csv()
       
       rows_count = len(read_rows())
       print(f"Winner logged: predicted={entry['predicted']}, actual={req.winning_number}, error={err}, total_rows={rows_count}")
       
       return {"ok": True, "stored": True, "dataset_rows": rows_count}
       
   except Exception as e:
       print(f"Log winner error: {e}")
       import traceback
       traceback.print_exc()
       return {"ok": False, "error": str(e)}

if __name__ == "__main__":
   # Print startup info
   print(f"Starting Roulette Server...")
   print(f"CSV Path: {CSV_PATH}")
   print(f"Max Rows: {MAX_ROWS}")
   
   # Ensure CSV exists at startup
   try:
       ensure_csv()
       print(f"CSV file ready at: {CSV_PATH}")
   except Exception as e:
       print(f"Failed to initialize CSV: {e}")
   
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)
