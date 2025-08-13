"""
Professional Roulette Prediction Server
Advanced physics-based prediction with bounce modeling
"""

from __future__ import annotations
import math, csv, os, time, uuid, json
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────  Roulette Layout  ─────────────────────────────
WHEEL_SEQUENCE = [
    0, 32, 15, 19,  4, 21,  2, 25, 17, 34,  6, 27,
   13, 36, 11, 30,  8, 23, 10,  5, 24, 16, 33,  1,
   20, 14, 31,  9, 22, 18, 29,  7, 28, 12, 35,  3, 26
]
POCKET_POSITION = {n: i for i, n in enumerate(WHEEL_SEQUENCE)}
WHEEL_SIZE = len(WHEEL_SEQUENCE)

def normalize_index(i: int) -> int: 
    return i % WHEEL_SIZE

def normalize_angle(a: float) -> float: 
    return (a + 2*math.pi) % (2*math.pi)

# ─────────────────────────  Request Models  ──────────────────────────────
class Crossing(BaseModel):
    idx: int
    t: float
    theta: float
    slot: Optional[int] = None
    phi: float

class PredictRequest(BaseModel):
    crossings: List[Crossing]
    direction: str
    theta_zero: float
    ts_start: Optional[int] = None

class LogWinnerRequest(BaseModel):
    round_id: str
    winning_number: int
    timestamp: Optional[int] = None
    predicted_number: Optional[int] = None

# ─────────────────────────  Physics Constants  ─────────────────────────────
# Fundamental physics
GRAVITY = 9.81  # m/s²
WHEEL_RADIUS = 0.41  # meters
TRACK_RADIUS = 0.48
DEFLECTOR_RADIUS = 0.38
TABLE_TILT = math.radians(2.5)
ROLLING_FRICTION = 0.005
INTEGRATION_STEP = 0.001

# Bounce physics
DEFLECTOR_ELASTICITY = 0.65  # Energy retention on deflector hit
FRET_HEIGHT = 0.008  # 8mm pocket dividers
BALL_RADIUS = 0.01   # 10mm ball
POCKET_WIDTH = 0.053  # 53mm pocket width
BOUNCE_RANDOMNESS = 0.15  # 15% random factor

# Learning parameters
MIN_DATA_FOR_BOUNCE_MODEL = 50
BOUNCE_PATTERN_WINDOW = 100
CONFIDENCE_THRESHOLD = 0.7

# File management
DATA_FILE_NAME = "roulette_data.csv"
MAX_DATASET_SIZE = 1000

CSV_COLUMNS = [
    "round_id", "ts_start", "direction", "theta_zero",
    "ball_t1", "ball_theta1", "ball_phi1",
    "ball_t2", "ball_theta2", "ball_phi2",
    "ball_t3", "ball_theta3", "ball_phi3",
    "omega_ball", "alpha_ball", "omega_wheel", "alpha_wheel",
    "predicted_number", "jump_numbers",
    "ts_predict",
    "winning_number", "ts_winner",
    "error_slots", "bounce_pattern",
]

# ─────────────────────────  File Management  ──────────────────────────────
def _as_file_path(p: str) -> str:
    """If p is a directory, append filename"""
    if not p:
        return ""
    p = p.strip().rstrip("/\\")
    base = os.path.basename(p)
    if os.path.isdir(p) or p.endswith(os.sep) or ("." not in base and os.path.isabs(p)):
        return os.path.join(p, DATA_FILE_NAME)
    return p

def _try_path(candidates: List[str]) -> str:
    """Try each candidate path until one works"""
    for c in candidates:
        if not c:
            continue
        c = _as_file_path(c)
        d = os.path.dirname(c) or "."
        try:
            os.makedirs(d, exist_ok=True)
            # Test write access
            with open(c, "a", encoding="utf-8") as f:
                pass
            # If new file, write header
            if os.path.getsize(c) == 0:
                with open(c, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(CSV_COLUMNS)
            return c
        except Exception as e:
            print(f"[init] skip '{c}': {e}")
            continue
    raise RuntimeError("No writable location for roulette_data.csv")

def get_data_path() -> str:
    """Determine optimal path for data storage using working server logic"""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    env_p = os.getenv("ROULETTE_DATA_PATH", "").strip()
    state = os.getenv("STATE_DIRECTORY", "").split(":")[0]  # systemd StateDirectory
    xdg = os.getenv("XDG_STATE_HOME", os.path.join(os.path.expanduser("~"), ".local", "state"))

    candidates = [
        env_p,
        os.path.join(state, DATA_FILE_NAME) if state else "",
        os.path.join(xdg, "roulette", DATA_FILE_NAME),
        os.path.join(os.path.expanduser("~"), "roulette", DATA_FILE_NAME),
        os.path.join(app_dir, "data", DATA_FILE_NAME),
        "/var/tmp/roulette_data.csv",  # Last resort (usually survives reboot)
    ]
    return _try_path(candidates)

DATA_PATH = get_data_path()
print(f"[init] Using dataset: {DATA_PATH}")

def initialize_csv():
    """Create CSV with headers if needed"""
    if not os.path.exists(DATA_PATH):
        with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLUMNS)

def read_dataset() -> List[Dict[str, str]]:
    """Read all data from CSV"""
    if not os.path.isfile(DATA_PATH):
        return []
    try:
        with open(DATA_PATH, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"Error reading data: {e}")
        return []

def append_record(record: Dict[str, str]):
    """Append new record to dataset"""
    try:
        with open(DATA_PATH, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(record)
    except Exception as e:
        print(f"Error writing record: {e}")

def maintain_dataset_size():
    """Keep dataset within size limits and remove poor quality data"""
    records = read_dataset()
    
    # First pass: remove very poor quality records if we have enough data
    if len(records) > 100:
        # Calculate average error for quality assessment
        recent_errors = []
        for r in records[-50:]:
            if r.get("error_slots"):
                try:
                    recent_errors.append(int(r["error_slots"]))
                except:
                    pass
        
        if recent_errors:
            avg_recent_error = sum(recent_errors) / len(recent_errors)
            
            # Remove records with very high errors if we have better data
            if avg_recent_error < 10:  # If we're doing well
                filtered_records = []
                for r in records:
                    try:
                        error = int(r.get("error_slots", 0))
                        # Keep if error is reasonable or it's recent data
                        if error < 15 or records.index(r) > len(records) - 50:
                            filtered_records.append(r)
                    except:
                        filtered_records.append(r)
                records = filtered_records
    
    # Second pass: enforce size limit
    if len(records) > MAX_DATASET_SIZE:
        records = records[-MAX_DATASET_SIZE:]
    
    # Rewrite file if changes were made
    if len(records) != len(read_dataset()):
        with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(records)
        print(f"Dataset cleaned: {len(records)} high-quality records retained")

# ─────────────────────────  Data Validation  ──────────────────────────────
def validate_crossings(crossings: List[Crossing]) -> Dict[str, Any]:
    """
    Validate quality of crossing data
    Returns validation result with quality score
    """
    issues = []
    deviations = []
    quality_score = 1.0
    
    # Check angle deviations from 180°
    for i, c in enumerate(crossings):
        angle_deg = (c.theta * 180 / math.pi) % 360
        
        # Calculate deviation from 180°
        # Handle both cases: near 180° and near 0°/360°
        dev_from_180 = abs(angle_deg - 180)
        dev_from_0 = min(angle_deg, 360 - angle_deg)
        
        # Take minimum deviation
        deviation = min(dev_from_180, dev_from_0)
        deviations.append(deviation)
        
        if deviation > 30:
            issues.append(f"Cross #{i+1}: Large deviation {deviation:.1f}°")
            quality_score *= 0.5
        elif deviation > 20:
            issues.append(f"Cross #{i+1}: Moderate deviation {deviation:.1f}°")
            quality_score *= 0.8
        elif deviation > 10:
            quality_score *= 0.95
    
    # Check time monotonicity
    times = [c.t for c in crossings]
    if not all(times[i] < times[i+1] for i in range(len(times)-1)):
        issues.append("Non-monotonic time sequence")
        quality_score *= 0.3
    
    # Check time intervals
    if len(times) >= 2:
        intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
        
        for i, interval in enumerate(intervals):
            if interval < 0.1:
                issues.append(f"Interval {i+1} too short: {interval:.3f}s")
                quality_score *= 0.5
            elif interval > 3.0:
                issues.append(f"Interval {i+1} too long: {interval:.3f}s")
                quality_score *= 0.7
        
        # Check consistency of intervals
        if len(intervals) >= 2:
            interval_ratio = max(intervals) / min(intervals)
            if interval_ratio > 2.0:
                issues.append(f"Inconsistent intervals (ratio: {interval_ratio:.1f})")
                quality_score *= 0.8
    
    # Check angular velocity reasonableness
    if len(times) >= 2 and len(crossings) >= 2:
        avg_angular_velocity = abs(crossings[-1].theta - crossings[0].theta) / (times[-1] - times[0])
        
        if avg_angular_velocity < 0.5:  # Too slow
            issues.append(f"Angular velocity too low: {avg_angular_velocity:.2f} rad/s")
            quality_score *= 0.6
        elif avg_angular_velocity > 20.0:  # Too fast
            issues.append(f"Angular velocity too high: {avg_angular_velocity:.2f} rad/s")
            quality_score *= 0.4
    
    # Final assessment
    valid = quality_score > 0.5  # Allow prediction if score > 0.5
    store_quality = quality_score > 0.7  # Only store if score > 0.7
    
    return {
        "valid": valid,
        "store_quality": store_quality,
        "quality_score": quality_score,
        "reason": "; ".join(issues) if issues else "OK",
        "issues": issues,
        "deviations": [f"{d:.1f}°" for d in deviations]
    }

# ─────────────────────────  Mathematical Functions  ────────────────────────
def smooth_data(data: List[float]) -> List[float]:
    """Apply Savitzky-Golay smoothing filter"""
    if len(data) < 5:
        return data
    
    coefficients = [-3, 12, 17, 12, -3]
    smoothed = []
    
    for i in range(len(data)):
        if i < 2 or i >= len(data) - 2:
            smoothed.append(data[i])
        else:
            value = sum(c * data[i - 2 + j] for j, c in enumerate(coefficients))
            smoothed.append(value / 35)
    
    return smoothed

def fit_trajectory(times: List[float], positions: List[float]) -> Tuple[float, float, float]:
    """
    Fit quadratic trajectory: θ(t) = a₀ + a₁t + a₂t²
    Returns: (a0, a1, a2) where ω=a1, α=2*a2
    """
    if len(times) < 3:
        raise ValueError("Need at least 3 points")
    
    t0 = times[0]
    dt = [t - t0 for t in times]
    n = len(dt)
    
    # Build normal equations
    sum_t = sum(dt)
    sum_t2 = sum(t**2 for t in dt)
    sum_t3 = sum(t**3 for t in dt)
    sum_t4 = sum(t**4 for t in dt)
    sum_y = sum(positions)
    sum_ty = sum(t*y for t, y in zip(dt, positions))
    sum_t2y = sum(t*t*y for t, y in zip(dt, positions))
    
    # Solve system
    det = n*sum_t2*sum_t4 + 2*sum_t*sum_t2*sum_t3 - sum_t2**3 - n*sum_t3**2 - sum_t**2*sum_t4
    
    if abs(det) < 1e-9:
        # Fallback to linear approximation
        omega = (positions[-1] - positions[0]) / (times[-1] - times[0])
        return positions[0], omega, -0.5
    
    a0 = (sum_y*sum_t2*sum_t4 + sum_t*sum_t3*sum_t2y + sum_t2*sum_t3*sum_ty -
          sum_t2**2*sum_y - sum_t*sum_t2y*sum_t4 - sum_t3**2*sum_ty) / det
    
    a1 = (n*sum_t3*sum_t2y + sum_y*sum_t*sum_t4 + sum_t2**2*sum_ty -
          sum_t2*sum_y*sum_t3 - n*sum_t4*sum_ty - sum_t**2*sum_t2y) / det
    
    a2 = (n*sum_t2*sum_t2y + sum_t*sum_t2*sum_ty + sum_y*sum_t*sum_t3 -
          sum_t2**2*sum_y - sum_t**2*sum_t2y - n*sum_t3*sum_ty) / det
    
    return a0, a1, a2

def calculate_impact_time(omega0: float, alpha: float) -> float:
    """Calculate time until ball reaches deflectors"""
    omega_critical = math.sqrt(GRAVITY * math.tan(TABLE_TILT) / TRACK_RADIUS)
    
    # Time on rim
    if omega0 > omega_critical:
        t_rim = (omega0 - omega_critical) / abs(alpha)
    else:
        t_rim = 0
    
    # Time sliding on track
    r = TRACK_RADIUS
    v_radial = 0
    t = 0
    omega = omega_critical if t_rim > 0 else omega0
    
    while r > DEFLECTOR_RADIUS and t < 5.0:
        a_radial = r * omega**2 * math.cos(TABLE_TILT) - GRAVITY * math.sin(TABLE_TILT)
        v_radial += a_radial * INTEGRATION_STEP
        r += v_radial * INTEGRATION_STEP
        omega += alpha * INTEGRATION_STEP
        t += INTEGRATION_STEP
        
        if omega <= 0:
            break
    
    return t_rim + t

def angle_to_pocket(angle: float) -> int:
    """Convert angle to pocket number"""
    index = int(round((angle / (2*math.pi)) * WHEEL_SIZE)) % WHEEL_SIZE
    return WHEEL_SEQUENCE[index]

def pocket_distance(pocket1: int, pocket2: int, direction: str = "cw") -> int:
    """Calculate distance between pockets"""
    i1, i2 = POCKET_POSITION[pocket1], POCKET_POSITION[pocket2]
    if direction.lower() == "cw":
        return normalize_index(i2 - i1)
    else:
        return normalize_index(i1 - i2)

# ─────────────────────────  Bounce Physics Engine  ────────────────────────
class BouncePredictor:
    """Advanced bounce prediction using physics and statistics"""
    
    def __init__(self):
        self.bounce_patterns = defaultdict(lambda: defaultdict(int))
        self.pattern_confidence = {}
        
    def update_patterns(self, impact_data: Dict[str, Any], winning_pocket: int):
        """Learn bounce patterns from historical data"""
        key = self._get_pattern_key(impact_data)
        offset = pocket_distance(impact_data['predicted'], winning_pocket, impact_data['direction'])
        
        # Store bounce pattern
        self.bounce_patterns[key][offset] += 1
        
        # Update confidence
        total = sum(self.bounce_patterns[key].values())
        if total >= MIN_DATA_FOR_BOUNCE_MODEL:
            self.pattern_confidence[key] = min(total / BOUNCE_PATTERN_WINDOW, 1.0)
    
    def predict_bounce_distribution(self, impact_data: Dict[str, Any]) -> List[int]:
        """Predict most likely pockets after bounce"""
        key = self._get_pattern_key(impact_data)
        
        # Check if we have enough data
        if key not in self.bounce_patterns or self.pattern_confidence.get(key, 0) < CONFIDENCE_THRESHOLD:
            # Use physics-based prediction
            return self._physics_based_prediction(impact_data)
        
        # Use statistical prediction
        pattern = self.bounce_patterns[key]
        
        # Sort by frequency
        sorted_offsets = sorted(pattern.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 4 predictions
        predicted_pocket = impact_data['predicted']
        direction = impact_data['direction']
        jump_pockets = []
        
        for offset, _ in sorted_offsets[:4]:
            if direction.lower() == "cw":
                idx = normalize_index(POCKET_POSITION[predicted_pocket] + offset)
            else:
                idx = normalize_index(POCKET_POSITION[predicted_pocket] - offset)
            jump_pockets.append(WHEEL_SEQUENCE[idx])
        
        # Ensure we have 4 predictions
        while len(jump_pockets) < 4:
            jump_pockets.extend(self._get_neighboring_pockets(predicted_pocket, len(jump_pockets)))
        
        return jump_pockets[:4]
    
    def _get_pattern_key(self, impact_data: Dict[str, Any]) -> str:
        """Create key for pattern matching"""
        # Discretize continuous values
        speed_category = int(impact_data['omega_ball'] * 2)  # 0.5 rad/s bins
        wheel_speed = int(impact_data.get('omega_wheel', 0) * 4)  # 0.25 rad/s bins
        
        return f"{speed_category}_{wheel_speed}_{impact_data['direction']}"
    
    def _physics_based_prediction(self, impact_data: Dict[str, Any]) -> List[int]:
        """Fallback physics-based prediction"""
        predicted = impact_data['predicted']
        omega_ball = abs(impact_data['omega_ball'])
        
        # Estimate bounce distance based on impact velocity
        if omega_ball > 4.0:  # High speed
            offsets = [2, -2, 4, -4]
        elif omega_ball > 2.5:  # Medium speed
            offsets = [1, -1, 2, -2]
        else:  # Low speed
            offsets = [1, 2, 3, -1]
        
        jump_pockets = []
        for offset in offsets:
            idx = normalize_index(POCKET_POSITION[predicted] + offset)
            jump_pockets.append(WHEEL_SEQUENCE[idx])
        
        return jump_pockets
    
    def _get_neighboring_pockets(self, pocket: int, skip: int = 0) -> List[int]:
        """Get neighboring pockets"""
        idx = POCKET_POSITION[pocket]
        neighbors = []
        
        for i in range(1, 6):
            if len(neighbors) >= 4 - skip:
                break
            neighbors.append(WHEEL_SEQUENCE[normalize_index(idx + i)])
            if len(neighbors) >= 4 - skip:
                break
            neighbors.append(WHEEL_SEQUENCE[normalize_index(idx - i)])
        
        return neighbors[:4 - skip]

# ─────────────────────────  Learning Control System  ──────────────────────
def should_stop_learning() -> bool:
    """
    Intelligent system to determine when to stop collecting data
    Returns True when optimal accuracy is reached or no improvement detected
    """
    records = read_dataset()
    
    # Minimum dataset requirement
    if len(records) < 200:
        return False
    
    # Analyze recent performance
    recent_50 = records[-50:]
    recent_errors = []
    direct_hits = 0
    jump_hits = 0
    
    for r in recent_50:
        if r.get("error_slots") and r.get("winning_number"):
            try:
                error = int(r["error_slots"])
                recent_errors.append(error)
                
                if error == 0:
                    direct_hits += 1
                
                pattern = r.get("bounce_pattern", "")
                if pattern == "direct_hit" or pattern == "jump_hit":
                    jump_hits += 1
            except:
                continue
    
    if not recent_errors:
        return False
    
    # Calculate metrics
    avg_error = sum(recent_errors) / len(recent_errors)
    accuracy_rate = (direct_hits + jump_hits) / len(recent_errors)
    
    # Criterion 1: Optimal accuracy achieved
    if avg_error <= 3.0 and accuracy_rate >= 0.85:
        print("🎯 OPTIMAL ACCURACY REACHED")
        print(f"   Average error: {avg_error:.1f} pockets")
        print(f"   Accuracy rate: {accuracy_rate*100:.1f}%")
        print("   ➡️ Learning stopped - Model is optimized")
        return True
    
    # Criterion 2: Check for improvement plateau
    if len(records) >= 400:
        # Compare two 100-record windows
        older_100 = records[-400:-200]
        newer_100 = records[-200:]
        
        older_errors = []
        newer_errors = []
        
        for r in older_100:
            if r.get("error_slots"):
                try:
                    older_errors.append(int(r["error_slots"]))
                except:
                    pass
        
        for r in newer_100:
            if r.get("error_slots"):
                try:
                    newer_errors.append(int(r["error_slots"]))
                except:
                    pass
        
        if older_errors and newer_errors:
            old_avg = sum(older_errors) / len(older_errors)
            new_avg = sum(newer_errors) / len(newer_errors)
            improvement = old_avg - new_avg
            
            # No significant improvement
            if improvement < 0.5:
                print("📊 LEARNING PLATEAU DETECTED")
                print(f"   Old average: {old_avg:.1f} pockets")
                print(f"   New average: {new_avg:.1f} pockets")
                print(f"   Improvement: {improvement:.1f} pockets")
                print("   ➡️ Learning stopped - No further improvement")
                return True
    
    # Criterion 3: Maximum dataset size
    if len(records) >= MAX_DATASET_SIZE - 50:
        print("📦 APPROACHING DATASET LIMIT")
        print(f"   Current records: {len(records)}")
        print(f"   Maximum allowed: {MAX_DATASET_SIZE}")
        
        # If we're also performing well, stop
        if avg_error <= 5.0:
            print("   ➡️ Learning stopped - Good accuracy + size limit")
            return True
    
    return False

def get_learning_status() -> Dict[str, Any]:
    """Get detailed learning status"""
    records = read_dataset()
    
    if not records:
        return {
            "status": "collecting",
            "progress": 0,
            "message": "Collecting initial data"
        }
    
    # Calculate metrics
    recent_errors = []
    for r in records[-50:]:
        if r.get("error_slots"):
            try:
                recent_errors.append(int(r["error_slots"]))
            except:
                pass
    
    avg_error = sum(recent_errors) / len(recent_errors) if recent_errors else float('inf')
    
    # Determine status
    if should_stop_learning():
        status = "optimized"
        message = "Model optimized - No longer collecting data"
    elif len(records) < 50:
        status = "initial"
        message = f"Initial learning phase ({len(records)}/50)"
    elif len(records) < 200:
        status = "training"
        message = f"Active training ({len(records)}/200)"
    else:
        status = "refining"
        message = f"Refining model (error: {avg_error:.1f})"
    
    return {
        "status": status,
        "records": len(records),
        "average_error": round(avg_error, 1) if avg_error != float('inf') else "N/A",
        "message": message,
        "learning_active": not should_stop_learning()
    }

# ─────────────────────────  Performance Metrics  ──────────────────────────
class PerformanceTracker:
    """Track prediction accuracy and improvement"""
    
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        self.total_predictions = 0
        self.direct_hits = 0
        self.jump_hits = 0
        self.error_history = deque(maxlen=50)
        self.improvement_baseline = None
    
    def update(self, predicted: int, jumps: List[int], actual: int, direction: str):
        """Update metrics with new result"""
        self.total_predictions += 1
        
        # Check hits
        if predicted == actual:
            self.direct_hits += 1
        if actual in jumps:
            self.jump_hits += 1
        
        # Calculate error
        error = abs(pocket_distance(predicted, actual, direction))
        self.error_history.append(error)
        
        # Set baseline after first 20 predictions
        if self.total_predictions == 20:
            self.improvement_baseline = self.get_average_error()
    
    def get_average_error(self) -> float:
        """Get average prediction error"""
        if not self.error_history:
            return float('inf')
        return sum(self.error_history) / len(self.error_history)
    
    def get_improvement_percentage(self) -> float:
        """Calculate improvement from baseline"""
        if not self.improvement_baseline or self.total_predictions < 30:
            return 0.0
        
        current_error = self.get_average_error()
        if self.improvement_baseline == 0:
            return 0.0
        
        improvement = (self.improvement_baseline - current_error) / self.improvement_baseline * 100
        return max(0, improvement)

# ─────────────────────────  Main Server  ──────────────────────────────────
app = FastAPI(title="Professional Roulette Prediction Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Global state
bounce_predictor = BouncePredictor()
performance_tracker = PerformanceTracker()
pending_predictions: Dict[str, Dict[str, Any]] = {}

@app.on_event("startup")
def startup():
    """Initialize server"""
    initialize_csv()
    
    # Load historical patterns
    records = read_dataset()
    for record in records:
        if record.get("winning_number") and record.get("predicted_number"):
            try:
                impact_data = {
                    'predicted': int(record["predicted_number"]),
                    'omega_ball': float(record.get("omega_ball", 0)),
                    'omega_wheel': float(record.get("omega_wheel", 0)),
                    'direction': record.get("direction", "cw")
                }
                bounce_predictor.update_patterns(impact_data, int(record["winning_number"]))
                
                # Update performance metrics
                jumps = [int(x) for x in record.get("jump_numbers", "").split(",") if x]
                performance_tracker.update(
                    int(record["predicted_number"]),
                    jumps,
                    int(record["winning_number"]),
                    record.get("direction", "cw")
                )
            except:
                continue
    
    print(f"Server initialized with {len(records)} historical records")
    print(f"Average error: {performance_tracker.get_average_error():.1f} pockets")

@app.get("/")
def root():
    """Server status with learning information"""
    records = read_dataset()
    learning_status = get_learning_status()
    
    return {
        "status": "Professional Roulette Prediction Server",
        "version": "2.0",
        "dataset_size": len(records),
        "average_error": round(performance_tracker.get_average_error(), 1),
        "improvement": round(performance_tracker.get_improvement_percentage(), 1),
        "learning": learning_status
    }

@app.post("/predict")
def predict(request: PredictRequest):
    """Generate prediction with bounce modeling"""
    try:
        # Validate crossing data quality
        validation_result = validate_crossings(request.crossings)
        if not validation_result["valid"]:
            print(f"⚠️ Data validation failed: {validation_result['reason']}")
            print(f"Deviations: {validation_result['deviations']}")
            
            # Still make prediction but don't store if quality is too poor
            if validation_result["quality_score"] < 0.3:
                print("❌ Data quality too poor for prediction")
                return {
                    "ok": False, 
                    "error": "Data quality insufficient",
                    "validation": validation_result
                }
        
        # Extract crossing data
        times = [c.t for c in request.crossings]
        ball_angles = [c.theta for c in request.crossings]
        wheel_angles = [c.phi for c in request.crossings]
        
        # Smooth data
        ball_angles_smooth = smooth_data(ball_angles)
        wheel_angles_smooth = smooth_data(wheel_angles)
        
        # Fit trajectories
        try:
            _, omega_ball, alpha_ball = fit_trajectory(times, ball_angles_smooth)
        except:
            omega_ball = (ball_angles[-1] - ball_angles[0]) / (times[-1] - times[0])
            alpha_ball = -0.5
        
        try:
            _, omega_wheel, alpha_wheel = fit_trajectory(times, wheel_angles_smooth)
        except:
            omega_wheel = 0
            alpha_wheel = 0
        
        # Calculate impact time
        t_impact = calculate_impact_time(abs(omega_ball), alpha_ball)
        
        # Predict positions at impact
        t_total = times[-1] - times[0] + t_impact
        ball_at_impact = ball_angles[0] + omega_ball * t_total + 0.5 * alpha_ball * t_total**2
        wheel_at_impact = wheel_angles[0] + omega_wheel * t_total + 0.5 * alpha_wheel * t_total**2
        
        # Calculate relative position
        relative_angle = normalize_angle(ball_at_impact - wheel_at_impact)
        predicted_number = angle_to_pocket(relative_angle)
        
        # Prepare impact data
        impact_data = {
            'predicted': predicted_number,
            'omega_ball': omega_ball,
            'omega_wheel': omega_wheel,
            'direction': request.direction
        }
        
        # Get bounce predictions
        jump_numbers = bounce_predictor.predict_bounce_distribution(impact_data)
        
        # Calculate current metrics
        avg_error = performance_tracker.get_average_error()
        improvement = performance_tracker.get_improvement_percentage()
        
        # Generate round ID
        round_id = str(uuid.uuid4())
        
        # Store for validation
        pending_predictions[round_id] = {
            "ts_start": request.ts_start or int(time.time() * 1000),
            "direction": request.direction,
            "theta_zero": request.theta_zero,
            "times": times,
            "ball_angles": ball_angles,
            "wheel_angles": wheel_angles,
            "omega_ball": omega_ball,
            "alpha_ball": alpha_ball,
            "omega_wheel": omega_wheel,
            "alpha_wheel": alpha_wheel,
            "predicted": predicted_number,
            "jump_numbers": jump_numbers,
            "ts_predict": int(time.time() * 1000),
            "quality_score": validation_result.get("quality_score", 1.0),
            "store_quality": validation_result.get("store_quality", True)
        }
        
        # Log to console with learning status
        print("✅ Prediction ready")
        console_output = {
            "predicted_number": predicted_number,
            "jump_numbers": jump_numbers,
            "accuracy": {
                "error_margin": int(avg_error) if avg_error != float('inf') else "N/A",
                "improvement": round(improvement, 1)
            }
        }
        
        # Add quality warning if needed
        if validation_result.get("quality_score", 1.0) < 0.7:
            console_output["data_quality"] = f"{validation_result['quality_score']*100:.0f}%"
            print("⚠️ Low quality data - results may be less accurate")
        
        # Add learning status
        learning_status = get_learning_status()
        if learning_status["status"] == "optimized":
            console_output["model_status"] = "OPTIMIZED ✨"
        
        print(json.dumps(console_output, indent=2))
        
        # Return response
        return {
            "ok": True,
            "round_id": round_id,
            "prediction": predicted_number,
            "jump_numbers": jump_numbers,
            "accuracy": {
                "error_margin": int(avg_error) if avg_error != float('inf') else "N/A",
                "improvement": round(improvement, 1)
            },
            "dataset_rows": len(read_dataset()),
            "data_quality": f"{validation_result.get('quality_score', 1.0)*100:.0f}%"
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/log_winner")
def log_winner(request: LogWinnerRequest):
    """Log actual result and update models"""
    try:
        # Get pending prediction
        prediction_data = pending_predictions.pop(request.round_id, None)
        if not prediction_data:
            return {"ok": True, "ignored": True, "reason": "no_matching_prediction"}
        
        # Check if learning should stop
        if should_stop_learning():
            learning_status = get_learning_status()
            print(f"🛑 Learning stopped: {learning_status['message']}")
            
            # Still update metrics but don't save
            if request.winning_number == prediction_data['predicted']:
                print(f"✅ Direct hit! (not saved - learning complete)")
            elif request.winning_number in prediction_data['jump_numbers']:
                print(f"✅ Jump hit! (not saved - learning complete)")
            else:
                error = abs(pocket_distance(prediction_data['predicted'], request.winning_number, prediction_data['direction']))
                print(f"❌ Miss by {error} pockets (not saved - learning complete)")
            
            return {
                "ok": True,
                "ignored": True,
                "reason": "learning_complete",
                "learning_status": learning_status
            }
        
        # Check if data quality was good enough to store
        if not prediction_data.get("store_quality", True):
            print(f"⚠️ Skipping storage due to poor data quality (score: {prediction_data.get('quality_score', 0)*100:.0f}%)")
            return {
                "ok": True,
                "ignored": True,
                "reason": "poor_data_quality",
                "quality_score": f"{prediction_data.get('quality_score', 0)*100:.0f}%"
            }
        
        # Update bounce patterns
        impact_data = {
            'predicted': prediction_data['predicted'],
            'omega_ball': prediction_data['omega_ball'],
            'omega_wheel': prediction_data['omega_wheel'],
            'direction': prediction_data['direction']
        }
        bounce_predictor.update_patterns(impact_data, request.winning_number)
        
        # Update performance metrics
        performance_tracker.update(
            prediction_data['predicted'],
            prediction_data['jump_numbers'],
            request.winning_number,
            prediction_data['direction']
        )
        
        # Calculate error
        error = abs(pocket_distance(prediction_data['predicted'], request.winning_number, prediction_data['direction']))
        
        # Determine bounce pattern
        if request.winning_number == prediction_data['predicted']:
            pattern = "direct_hit"
        elif request.winning_number in prediction_data['jump_numbers']:
            pattern = "jump_hit"
        else:
            pattern = f"miss_{error}"
        
        # Save to CSV
        record = {
            "round_id": request.round_id,
            "ts_start": str(prediction_data["ts_start"]),
            "direction": prediction_data["direction"],
            "theta_zero": f"{prediction_data['theta_zero']:.6f}",
            "ball_t1": f"{prediction_data['times'][0]:.6f}",
            "ball_theta1": f"{prediction_data['ball_angles'][0]:.6f}",
            "ball_phi1": f"{prediction_data['wheel_angles'][0]:.6f}",
            "ball_t2": f"{prediction_data['times'][1]:.6f}",
            "ball_theta2": f"{prediction_data['ball_angles'][1]:.6f}",
            "ball_phi2": f"{prediction_data['wheel_angles'][1]:.6f}",
            "ball_t3": f"{prediction_data['times'][2]:.6f}",
            "ball_theta3": f"{prediction_data['ball_angles'][2]:.6f}",
            "ball_phi3": f"{prediction_data['wheel_angles'][2]:.6f}",
            "omega_ball": f"{prediction_data['omega_ball']:.6f}",
            "alpha_ball": f"{prediction_data['alpha_ball']:.6f}",
            "omega_wheel": f"{prediction_data['omega_wheel']:.6f}",
            "alpha_wheel": f"{prediction_data['alpha_wheel']:.6f}",
            "predicted_number": str(prediction_data["predicted"]),
            "jump_numbers": ",".join(map(str, prediction_data["jump_numbers"])),
            "ts_predict": str(prediction_data["ts_predict"]),
            "winning_number": str(request.winning_number),
            "ts_winner": str(request.timestamp or int(time.time() * 1000)),
            "error_slots": str(error),
            "bounce_pattern": pattern
        }
        
        append_record(record)
        maintain_dataset_size()
        
        # Log result
        hit_type = "DIRECT HIT" if pattern == "direct_hit" else "JUMP HIT" if pattern == "jump_hit" else "MISS"
        print(f"Result: {hit_type} - Predicted: {prediction_data['predicted']}, Actual: {request.winning_number}, Error: {error}")
        
        return {
            "ok": True,
            "stored": True,
            "hit_type": pattern,
            "error_slots": error,
            "current_accuracy": {
                "average_error": round(performance_tracker.get_average_error(), 1),
                "improvement": round(performance_tracker.get_improvement_percentage(), 1)
            }
        }
        
    except Exception as e:
        print(f"Error logging winner: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/statistics")
def get_statistics():
    """Get detailed performance statistics"""
    records = read_dataset()
    learning_status = get_learning_status()
    
    # Calculate phase-specific metrics
    phase_metrics = {
        "initial_50": {"errors": [], "period": "Records 1-50"},
        "middle_phase": {"errors": [], "period": "Records 51-150"},
        "recent_50": {"errors": [], "period": f"Records {max(1, len(records)-49)}-{len(records)}"}
    }
    
    # Fill phase metrics
    for i, r in enumerate(records):
        if r.get("error_slots"):
            try:
                error = int(r["error_slots"])
                if i < 50:
                    phase_metrics["initial_50"]["errors"].append(error)
                elif i < 150:
                    phase_metrics["middle_phase"]["errors"].append(error)
                if i >= len(records) - 50:
                    phase_metrics["recent_50"]["errors"].append(error)
            except:
                pass
    
    # Calculate averages
    for phase in phase_metrics.values():
        if phase["errors"]:
            phase["average_error"] = round(sum(phase["errors"]) / len(phase["errors"]), 1)
        else:
            phase["average_error"] = "N/A"
        phase["sample_size"] = len(phase["errors"])
    
    return {
        "total_predictions": performance_tracker.total_predictions,
        "direct_hits": performance_tracker.direct_hits,
        "jump_hits": performance_tracker.jump_hits,
        "hit_rate": {
            "direct": round(performance_tracker.direct_hits / max(1, performance_tracker.total_predictions) * 100, 1),
            "jumps": round(performance_tracker.jump_hits / max(1, performance_tracker.total_predictions) * 100, 1)
        },
        "average_error": round(performance_tracker.get_average_error(), 1),
        "improvement": round(performance_tracker.get_improvement_percentage(), 1),
        "dataset_size": len(records),
        "confidence_level": "High" if len(records) > 200 else "Medium" if len(records) > 50 else "Low",
        "learning_status": learning_status,
        "performance_by_phase": phase_metrics
    }

if __name__ == "__main__":
    print("Starting Professional Roulette Prediction Server")
    print(f"Data location: {DATA_PATH}")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
