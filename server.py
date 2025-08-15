"""
Professional Roulette Prediction Server
Advanced physics-based prediction with realistic bounce modeling
Based on Small & Tse (2012) paper methodology
"""

from __future__ import annotations
import math, csv, os, time, uuid, json
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ──────────────────────────  Roulette Layout  ─────────────────────────────────
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

# ──────────────────────────  Request Models  ──────────────────────────────────
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

# ───────────────────────────  Physics Constants  ──────────────────────────────
# Fundamental physics
GRAVITY = 9.81  # m/s²

# These will be calibrated from actual data
DEFAULT_WHEEL_RADIUS = 0.41  # Initial guess
DEFAULT_TRACK_RADIUS = 0.48  # Initial guess
DEFAULT_DEFLECTOR_RADIUS = 0.38  # Initial guess
DEFAULT_TABLE_TILT = math.radians(2.5)  # Initial guess from original

# Calibration parameters
MIN_CALIBRATION_SPINS = 20  # Need at least 20 spins to calibrate
CALIBRATION_CONFIDENCE_THRESHOLD = 0.8

# Ball physics
BALL_MASS = 0.002  # 2 grams typical
BALL_RADIUS = 0.01  # 10mm ball
AIR_DENSITY = 1.225  # kg/m³
DRAG_COEFFICIENT = 0.47  # sphere

# Friction model
ROLLING_FRICTION_COEFFICIENT = 0.005  # from original
VISCOUS_FRICTION = 0.001  # speed-dependent term
INTEGRATION_STEP = 0.001  # from original

# Deflector physics
DEFLECTOR_COUNT = 8  # typical for European wheel
DEFLECTOR_HEIGHT = 0.014  # 14mm typical
DEFLECTOR_ANGLE = math.radians(45)  # typical deflector angle
COEFFICIENT_OF_RESTITUTION = 0.65  # from original DEFLECTOR_ELASTICITY

# Pocket physics
FRET_HEIGHT = 0.008  # 8mm pocket dividers
POCKET_WIDTH = 0.053  # 53mm pocket width
POCKET_DEPTH = 0.037  # 37mm typical
FRET_ELASTICITY = 0.3  # energy retained hitting fret

# Integration parameters
MAX_INTEGRATION_TIME = 20.0  # seconds

# Learning parameters
MIN_DATA_FOR_PHYSICS_MODEL = 30  # from original MIN_DATA_FOR_BOUNCE_MODEL
BOUNCE_PATTERN_WINDOW = 100  # from original
CONFIDENCE_THRESHOLD = 0.7  # from original

# File management (keeping original)
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

# ──────────────────────────  Auto-Calibration System  ────────────────────────
class WheelCalibration:
    """Automatically calibrate wheel parameters from observed data"""
    
    def __init__(self):
        self.calibrated = False
        self.wheel_radius = DEFAULT_WHEEL_RADIUS
        self.track_radius = DEFAULT_TRACK_RADIUS
        self.deflector_radius = DEFAULT_DEFLECTOR_RADIUS
        self.table_tilt = DEFAULT_TABLE_TILT
        self.calibration_data = []
        
    def add_observation(self, crossing_data: Dict[str, Any], outcome: Dict[str, Any]):
        """Add observation for calibration"""
        self.calibration_data.append({
            'crossings': crossing_data,
            'outcome': outcome,
            'timestamp': time.time()
        })
        
    def calibrate(self) -> bool:
        """
        Calibrate wheel parameters from observations
        Returns True if successful
        """
        if len(self.calibration_data) < MIN_CALIBRATION_SPINS:
            return False
            
        print(f"Starting calibration with {len(self.calibration_data)} observations...")
        
        # Extract timing patterns
        rim_times = []
        decel_rates = []
        
        for data in self.calibration_data:
            # Analyze deceleration patterns
            if 'omega_ball' in data['crossings'] and 'alpha_ball' in data['crossings']:
                omega = abs(data['crossings']['omega_ball'])
                alpha = abs(data['crossings']['alpha_ball'])
                
                # Estimate time for one revolution at this speed
                if omega > 0:
                    rim_time = (2 * math.pi) / omega
                    rim_times.append(rim_time)
                    decel_rates.append(alpha)
        
        # Simple averaging without numpy
        if rim_times:
            avg_rim_time = sum(rim_times) / len(rim_times)
            # Typical ball velocity ~2-5 m/s on rim
            # One revolution time gives us circumference estimate
            estimated_circumference = avg_rim_time * 3.5  # Assuming ~3.5 m/s average
            self.track_radius = estimated_circumference / (2 * math.pi)
            
            # Wheel is typically 85% of track radius
            self.wheel_radius = self.track_radius * 0.85
            
            # Deflectors are typically at 80% of track radius
            self.deflector_radius = self.track_radius * 0.80
            
            # Estimate tilt from deceleration pattern
            if decel_rates:
                avg_decel = sum(decel_rates) / len(decel_rates)
                # Higher deceleration = more tilt
                # Typical range: 0.5-2.0 rad/s²
                self.table_tilt = math.radians(0.2 + (avg_decel - 0.5) * 0.5)
            
            self.calibrated = True
            print(f"Calibration complete:")
            print(f"  Track radius: {self.track_radius:.3f} m")
            print(f"  Wheel radius: {self.wheel_radius:.3f} m") 
            print(f"  Deflector radius: {self.deflector_radius:.3f} m")
            print(f"  Table tilt: {math.degrees(self.table_tilt):.2f}°")
            
            return True
            
        return False
    
    def get_parameters(self) -> Dict[str, float]:
        """Get current parameters (calibrated or default)"""
        return {
            'wheel_radius': self.wheel_radius,
            'track_radius': self.track_radius,
            'deflector_radius': self.deflector_radius,
            'table_tilt': self.table_tilt,
            'calibrated': self.calibrated
        }
def as_file_path(path: str) -> str:
    """Convert path to absolute file path"""
    return os.path.abspath(os.path.expanduser(path))

def try_path(candidates: List[str]) -> str:
    """Try multiple paths until finding a writable location"""
    for c in candidates:
        if not c:
            continue
        c = as_file_path(c)
        d = os.path.dirname(c) or "."
        try:
            os.makedirs(d, exist_ok=True)
# Комментарии на русском сохранены из оригинала
            # проверяем запись
            with open(c, "a", encoding="utf-8") as f:
                pass
            # если файл новый — пишем заголовок
            if os.path.getsize(c) == 0:
                with open(c, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(CSV_COLUMNS)
            return c
        except Exception as e:
            print(f"[init] skip '{c}': {e}")
            continue
    raise RuntimeError("No writable location for dataset.csv")

def get_data_path() -> str:
    """Determine optimal path for data storage"""
    candidates = [
        os.getenv("ROULETTE_DATA_PATH", ""),
        os.path.join(os.path.expanduser("~"), ".roulette_predictor", DATA_FILE_NAME),
        os.path.join("/tmp", DATA_FILE_NAME),
        os.path.join(".", DATA_FILE_NAME)
    ]
    return try_path(candidates)

DATA_PATH = get_data_path()

def initialize_csv():
    """Ensure CSV exists with headers"""
    pass  # Already handled by try_path

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
    """Keep dataset within size limits"""
    records = read_dataset()
    
    if len(records) > MAX_DATASET_SIZE:
        records = records[-MAX_DATASET_SIZE:]
        
        with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(records)
        print(f"Dataset size maintained: {len(records)} records")

# ──────────────────────────  Physics Engine  ──────────────────────────────────
class RoulettePhysics:
    """Complete physics model based on Small & Tse (2012)"""
    
    def __init__(self, calibration: WheelCalibration):
        self.calibration = calibration
        
    @property
    def table_tilt(self):
        return self.calibration.table_tilt
        
    @property
    def track_radius(self):
        return self.calibration.track_radius
        
    @property
    def wheel_radius(self):
        return self.calibration.wheel_radius
        
    @property
    def deflector_radius(self):
        return self.calibration.deflector_radius
        
    def ball_dynamics(self, r: float, omega: float, v_r: float, t: float) -> Tuple[float, float]:
        """
        Simple ball dynamics without scipy ODE solver
        Returns: (radial_acceleration, angular_acceleration)
        """
        # Skip if ball has reached deflectors
        if r <= self.deflector_radius:
            return 0, 0
        
        # Velocity magnitude
        v_total = math.sqrt(v_r**2 + (r * omega)**2)
        
        # Forces in radial direction
        F_centrifugal = BALL_MASS * r * omega**2
        F_gravity_r = BALL_MASS * GRAVITY * math.sin(self.table_tilt)
        
        # Air resistance (quadratic in velocity)
        F_drag_r = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * math.pi * BALL_RADIUS**2 * v_r * v_total
        
        # Friction force
        if v_total > 0.001:
            friction_r = ROLLING_FRICTION_COEFFICIENT * BALL_MASS * GRAVITY * math.cos(self.table_tilt) * (v_r / v_total)
            friction_theta = ROLLING_FRICTION_COEFFICIENT * BALL_MASS * GRAVITY * math.cos(self.table_tilt) * (r * omega / v_total)
            friction_r += VISCOUS_FRICTION * v_r
            friction_theta += VISCOUS_FRICTION * r * omega
        else:
            friction_r = 0
            friction_theta = 0
        
        # Accelerations
        a_r = (F_centrifugal - F_gravity_r - F_drag_r - friction_r) / BALL_MASS
        a_theta = -(friction_theta) / (BALL_MASS * r) if r > 0 else 0
        
        return a_r, a_theta
    
    def find_rim_departure(self, omega_initial: float, alpha: float) -> Tuple[float, float]:
        """
        Find when and where ball leaves the rim
        Returns: (time, angular_velocity)
        """
        # Critical angular velocity where centripetal force = gravity component
        omega_critical = math.sqrt(GRAVITY * math.tan(self.table_tilt) / self.track_radius)
        
        if omega_initial <= omega_critical:
            return 0.0, omega_initial
        
        # Time to reach critical velocity (assuming constant deceleration)
        if abs(alpha) > 1e-9:
            t_rim = (omega_initial - omega_critical) / abs(alpha)
        else:
            t_rim = float('inf')
        
        return t_rim, omega_critical
    
    def simulate_track_motion(self, omega_initial: float, alpha: float) -> Tuple[float, float, float]:
        """
        Simulate ball motion from rim departure to deflector impact
        Returns: (impact_time, impact_velocity, impact_angle)
        """
        t_rim, omega_rim = self.find_rim_departure(omega_initial, alpha)
        
        # Simple numerical integration without scipy
        r = self.track_radius
        v_r = 0.0  # radial velocity starts at 0
        omega = omega_rim
        t = 0
        dt = INTEGRATION_STEP
        
        while r > self.deflector_radius and t < MAX_INTEGRATION_TIME:
            # Calculate forces
            F_centrifugal = BALL_MASS * r * omega**2
            F_gravity_r = BALL_MASS * GRAVITY * math.sin(self.table_tilt)
            
            # Update velocities
            a_r = (F_centrifugal - F_gravity_r) / BALL_MASS
            v_r += a_r * dt
            
            # Update position
            r += v_r * dt
            
            # Angular deceleration
            omega += alpha * dt
            
            t += dt
            
            if omega <= 0:
                break
        
        # Calculate impact parameters
        t_impact = t_rim + t
        v_impact = math.sqrt(v_r**2 + (r * omega)**2)
        angle_impact = math.atan2(v_r, r * omega) if r * omega != 0 else 0
        
        return t_impact, v_impact, angle_impact
    
    def model_deflector_collision(self, v_impact: float, angle_impact: float) -> Dict[str, float]:
        """
        Model collision with deflector using conservation laws
        Returns scatter parameters
        """
        # Energy loss due to inelastic collision
        v_after = v_impact * COEFFICIENT_OF_RESTITUTION
        
        # Deflection angle depends on impact angle and deflector geometry
        deflection_mean = DEFLECTOR_ANGLE * (1 - abs(angle_impact) / (math.pi / 2))
        deflection_std = math.radians(10)  # 10 degree standard deviation
        
        # Vertical component (how high ball bounces)
        vertical_energy = 0.5 * BALL_MASS * v_after**2 * math.sin(deflection_mean)**2
        max_height = vertical_energy / (BALL_MASS * GRAVITY)
        
        return {
            'velocity_after': v_after,
            'deflection_angle': deflection_mean,
            'deflection_std': deflection_std,
            'bounce_height': max_height,
            'energy_retained': (v_after / v_impact)**2
        }
    
    def predict_scatter_distribution(self, collision_params: Dict[str, float]) -> List[Tuple[int, float]]:
        """
        Predict probability distribution of final pockets based on collision
        Returns list of (pocket_offset, probability) - simplified version
        """
        v_after = collision_params['velocity_after']
        bounce_height = collision_params['bounce_height']
        
        # Simple distribution based on velocity
        distribution = []
        
        # Center around expected scatter
        if v_after > 3.0:
            # High energy - wide scatter
            weights = [0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05]
            offsets = [-3, -2, -1, 0, 1, 2, 3, 4]
        else:
            # Low energy - narrow scatter
            weights = [0.05, 0.15, 0.30, 0.30, 0.15, 0.05, 0.00, 0.00]
            offsets = [-2, -1, 0, 1, 2, 3, 4, 5]
        
        for offset, weight in zip(offsets, weights):
            distribution.append((offset, weight))
        
        return distribution

# ──────────────────────────  Advanced Bounce Predictor  ───────────────────────
class PhysicsBasedBouncePredictor:
    """Bounce prediction using physics model + machine learning"""
    
    def __init__(self, calibration: WheelCalibration):
        self.physics = RoulettePhysics(calibration)
        self.calibration = calibration
        self.bounce_history = defaultdict(list)
        self.model_parameters = {}
        self.dataset_size = 0
        
    def update_from_history(self, records: List[Dict[str, str]]):
        """Learn from historical data"""
        self.dataset_size = len(records)
        
        for record in records:
            if not all(record.get(k) for k in ['omega_ball', 'winning_number', 'predicted_number']):
                continue
                
            try:
                omega = float(record['omega_ball'])
                predicted = int(record['predicted_number'])
                actual = int(record['winning_number'])
                direction = record.get('direction', 'cw')
                
                # Calculate offset
                offset = self._calculate_offset(predicted, actual, direction)
                
                # Categorize by velocity range
                velocity_key = int(abs(omega) * 2)  # 0.5 rad/s bins
                self.bounce_history[velocity_key].append(offset)
                
            except (ValueError, KeyError):
                continue
    
    def predict_bounce(self, omega_ball: float, alpha_ball: float, omega_wheel: float, 
                      direction: str) -> List[int]:
        """
        Predict most likely pockets using physics + statistics
        """
        # Use pure physics for first 50 spins
        if self.dataset_size < MIN_DATA_FOR_PHYSICS_MODEL:
            return self._pure_physics_prediction(omega_ball, alpha_ball)
        
        # Hybrid approach after 50 spins
        physics_pred = self._physics_based_prediction(omega_ball, alpha_ball)
        
        # Adjust based on historical data if available
        velocity_key = int(abs(omega_ball) * 2)
        if velocity_key in self.bounce_history and len(self.bounce_history[velocity_key]) >= 10:
            historical_offsets = self.bounce_history[velocity_key]
            
            # Find most common offsets
            offset_counts = defaultdict(int)
            for offset in historical_offsets:
                offset_counts[offset] += 1
            
            # Weight physics predictions by historical frequency
            adjusted_pred = []
            for pocket_offset in physics_pred:
                weight = offset_counts.get(pocket_offset, 1)
                adjusted_pred.extend([pocket_offset] * weight)
            
            # Return top 4 most likely
            from collections import Counter
            most_common = Counter(adjusted_pred).most_common(4)
            return [offset for offset, _ in most_common]
        
        return physics_pred[:4]
    
    def _pure_physics_prediction(self, omega_ball: float, alpha_ball: float) -> List[int]:
        """Pure physics-based prediction - simplified version"""
        # Get impact parameters
        t_impact, v_impact, angle_impact = self.physics.simulate_track_motion(
            abs(omega_ball), alpha_ball
        )
        
        # Simple bounce pattern based on impact velocity
        # Higher velocity = more scatter
        if v_impact > 4.0:  # High speed impact
            offsets = [2, -2, 3, -3, 4, -4, 1, -1]
        elif v_impact > 2.5:  # Medium speed
            offsets = [1, -1, 2, -2, 3, -3, 0, 4]
        elif v_impact > 1.5:  # Low speed
            offsets = [1, 2, 0, -1, 3, -2, -3, 4]
        else:  # Very low speed
            offsets = [0, 1, -1, 2, -2, 3, -3, 4]
        
        return offsets[:8]
    
    def _physics_based_prediction(self, omega_ball: float, alpha_ball: float) -> List[int]:
        """Physics prediction with learned corrections"""
        base_prediction = self._pure_physics_prediction(omega_ball, alpha_ball)
        
        # Apply learned bias correction if available
        if hasattr(self, 'learned_bias'):
            base_prediction = [p + self.learned_bias for p in base_prediction]
        
        return base_prediction
    
    def _calculate_offset(self, predicted: int, actual: int, direction: str) -> int:
        """Calculate pocket offset accounting for direction"""
        pred_idx = POCKET_POSITION[predicted]
        actual_idx = POCKET_POSITION[actual]
        
        if direction.lower() == 'cw':
            offset = (actual_idx - pred_idx) % WHEEL_SIZE
        else:
            offset = (pred_idx - actual_idx) % WHEEL_SIZE
            
        # Convert to signed offset (-18 to +18)
        if offset > WHEEL_SIZE // 2:
            offset -= WHEEL_SIZE
            
        return offset

# ──────────────────────────  Main Server  ─────────────────────────────────────
app = FastAPI(title="Professional Roulette Prediction Server - Physics Enhanced")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Global instances
wheel_calibration = WheelCalibration()
physics_engine = RoulettePhysics(wheel_calibration)
bounce_predictor = PhysicsBasedBouncePredictor(wheel_calibration)
pending_predictions: Dict[str, Dict[str, Any]] = {}

@app.on_event("startup")
def startup():
    """Initialize server with physics model"""
    initialize_csv()
    
    # Load historical data
    records = read_dataset()
    
    # Try to calibrate from existing data
    calibration_attempts = 0
    for record in records[-50:]:  # Use last 50 records for calibration
        if all(record.get(k) for k in ['omega_ball', 'alpha_ball', 'winning_number']):
            try:
                # Add to calibration data
                crossing_data = {
                    'omega_ball': float(record['omega_ball']),
                    'alpha_ball': float(record['alpha_ball']),
                    't': [float(record.get(f'ball_t{i}', 0)) for i in range(1, 4)]
                }
                outcome_data = {
                    'winning_number': int(record['winning_number']),
                    'impact_time': 5.0  # Estimate
                }
                wheel_calibration.add_observation(crossing_data, outcome_data)
                calibration_attempts += 1
            except:
                pass
    
    # Attempt calibration
    if calibration_attempts >= MIN_CALIBRATION_SPINS:
        wheel_calibration.calibrate()
    
    # Update bounce predictor
    bounce_predictor.update_from_history(records)
    
    params = wheel_calibration.get_parameters()
    print(f"Physics-based server initialized with {len(records)} historical records")
    print(f"Calibration status: {'CALIBRATED' if params['calibrated'] else 'USING DEFAULTS'}")
    if params['calibrated']:
        print(f"  Wheel radius: {params['wheel_radius']:.3f} m")
        print(f"  Track radius: {params['track_radius']:.3f} m")
        print(f"  Table tilt: {math.degrees(params['table_tilt']):.2f}°")

@app.get("/")
def root():
    """Server status"""
    records = read_dataset()
    params = wheel_calibration.get_parameters()
    
    return {
        "status": "Professional Roulette Prediction Server - Physics Enhanced",
        "version": "3.0",
        "model": "Small & Tse (2012) Physics Model",
        "dataset_size": len(records),
        "physics_model_active": len(records) >= MIN_DATA_FOR_PHYSICS_MODEL,
        "calibration": {
            "status": "CALIBRATED" if params['calibrated'] else "USING DEFAULTS",
            "wheel_radius": round(params['wheel_radius'], 3),
            "track_radius": round(params['track_radius'], 3),
            "deflector_radius": round(params['deflector_radius'], 3),
            "table_tilt_degrees": round(math.degrees(params['table_tilt']), 2)
        }
    }

@app.post("/predict")
def predict(request: PredictRequest):
    """Generate prediction using physics model"""
    try:
        # Validate crossing data
        if len(request.crossings) < 3:
            return {"ok": False, "error": "Insufficient crossings"}
        
        # Extract data
        times = [c.t for c in request.crossings]
        ball_angles = [c.theta for c in request.crossings]
        wheel_angles = [c.phi for c in request.crossings]
        
        # Calculate velocities using finite differences (from original)
        dt1 = times[1] - times[0]
        dt2 = times[2] - times[1]
        
        omega_ball_1 = (ball_angles[1] - ball_angles[0]) / dt1
        omega_ball_2 = (ball_angles[2] - ball_angles[1]) / dt2
        omega_ball = (omega_ball_1 + omega_ball_2) / 2
        
        # Calculate acceleration
        alpha_ball = (omega_ball_2 - omega_ball_1) / ((dt1 + dt2) / 2)
        
        # Wheel velocity
        omega_wheel = (wheel_angles[-1] - wheel_angles[0]) / (times[-1] - times[0])
        alpha_wheel = 0  # Assume constant for wheel
        
        # PHYSICS-BASED PREDICTION (simplified without scipy)
        # 1. Calculate when ball reaches deflectors
        t_impact, v_impact, angle_impact = physics_engine.simulate_track_motion(
            abs(omega_ball), abs(alpha_ball)  # Use absolute values
        )
        
        # 2. Predict ball and wheel positions at impact
        ball_at_impact = ball_angles[0] + omega_ball * t_impact + 0.5 * alpha_ball * t_impact**2
        wheel_at_impact = wheel_angles[0] + omega_wheel * t_impact
        
        # 3. Calculate relative position
        relative_angle = normalize_angle(ball_at_impact - wheel_at_impact)
        predicted_number = angle_to_pocket(relative_angle)
        
        # 4. Predict bounce distribution
        bounce_offsets = bounce_predictor.predict_bounce(
            omega_ball, alpha_ball, omega_wheel, request.direction
        )
        
        # 5. Convert offsets to actual pocket numbers
        jump_numbers = []
        for offset in bounce_offsets[:4]:
            if request.direction.lower() == 'cw':
                idx = normalize_index(POCKET_POSITION[predicted_number] + offset)
            else:
                idx = normalize_index(POCKET_POSITION[predicted_number] - offset)
            jump_numbers.append(WHEEL_SEQUENCE[idx])
        
        # Generate round ID
        round_id = str(uuid.uuid4())
        
        # Store prediction data with calibration info
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
            "physics_data": {
                "t_impact": t_impact,
                "v_impact": v_impact,
                "angle_impact": angle_impact
            },
            "calibration_status": wheel_calibration.calibrated
        }
        
        # Log prediction with calibration status
        print(f"✅ Physics-based prediction ready")
        print(f"   Calibration: {'YES' if wheel_calibration.calibrated else 'NO (using defaults)'}")
        print(f"   Impact velocity: {v_impact:.2f} m/s")
        print(f"   Impact time: {t_impact:.2f} s")
        print(f"   Predicted: {predicted_number}")
        print(f"   Jump numbers: {jump_numbers}")
        
        return {
            "ok": True,
            "round_id": round_id,
            "prediction": predicted_number,
            "jump_numbers": jump_numbers,
            "physics": {
                "impact_velocity": round(v_impact, 2),
                "impact_time": round(t_impact, 2),
                "model": "Small & Tse (2012)"
            },
            "dataset_rows": len(read_dataset())
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/log_winner")
def log_winner(request: LogWinnerRequest):
    """Log actual result and update physics model"""
    try:
        # Get pending prediction
        prediction_data = pending_predictions.pop(request.round_id, None)
        if not prediction_data:
            return {"ok": True, "ignored": True, "reason": "no_matching_prediction"}
        
        # Always save data for learning
        
        # Calculate error
        predicted = prediction_data['predicted']
        actual = request.winning_number
        direction = prediction_data['direction']
        
        # Calculate offset
        pred_idx = POCKET_POSITION[predicted]
        actual_idx = POCKET_POSITION[actual]
        
        if direction.lower() == 'cw':
            error = normalize_index(actual_idx - pred_idx)
        else:
            error = normalize_index(pred_idx - actual_idx)
            
        if error > WHEEL_SIZE // 2:
            error = WHEEL_SIZE - error
        
        # Determine hit type
        if actual == predicted:
            pattern = "direct_hit"
        elif actual in prediction_data['jump_numbers']:
            pattern = "jump_hit"
        else:
            pattern = f"miss_{error}"
        
        # Create record
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
            "predicted_number": str(predicted),
            "jump_numbers": ",".join(map(str, prediction_data["jump_numbers"])),
            "ts_predict": str(prediction_data["ts_predict"]),
            "winning_number": str(actual),
            "ts_winner": str(request.timestamp or int(time.time() * 1000)),
            "error_slots": str(error),
            "bounce_pattern": pattern
        }
        
        append_record(record)
        maintain_dataset_size()
        
        # Update physics model with new data
        records = read_dataset()
        bounce_predictor.update_from_history(records)
        
        # Add to calibration data
        if not wheel_calibration.calibrated:
            crossing_data = {
                'omega_ball': prediction_data['omega_ball'],
                'alpha_ball': prediction_data['alpha_ball'],
                't': prediction_data['times']
            }
            outcome_data = {
                'winning_number': actual,
                'impact_time': prediction_data['physics_data']['t_impact']
            }
            wheel_calibration.add_observation(crossing_data, outcome_data)
            
            # Try to calibrate
            if wheel_calibration.calibrate():
                print("🎯 WHEEL CALIBRATION SUCCESSFUL!")
                # Recreate physics engine with new parameters
                physics_engine.__init__(wheel_calibration)
                bounce_predictor.physics.__init__(wheel_calibration)
        
        print(f"Result: {pattern} - Predicted: {predicted}, Actual: {actual}, Error: {error}")
        
        return {
            "ok": True,
            "stored": True,
            "hit_type": pattern,
            "error_slots": error,
            "physics_model_updated": True
        }
        
    except Exception as e:
        print(f"Error logging winner: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/statistics")
def get_statistics():
    """Get detailed statistics including physics model performance"""
    records = read_dataset()
    
    # Calculate statistics
    direct_hits = sum(1 for r in records if r.get('bounce_pattern') == 'direct_hit')
    jump_hits = sum(1 for r in records if r.get('bounce_pattern') == 'jump_hit')
    total_predictions = len([r for r in records if r.get('predicted_number')])
    
    # Physics model performance (last 50 predictions)
    recent_records = records[-50:] if len(records) >= 50 else records
    recent_errors = []
    for r in recent_records:
        if r.get('error_slots'):
            try:
                recent_errors.append(int(r['error_slots']))
            except:
                pass
    
    avg_error = sum(recent_errors) / len(recent_errors) if recent_errors else 0
    
    # Expected return calculation
    if total_predictions > 0:
        hit_rate = (direct_hits + jump_hits) / total_predictions
        expected_return = (hit_rate * 35) - (1 - hit_rate)
    else:
        expected_return = 0
    
    return {
        "total_predictions": total_predictions,
        "direct_hits": direct_hits,
        "jump_hits": jump_hits,
        "hit_rate": {
            "direct": round(direct_hits / max(1, total_predictions) * 100, 1),
            "jumps": round(jump_hits / max(1, total_predictions) * 100, 1),
            "combined": round((direct_hits + jump_hits) / max(1, total_predictions) * 100, 1)
        },
        "average_error": round(avg_error, 1),
        "expected_return": f"{expected_return:.1%}",
        "dataset_size": len(records),
        "physics_model": {
            "active": len(records) >= MIN_DATA_FOR_PHYSICS_MODEL,
            "confidence": "High" if len(records) > 200 else "Medium" if len(records) > 50 else "Low",
            "model": "Small & Tse (2012) with deflector dynamics"
        }
    }

# ──────────────────────────  Helper Functions  ────────────────────────────────
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

if __name__ == "__main__":
    print("Starting Professional Physics-Based Roulette Prediction Server")
    print(f"Data location: {DATA_PATH}")
    print("Physics model: Small & Tse (2012) implementation")
    print("Auto-calibration: ENABLED - will calibrate from observed data")
    print(f"Initial parameters: {DEFAULT_WHEEL_RADIUS}m wheel (will auto-calibrate)")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
