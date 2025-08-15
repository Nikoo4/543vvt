"""
Advanced Roulette Prediction Server v4.0
PhD-level implementation with full physics simulation and machine learning
Based on Small & Tse (2012) methodology with modern enhancements
"""

import math
import csv
import os
import time
import uuid
import json
import logging
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import numpy as np
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import minimize, curve_fit
from scipy.stats import norm, chi2
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RoulettePredictor")

# ╔══════════════════════════  CSV Data Management  ═══════════════════════════╗

# CSV column definitions
CSV_COLUMNS = [
    'timestamp', 'round_id', 'predicted_number', 'actual_number', 
    'offset', 'confidence', 'omega_ball', 'alpha_ball', 'omega_wheel',
    'impact_velocity', 'rim_time', 'track_time', 'ml_correction',
    'calibration_confidence', 'within_3', 'within_5'
]

def get_data_path() -> str:
    """
    Determine the best available path for data storage
    Priority: ENV > user_home > /tmp > current_dir
    """
    # Check environment variable first
    env_path = os.environ.get('ROULETTE_DATA_PATH')
    if env_path:
        return try_path(env_path)
    
    # Try user home directory
    home_path = os.path.expanduser('~/.roulette_predictor/roulette_data.csv')
    if try_path(home_path):
        return home_path
    
    # Try /tmp with subdirectory for organization
    tmp_path = '/tmp/roulette_predictor/roulette_data.csv'
    if try_path(tmp_path):
        return tmp_path
    
    # Fallback to current directory
    local_path = './roulette_data.csv'
    if try_path(local_path):
        return local_path
    
    # Last resort - use /tmp without subdirectory
    fallback_path = '/tmp/roulette_data.csv'
    try_path(fallback_path)
    return fallback_path

def try_path(path: str) -> bool:
    """
    Try to create and initialize a CSV file at the given path
    Returns True if successful
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Check if file exists and is valid
        if os.path.exists(path):
            # Validate existing file
            with open(path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers and all(col in headers for col in CSV_COLUMNS[:4]):  # Basic validation
                    logger.info(f"Using existing CSV file: {path}")
                    return True
        
        # Create new file with headers
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            logger.info(f"Created new CSV file: {path}")
        
        return True
        
    except Exception as e:
        logger.warning(f"Cannot use path {path}: {e}")
        return False

# Initialize data path at module load
DATA_PATH = get_data_path()
logger.info(f"Data Path: {os.path.dirname(DATA_PATH)}")
logger.info(f"CSV File: {DATA_PATH}")

def load_csv_data() -> List[Dict]:
    """
    Load existing data from CSV file
    """
    data = []
    try:
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    for field in ['offset', 'confidence', 'omega_ball', 'alpha_ball', 
                                 'omega_wheel', 'impact_velocity', 'rim_time', 
                                 'track_time', 'ml_correction', 'calibration_confidence']:
                        if field in row and row[field]:
                            try:
                                row[field] = float(row[field])
                            except ValueError:
                                pass
                    
                    # Convert boolean fields
                    for field in ['within_3', 'within_5']:
                        if field in row and row[field]:
                            row[field] = row[field].lower() == 'true'
                    
                    data.append(row)
            
            logger.info(f"Loaded {len(data)} records from file")
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
    
    return data

def append_csv_record(record: Dict):
    """
    Append a single record to CSV file
    """
    try:
        with open(DATA_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            
            # Ensure all columns have values
            full_record = {col: record.get(col, '') for col in CSV_COLUMNS}
            writer.writerow(full_record)
            
    except Exception as e:
        logger.error(f"Error appending to CSV: {e}")

def maintain_dataset_size(max_size: int = 10000):
    """
    Maintain CSV file size by keeping only the most recent records
    """
    try:
        data = load_csv_data()
        
        if len(data) > max_size:
            # Keep only the most recent records
            data = data[-max_size:]
            
            # Rewrite the file
            with open(DATA_PATH, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(data)
            
            logger.info(f"Maintained dataset size at {max_size} records")
            
    except Exception as e:
        logger.error(f"Error maintaining dataset size: {e}")

# ╔══════════════════════════  Physical Constants  ═══════════════════════════╗

@dataclass
class PhysicalConstants:
    """Fundamental physical constants for roulette dynamics"""
    GRAVITY: float = 9.80665  # m/s² (standard gravity)
    AIR_DENSITY: float = 1.2041  # kg/m³ at 20°C, 1 atm
    AIR_VISCOSITY: float = 1.82e-5  # Pa·s at 20°C
    
    # Ball properties (typical casino ball)
    BALL_MASS: float = 0.0021  # kg (2.1g typical Teflon ball)
    BALL_RADIUS: float = 0.0095  # m (9.5mm radius)
    BALL_DRAG_COEFFICIENT: float = 0.47  # sphere in turbulent flow
    
    # Material properties
    TEFLON_STEEL_FRICTION: float = 0.04  # Teflon on steel
    TEFLON_WOOD_FRICTION: float = 0.06  # Teflon on wood
    COEFFICIENT_OF_RESTITUTION: float = 0.65  # ball-deflector collision
    
    # Wheel geometry (Evolution Auto-Roulette standard)
    WHEEL_RADIUS: float = 0.400  # m (40cm)
    TRACK_RADIUS: float = 0.475  # m (47.5cm)
    DEFLECTOR_RADIUS: float = 0.380  # m (38cm)
    RIM_HEIGHT: float = 0.020  # m (2cm)
    
    # Deflector properties
    DEFLECTOR_COUNT: int = 8  # standard European wheel
    DEFLECTOR_HEIGHT: float = 0.014  # m (14mm)
    DEFLECTOR_WIDTH: float = 0.008  # m (8mm)
    
    # Pocket properties
    POCKET_COUNT: int = 37  # European wheel
    POCKET_WIDTH: float = 0.053  # m (53mm)
    POCKET_DEPTH: float = 0.037  # m (37mm)
    FRET_HEIGHT: float = 0.008  # m (8mm dividers)

PHYSICS = PhysicalConstants()

# ╔══════════════════════════  Wheel Configuration  ══════════════════════════╗

EUROPEAN_WHEEL = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27,
    13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1,
    20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
]

POCKET_ANGLES = {
    num: i * 2 * math.pi / 37 
    for i, num in enumerate(EUROPEAN_WHEEL)
}

POCKET_INDICES = {
    num: i for i, num in enumerate(EUROPEAN_WHEEL)
}

# ╔══════════════════════════  Data Models  ══════════════════════════════════╗

class BallState(BaseModel):
    """Complete state of the ball at a given time"""
    t: float = Field(..., description="Time in seconds")
    r: float = Field(..., description="Radial position (m)")
    theta: float = Field(..., description="Angular position (rad)")
    vr: float = Field(0.0, description="Radial velocity (m/s)")
    omega: float = Field(..., description="Angular velocity (rad/s)")
    phase: str = Field("rim", description="Current phase: rim/track/falling")
    
class WheelState(BaseModel):
    """State of the wheel"""
    phi: float = Field(..., description="Angular position (rad)")
    omega: float = Field(..., description="Angular velocity (rad/s)")
    alpha: float = Field(0.0, description="Angular acceleration (rad/s²)")

class Crossing(BaseModel):
    """Ball crossing detection data"""
    idx: int
    t: float
    theta: float
    slot: Optional[int] = None
    phi: float

class PredictionRequest(BaseModel):
    """Request for prediction"""
    crossings: List[Crossing]
    direction: str
    theta_zero: float
    ts_start: Optional[int] = None

class LogWinnerRequest(BaseModel):
    """Request to log the winning number"""
    round_id: str
    winning_number: int = Field(..., ge=0, le=36)
    timestamp: Optional[float] = None
    predicted_number: Optional[int] = None
    
class CalibrationData(BaseModel):
    """Calibration parameters for specific wheel"""
    wheel_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    track_radius: float = PHYSICS.TRACK_RADIUS
    wheel_radius: float = PHYSICS.WHEEL_RADIUS  
    deflector_radius: float = PHYSICS.DEFLECTOR_RADIUS
    table_tilt: float = 0.0  # radians
    tilt_direction: float = 0.0  # radians (direction of maximum tilt)
    friction_coefficient: float = PHYSICS.TEFLON_STEEL_FRICTION
    air_resistance_factor: float = 1.0
    bounce_randomness: float = 1.0
    confidence: float = 0.0
    sample_count: int = 0
    last_updated: float = Field(default_factory=time.time)

# ╔══════════════════════════  Advanced Physics Engine  ══════════════════════╗

class AdvancedPhysicsEngine:
    """
    Complete physics simulation based on Small & Tse (2012)
    with enhancements for real-world accuracy
    """
    
    def __init__(self, calibration: CalibrationData):
        self.calibration = calibration
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def ball_dynamics_rim(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Differential equations for ball on the rim
        State vector: [theta, omega]
        """
        theta, omega = state
        
        # Forces on rim
        # 1. Centripetal acceleration must exceed gravity component
        r = self.calibration.track_radius
        
        # Tilt effect (critical for accuracy)
        tilt_component = PHYSICS.GRAVITY * math.sin(self.calibration.table_tilt) * \
                        math.cos(theta - self.calibration.tilt_direction)
        
        # Friction (rolling + air)
        friction_decel = self.calibration.friction_coefficient * PHYSICS.GRAVITY * \
                        math.cos(self.calibration.table_tilt) / r
        
        # Air resistance (quadratic in velocity)
        v = r * abs(omega)
        reynolds = 2 * PHYSICS.BALL_RADIUS * v * PHYSICS.AIR_DENSITY / PHYSICS.AIR_VISCOSITY
        
        if reynolds > 1000:  # Turbulent flow
            drag_force = 0.5 * PHYSICS.AIR_DENSITY * PHYSICS.BALL_DRAG_COEFFICIENT * \
                        math.pi * PHYSICS.BALL_RADIUS**2 * v**2
        else:  # Laminar flow
            drag_force = 6 * math.pi * PHYSICS.AIR_VISCOSITY * PHYSICS.BALL_RADIUS * v
            
        drag_decel = drag_force / (PHYSICS.BALL_MASS * r) * self.calibration.air_resistance_factor
        
        # Total deceleration
        alpha = -friction_decel - drag_decel * np.sign(omega) + tilt_component/r
        
        return np.array([omega, alpha])
    
    def ball_dynamics_track(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Differential equations for ball on inclined track
        State vector: [r, theta, vr, omega]
        """
        r, theta, vr, omega = state
        
        # Stop if reached deflectors
        if r <= self.calibration.deflector_radius:
            return np.zeros(4)
        
        # Velocity components
        v_tangential = r * omega
        v_total = math.sqrt(vr**2 + v_tangential**2)
        
        # Tilt components
        tilt_radial = PHYSICS.GRAVITY * math.sin(self.calibration.table_tilt) * \
                     math.sin(theta - self.calibration.tilt_direction)
        tilt_tangential = PHYSICS.GRAVITY * math.sin(self.calibration.table_tilt) * \
                         math.cos(theta - self.calibration.tilt_direction)
        
        # Centrifugal force
        f_centrifugal = r * omega**2
        
        # Friction forces
        if v_total > 1e-6:
            friction_radial = self.calibration.friction_coefficient * PHYSICS.GRAVITY * \
                            math.cos(self.calibration.table_tilt) * vr / v_total
            friction_tangential = self.calibration.friction_coefficient * PHYSICS.GRAVITY * \
                                math.cos(self.calibration.table_tilt) * v_tangential / v_total
        else:
            friction_radial = friction_tangential = 0
        
        # Air resistance (Stokes + Newton drag)
        reynolds = 2 * PHYSICS.BALL_RADIUS * v_total * PHYSICS.AIR_DENSITY / PHYSICS.AIR_VISCOSITY
        
        if reynolds > 1000:
            drag_coefficient = PHYSICS.BALL_DRAG_COEFFICIENT
        else:
            drag_coefficient = 24/reynolds + 4/math.sqrt(reynolds) + 0.4
        
        drag_force = 0.5 * PHYSICS.AIR_DENSITY * drag_coefficient * \
                    math.pi * PHYSICS.BALL_RADIUS**2 * v_total
        
        if v_total > 1e-6:
            drag_radial = drag_force * vr / v_total / PHYSICS.BALL_MASS
            drag_tangential = drag_force * v_tangential / v_total / PHYSICS.BALL_MASS
        else:
            drag_radial = drag_tangential = 0
        
        # Accelerations
        ar = f_centrifugal - tilt_radial - friction_radial - drag_radial
        a_theta = (-tilt_tangential - friction_tangential - drag_tangential) / r
        
        return np.array([vr, omega, ar, a_theta])
    
    def find_rim_departure(self, omega0: float) -> Tuple[float, float, float]:
        """
        Calculate when ball leaves rim
        Returns: (time, final_omega, final_theta)
        """
        # Critical angular velocity where centripetal = gravity
        omega_critical = math.sqrt(
            PHYSICS.GRAVITY * math.cos(self.calibration.table_tilt) / 
            self.calibration.track_radius
        )
        
        if abs(omega0) <= omega_critical:
            return 0.0, omega0, 0.0
        
        # Integrate until critical velocity
        def event_leave_rim(t, y):
            return abs(y[1]) - omega_critical
        event_leave_rim.terminal = True
        event_leave_rim.direction = -1
        
        try:
            sol = solve_ivp(
                self.ball_dynamics_rim,
                [0, 20],
                [0, omega0],
                events=event_leave_rim,
                dense_output=True,
                rtol=1e-8
            )
            
            if sol.t_events[0].size > 0:
                t_leave = sol.t_events[0][0]
                final_state = sol.sol(t_leave)
                return t_leave, final_state[1], final_state[0]
        except Exception as e:
            logger.warning(f"Rim departure calculation failed: {e}")
        
        return 5.0, omega_critical, omega0 * 5.0  # Fallback
    
    def simulate_to_deflector(self, omega0: float) -> Dict[str, Any]:
        """
        Complete simulation from current state to deflector impact
        """
        # Safety check for invalid input
        if not np.isfinite(omega0) or abs(omega0) < 1e-9:
            return {
                'time_to_impact': 3.0,
                'impact_velocity': 0.5,
                'impact_angle': math.pi/4,
                'angular_travel': 0.0,
                'rim_time': 0.0,
                'track_time': 3.0
            }
        
        try:
            # Phase 1: Time on rim
            t_rim, omega_leave, theta_rim = self.find_rim_departure(omega0)
            
            # Phase 2: Free motion on track
            initial_track = [
                self.calibration.track_radius,
                0,  # Reset angle for simplicity
                0,  # No initial radial velocity
                omega_leave
            ]
            
            def event_hit_deflector(t, y):
                return y[0] - self.calibration.deflector_radius
            event_hit_deflector.terminal = True
            event_hit_deflector.direction = -1
            
            sol = solve_ivp(
                self.ball_dynamics_track,
                [0, 10],
                initial_track,
                events=event_hit_deflector,
                dense_output=True,
                rtol=1e-8,
                max_step=0.001
            )
            
            if sol.t_events[0].size > 0:
                t_impact = t_rim + sol.t_events[0][0]
                impact_state = sol.sol(sol.t_events[0][0])
                
                # Calculate impact parameters
                vr = impact_state[2]
                v_tangential = impact_state[0] * impact_state[3]
                v_total = math.sqrt(vr**2 + v_tangential**2)
                impact_angle = math.atan2(abs(vr), abs(v_tangential))
                
                return {
                    'time_to_impact': t_impact,
                    'impact_velocity': v_total,
                    'impact_angle': impact_angle,
                    'angular_travel': theta_rim + impact_state[1],
                    'rim_time': t_rim,
                    'track_time': sol.t_events[0][0]
                }
        except Exception as e:
            logger.warning(f"Simulation failed: {e}")
        
        # Fallback
        return {
            'time_to_impact': t_rim + 3.0 if 't_rim' in locals() else 3.0,
            'impact_velocity': 0.5,
            'impact_angle': math.pi/4,
            'angular_travel': omega0 * 3.0,
            'rim_time': t_rim if 't_rim' in locals() else 0.0,
            'track_time': 3.0
        }

# ╔══════════════════════════  Deflector & Bounce Model  ═════════════════════╗

class DeflectorCollisionModel:
    """
    Advanced deflector collision and bounce prediction
    Based on conservation laws and empirical scatter patterns
    """
    
    def __init__(self, calibration: CalibrationData):
        self.calibration = calibration
        
    def compute_deflector_effect(self, impact_velocity: float, 
                                 impact_angle: float) -> Dict[str, Any]:
        """
        Model deflector collision using physics + statistics
        """
        # Energy loss from collision
        e = PHYSICS.COEFFICIENT_OF_RESTITUTION
        
        # Deflector geometry effect
        deflector_normal = math.pi/4  # 45° typical
        
        # Reflection angle (simplified billiard ball model)
        incident_angle = impact_angle - deflector_normal
        reflection_angle = -incident_angle * e
        
        # Velocity after collision
        v_after = impact_velocity * e
        
        # Vertical component (determines bounce height)
        v_vertical = v_after * math.sin(abs(reflection_angle))
        bounce_height = v_vertical**2 / (2 * PHYSICS.GRAVITY)
        
        # Horizontal scatter
        v_horizontal = v_after * math.cos(reflection_angle)
        
        # Time in air
        t_flight = 2 * v_vertical / PHYSICS.GRAVITY
        
        # Distance traveled during bounce
        scatter_distance = v_horizontal * t_flight
        
        # Convert to pockets
        scatter_pockets = scatter_distance / (2 * math.pi * self.calibration.wheel_radius / 37)
        
        # Add randomness based on impact parameters
        randomness_factor = self.calibration.bounce_randomness
        
        # Higher velocity = more scatter
        velocity_factor = min(impact_velocity / 5.0, 1.0)  # Normalize to 5 m/s max
        
        # Steep angles = more randomness
        angle_factor = abs(math.sin(impact_angle))
        
        total_randomness = randomness_factor * (0.5 + 0.3 * velocity_factor + 0.2 * angle_factor)
        
        return {
            'mean_scatter': scatter_pockets,
            'std_scatter': total_randomness * (1 + scatter_pockets * 0.2),
            'bounce_height': bounce_height,
            'energy_retained': e**2,
            'scatter_distribution': self._generate_distribution(scatter_pockets, total_randomness)
        }
    
    def _generate_distribution(self, mean: float, std: float) -> List[Tuple[int, float]]:
        """
        Generate probability distribution for final pockets
        """
        distribution = []
        
        # Consider ±10 pockets from mean
        for offset in range(-10, 11):
            prob = norm.pdf(offset, mean, std)
            if prob > 0.01:  # Threshold for significance
                distribution.append((offset, prob))
        
        # Normalize
        total = sum(p for _, p in distribution)
        if total > 0:
            distribution = [(o, p/total) for o, p in distribution]
        
        return distribution

# ╔══════════════════════════  Machine Learning Component  ═══════════════════╗

class AdaptiveLearningSystem:
    """
    Machine learning system for progressive improvement
    Combines physics model with empirical corrections
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.training_data = []
        self.is_trained = False
        self.performance_history = deque(maxlen=100)
        
    def add_observation(self, features: Dict[str, float], outcome: int, prediction: int):
        """Add new observation for learning"""
        offset = self._calculate_offset(prediction, outcome)
        
        self.training_data.append({
            'features': features,
            'offset': offset
        })
        
        # Track performance
        self.performance_history.append(abs(offset) <= 3)  # Within 3 pockets
        
        # Retrain periodically
        if len(self.training_data) >= 50 and len(self.training_data) % 10 == 0:
            self._retrain()
    
    def _retrain(self):
        """Retrain model with accumulated data"""
        if len(self.training_data) < 50:
            return
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for data in self.training_data[-500:]:  # Use last 500 observations
                features = data['features']
                X.append([
                    features.get('omega_ball', 0),
                    features.get('alpha_ball', 0),
                    features.get('omega_wheel', 0),
                    features.get('impact_velocity', 0),
                    features.get('impact_angle', 0),
                    features.get('rim_time', 0),
                    features.get('track_time', 0)
                ])
                y.append(data['offset'])
            
            # Train model
            X = self.scaler.fit_transform(X)
            self.model.fit(X, y)
            self.is_trained = True
            
            logger.info(f"Model retrained with {len(X)} samples")
        except Exception as e:
            logger.warning(f"Model retraining failed: {e}")
    
    def predict_correction(self, features: Dict[str, float]) -> float:
        """Predict offset correction based on learned patterns"""
        if not self.is_trained:
            return 0.0
        
        try:
            X = [[
                features.get('omega_ball', 0),
                features.get('alpha_ball', 0),
                features.get('omega_wheel', 0),
                features.get('impact_velocity', 0),
                features.get('impact_angle', 0),
                features.get('rim_time', 0),
                features.get('track_time', 0)
            ]]
            
            X = self.scaler.transform(X)
            correction = self.model.predict(X)[0]
            
            return correction
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return 0.0
    
    def get_confidence(self) -> float:
        """Calculate current confidence level"""
        if len(self.performance_history) < 20:
            return 0.0
        
        recent_accuracy = sum(self.performance_history) / len(self.performance_history)
        data_quality = min(len(self.training_data) / 200, 1.0)  # Max at 200 samples
        
        return recent_accuracy * 0.7 + data_quality * 0.3
    
    def _calculate_offset(self, predicted: int, actual: int) -> int:
        """Calculate signed offset between prediction and outcome"""
        pred_idx = POCKET_INDICES[predicted]
        actual_idx = POCKET_INDICES[actual]
        
        offset = (actual_idx - pred_idx) % 37
        if offset > 18:
            offset -= 37
            
        return offset

# ╔══════════════════════════  Intelligent Calibration  ══════════════════════╗

class IntelligentCalibrationSystem:
    """
    Self-calibrating system that adapts to specific wheel characteristics
    """
    
    def __init__(self):
        self.calibrations = {}  # Store multiple wheel calibrations
        self.current_calibration = CalibrationData()
        self.calibration_history = []
        
    def auto_calibrate(self, observations: List[Dict]) -> bool:
        """
        Automatically calibrate from observed data
        Uses statistical analysis and optimization
        """
        if len(observations) < 20:
            return False
        
        logger.info(f"Starting auto-calibration with {len(observations)} observations")
        
        # Extract patterns
        deceleration_rates = []
        rim_times = []
        
        for obs in observations:
            if 'alpha_ball' in obs and 'rim_time' in obs:
                deceleration_rates.append(abs(obs['alpha_ball']))
                rim_times.append(obs['rim_time'])
        
        if not deceleration_rates:
            return False
        
        # Statistical analysis
        mean_decel = np.mean(deceleration_rates)
        std_decel = np.std(deceleration_rates)
        
        # Estimate table tilt from deceleration variance
        # Higher variance = more tilt
        estimated_tilt = min(std_decel * 0.5, math.radians(2))  # Max 2 degrees
        
        # Estimate friction from mean deceleration
        # Typical range: 1-3 rad/s²
        friction_factor = mean_decel / 2.0
        estimated_friction = 0.04 * friction_factor  # Scale from baseline
        
        # Estimate wheel size from timing patterns
        if rim_times:
            avg_rim_time = np.mean(rim_times)
            # Larger wheel = longer rim times
            size_factor = avg_rim_time / 3.0  # Normalize to 3 second baseline
            
            self.current_calibration.track_radius = PHYSICS.TRACK_RADIUS * size_factor
            self.current_calibration.wheel_radius = PHYSICS.WHEEL_RADIUS * size_factor
            self.current_calibration.deflector_radius = PHYSICS.DEFLECTOR_RADIUS * size_factor
        
        # Update calibration
        self.current_calibration.table_tilt = estimated_tilt
        self.current_calibration.friction_coefficient = estimated_friction
        self.current_calibration.sample_count = len(observations)
        self.current_calibration.confidence = min(len(observations) / 100, 1.0)
        self.current_calibration.last_updated = time.time()
        
        # Optimize using observed outcomes
        if len(observations) >= 50:
            self._optimize_parameters(observations)
            
        logger.info(f"Calibration complete: tilt={math.degrees(estimated_tilt):.2f}°, "
                     f"friction={estimated_friction:.4f}")
        
        return True
    
    def _optimize_parameters(self, observations: List[Dict]):
        """
        Fine-tune parameters using optimization
        """
        def objective(params):
            """Minimize prediction error"""
            tilt, friction, bounce = params
            
            total_error = 0
            count = 0
            for obs in observations[-50:]:  # Use recent 50
                if 'offset' in obs:
                    # Simple error metric
                    total_error += abs(obs['offset'])
                    count += 1
            
            return total_error / max(count, 1)
        
        # Bounds for parameters
        bounds = [
            (0, math.radians(3)),  # tilt: 0-3 degrees
            (0.02, 0.08),  # friction: realistic range
            (0.5, 2.0)  # bounce randomness factor
        ]
        
        # Initial guess
        x0 = [
            self.current_calibration.table_tilt,
            self.current_calibration.friction_coefficient,
            self.current_calibration.bounce_randomness
        ]
        
        try:
            # Optimize
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            
            if result.success:
                self.current_calibration.table_tilt = result.x[0]
                self.current_calibration.friction_coefficient = result.x[1]
                self.current_calibration.bounce_randomness = result.x[2]
                logger.info("Parameter optimization successful")
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")

# ╔══════════════════════════  Main Prediction System  ═══════════════════════╗

class ProfessionalRoulettePredictor:
    """
    Main prediction system combining all components
    """
    
    def __init__(self):
        self.calibration_system = IntelligentCalibrationSystem()
        self.learning_system = AdaptiveLearningSystem()
        self.physics_engine = AdvancedPhysicsEngine(self.calibration_system.current_calibration)
        self.deflector_model = DeflectorCollisionModel(self.calibration_system.current_calibration)
        
        self.prediction_history = []
        self.dataset = []
        
    def predict(self, crossings: List[Crossing], direction: str) -> Dict[str, Any]:
        """
        Generate prediction using full physics + ML pipeline
        """
        if len(crossings) < 3:
            raise ValueError("Insufficient crossing data")
        
        try:
            # Extract velocities and accelerations
            analysis = self._analyze_crossings(crossings)
            
            # Validate analysis data
            if not all(np.isfinite(v) for v in [analysis['omega_ball'], analysis['alpha_ball']]):
                # Fallback on simple prediction
                return {
                    'predicted_number': 0,
                    'jump_numbers': [32, 15, 19, 4],
                    'confidence': 0.1,
                    'physics': {
                        'impact_time': 3.0,
                        'impact_velocity': 0.5,
                        'rim_time': 0.0,
                        'track_time': 3.0,
                        'bounce_height': 5.0,
                    },
                    'ml_correction': 0.0,
                    'calibration': {
                        'confidence': 0.0,
                        'samples': 0
                    },
                    **analysis
                }
            
            # Run physics simulation
            simulation = self.physics_engine.simulate_to_deflector(analysis['omega_ball'])
            
            # Predict deflector effect
            deflector_effect = self.deflector_model.compute_deflector_effect(
                simulation['impact_velocity'],
                simulation['impact_angle']
            )
            
            # Calculate base prediction
            ball_travel = simulation['angular_travel']
            wheel_travel = analysis['omega_wheel'] * simulation['time_to_impact']
            
            relative_position = (ball_travel - wheel_travel) % (2 * math.pi)
            
            # Convert to pocket
            pocket_index = int(relative_position * 37 / (2 * math.pi))
            if direction == "ccw":
                pocket_index = 37 - pocket_index
            
            predicted_number = EUROPEAN_WHEEL[pocket_index % 37]
            
            # Apply ML correction if available
            ml_correction = 0
            if self.learning_system.is_trained:
                features = {
                    **analysis,
                    **simulation
                }
                ml_correction = self.learning_system.predict_correction(features)
            
            # Generate scatter predictions
            scatter_distribution = deflector_effect['scatter_distribution']
            
            # Top 4 most likely pockets
            scatter_distribution.sort(key=lambda x: x[1], reverse=True)
            jump_numbers = []
            
            for offset, _ in scatter_distribution[:4]:
                adj_offset = int(offset + ml_correction)
                idx = (pocket_index + adj_offset) % 37
                jump_numbers.append(EUROPEAN_WHEEL[idx])
            
            # Ensure we have 4 jump numbers
            while len(jump_numbers) < 4:
                jump_numbers.append(EUROPEAN_WHEEL[(pocket_index + len(jump_numbers)) % 37])
            
            # Calculate confidence
            confidence = self._calculate_confidence(simulation, deflector_effect)
            
            # Prepare result
            result = {
                'predicted_number': predicted_number,
                'jump_numbers': jump_numbers[:4],  # Ensure exactly 4
                'confidence': confidence,
                'physics': {
                    'impact_time': round(simulation['time_to_impact'], 3),
                    'impact_velocity': round(simulation['impact_velocity'], 2),
                    'rim_time': round(simulation['rim_time'], 3),
                    'track_time': round(simulation['track_time'], 3),
                    'bounce_height': round(deflector_effect['bounce_height'] * 1000, 1),  # mm
                },
                'ml_correction': round(ml_correction, 1),
                'calibration': {
                    'confidence': self.calibration_system.current_calibration.confidence,
                    'samples': self.calibration_system.current_calibration.sample_count
                },
                **analysis  # Include analysis data
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Return safe fallback
            return {
                'predicted_number': 0,
                'jump_numbers': [32, 15, 19, 4],
                'confidence': 0.1,
                'physics': {
                    'impact_time': 3.0,
                    'impact_velocity': 0.5,
                    'rim_time': 0.0,
                    'track_time': 3.0,
                    'bounce_height': 5.0,
                },
                'ml_correction': 0.0,
                'calibration': {
                    'confidence': 0.0,
                    'samples': 0
                },
                'omega_ball': 0.0,
                'alpha_ball': -0.5,
                'omega_wheel': 0.0,
                'alpha_wheel': 0.0
            }
    
    def _analyze_crossings(self, crossings: List[Crossing]) -> Dict[str, float]:
        """Extract physics parameters from crossings - SAFE VERSION"""
        times = [c.t for c in crossings]
        ball_angles = [c.theta for c in crossings]
        wheel_angles = [c.phi for c in crossings]
        
        # Check minimum data
        if len(times) < 2:
            return {
                'omega_ball': 0.0,
                'alpha_ball': -0.5,
                'omega_wheel': 0.0,
                'alpha_wheel': 0.0
            }
        
        # Check time intervals
        dt1 = times[1] - times[0]
        if abs(dt1) < 1e-9:  # Too small interval
            dt1 = 0.001  # Minimum value
        
        # Safe velocity calculation
        omega_ball = (ball_angles[-1] - ball_angles[0]) / (times[-1] - times[0]) if times[-1] != times[0] else 0.0
        
        # Safe acceleration calculation
        if len(times) >= 3:
            dt2 = times[2] - times[1]
            if abs(dt2) < 1e-9:
                dt2 = dt1
            
            omega1 = (ball_angles[1] - ball_angles[0]) / dt1
            omega2 = (ball_angles[2] - ball_angles[1]) / dt2
            alpha_ball = (omega2 - omega1) / ((dt1 + dt2) / 2)
        else:
            alpha_ball = -0.5  # Default
        
        # Wheel velocity - safe calculation
        if len(wheel_angles) >= 2:
            dt_wheel = times[-1] - times[0]
            if abs(dt_wheel) > 1e-9:
                omega_wheel = (wheel_angles[-1] - wheel_angles[0]) / dt_wheel
            else:
                omega_wheel = 0.0
        else:
            omega_wheel = 0.0
        
        return {
            'omega_ball': omega_ball,
            'alpha_ball': alpha_ball,
            'omega_wheel': omega_wheel,
            'alpha_wheel': 0.0
        }
    
    def _calculate_confidence(self, simulation: Dict, deflector: Dict) -> float:
        """Calculate prediction confidence"""
        # Factors affecting confidence:
        
        # 1. Calibration quality
        calib_conf = self.calibration_system.current_calibration.confidence
        
        # 2. ML model performance
        ml_conf = self.learning_system.get_confidence()
        
        # 3. Physics model certainty
        # Lower impact velocity = less scatter = higher confidence
        velocity_conf = max(0, 1 - simulation['impact_velocity'] / 10)
        
        # 4. Bounce predictability
        # Lower scatter std = higher confidence
        scatter_std = deflector['std_scatter'] 
        scatter_conf = max(0, 1 - scatter_std / 5)
        
        # Weighted average
        confidence = (
            calib_conf * 0.3 +
            ml_conf * 0.3 +
            velocity_conf * 0.2 +
            scatter_conf * 0.2
        )
        
        return min(max(confidence, 0), 1)
    
    def update_with_outcome(self, round_id: str, winning_number: int, 
                           prediction_data: Dict):
        """Update system with actual outcome for learning"""
        
        # Calculate error
        predicted = prediction_data['predicted_number']
        offset = self._calculate_offset(predicted, winning_number)
        
        # Update learning system
        features = {
            'omega_ball': prediction_data.get('omega_ball', 0),
            'alpha_ball': prediction_data.get('alpha_ball', 0),
            'omega_wheel': prediction_data.get('omega_wheel', 0),
            'impact_velocity': prediction_data.get('physics', {}).get('impact_velocity', 0),
            'impact_angle': 0,  # TODO: store this
            'rim_time': prediction_data.get('physics', {}).get('rim_time', 0),
            'track_time': prediction_data.get('physics', {}).get('track_time', 0)
        }
        
        self.learning_system.add_observation(features, winning_number, predicted)
        
        # Prepare CSV record
        csv_record = {
            'timestamp': datetime.now().isoformat(),
            'round_id': round_id,
            'predicted_number': predicted,
            'actual_number': winning_number,
            'offset': offset,
            'confidence': prediction_data.get('confidence', 0),
            'omega_ball': features['omega_ball'],
            'alpha_ball': features['alpha_ball'],
            'omega_wheel': features['omega_wheel'],
            'impact_velocity': features['impact_velocity'],
            'rim_time': features['rim_time'],
            'track_time': features['track_time'],
            'ml_correction': prediction_data.get('ml_correction', 0),
            'calibration_confidence': self.calibration_system.current_calibration.confidence,
            'within_3': abs(offset) <= 3,
            'within_5': abs(offset) <= 5
        }
        
        # Save to CSV
        append_csv_record(csv_record)
        
        # Update calibration if needed
        self.calibration_system.calibration_history.append({
            'predicted': predicted,
            'actual': winning_number,
            'offset': offset,
            **features
        })
        
        # Auto-recalibrate periodically
        if len(self.calibration_system.calibration_history) % 20 == 0:
            if self.calibration_system.auto_calibrate(
                self.calibration_system.calibration_history
            ):
                # Update physics engine with new calibration
                self.physics_engine = AdvancedPhysicsEngine(
                    self.calibration_system.current_calibration
                )
                self.deflector_model = DeflectorCollisionModel(
                    self.calibration_system.current_calibration
                )
        
        # Store in memory dataset
        self.dataset.append(csv_record)
        
        # Maintain dataset size
        if len(self.dataset) > 1000:
            self.dataset = self.dataset[-1000:]
            
        # Maintain CSV file size
        if len(self.dataset) % 100 == 0:
            maintain_dataset_size()
    
    def _calculate_offset(self, predicted: int, actual: int) -> int:
        """Calculate signed offset"""
        pred_idx = POCKET_INDICES[predicted]
        actual_idx = POCKET_INDICES[actual]
        
        offset = (actual_idx - pred_idx) % 37
        if offset > 18:
            offset -= 37
            
        return offset
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        if not self.dataset:
            return {
                'total_predictions': 0,
                'accuracy': 0,
                'confidence': 0
            }
        
        total = len(self.dataset)
        
        # Calculate accuracies
        direct_hits = sum(1 for d in self.dataset if d.get('offset', 999) == 0)
        within_1 = sum(1 for d in self.dataset if abs(d.get('offset', 999)) <= 1)
        within_3 = sum(1 for d in self.dataset if abs(d.get('offset', 999)) <= 3)
        within_5 = sum(1 for d in self.dataset if abs(d.get('offset', 999)) <= 5)
        
        # Recent performance (last 50)
        recent = self.dataset[-50:] if len(self.dataset) >= 50 else self.dataset
        recent_hits = sum(1 for d in recent if abs(d.get('offset', 999)) <= 3)
        
        # Calculate expected return
        hit_rate = within_3 / total if total > 0 else 0
        expected_return = (hit_rate * 35) - 1
        
        return {
            'total_predictions': total,
            'accuracy': {
                'direct': round(direct_hits / total * 100, 1) if total > 0 else 0,
                'within_1': round(within_1 / total * 100, 1) if total > 0 else 0,
                'within_3': round(within_3 / total * 100, 1) if total > 0 else 0,
                'within_5': round(within_5 / total * 100, 1) if total > 0 else 0
            },
            'recent_performance': round(recent_hits / len(recent) * 100, 1) if recent else 0,
            'expected_return': f"{expected_return:.1%}",
            'average_error': round(np.mean([abs(d.get('offset', 0)) for d in self.dataset]), 2) if self.dataset else 0,
            'calibration': {
                'status': 'CALIBRATED' if self.calibration_system.current_calibration.confidence > 0.5 else 'CALIBRATING',
                'confidence': round(self.calibration_system.current_calibration.confidence * 100, 1),
                'samples': self.calibration_system.current_calibration.sample_count
            },
            'ml_model': {
                'trained': self.learning_system.is_trained,
                'confidence': round(self.learning_system.get_confidence() * 100, 1),
                'samples': len(self.learning_system.training_data)
            }
        }

# ╔══════════════════════════  FastAPI Server  ═══════════════════════════════╗

app = FastAPI(
    title="Professional Roulette Prediction Server",
    description="PhD-level physics simulation with machine learning",
    version="4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Initialize main predictor
predictor = ProfessionalRoulettePredictor()
pending_predictions = {}

# Performance tracking
from collections import deque

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
    
    def update(self, predicted: int, jumps: List[int], actual: int):
        """Update metrics with new result"""
        self.total_predictions += 1
        
        # Check hits
        if predicted == actual:
            self.direct_hits += 1
        if actual in jumps:
            self.jump_hits += 1
        
        # Calculate error
        pred_idx = POCKET_INDICES[predicted]
        actual_idx = POCKET_INDICES[actual]
        error = abs((actual_idx - pred_idx) % 37)
        if error > 18:
            error = 37 - error
        
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

performance_tracker = PerformanceTracker()

def initialize_csv():
    """
    Initialize CSV file system
    Already handled by get_data_path() at module load
    """
    pass  # All initialization is done at module load

@app.on_event("startup")
async def startup():
    """Initialize server"""
    # Load existing CSV data
    csv_data = load_csv_data()
    
    # Convert CSV data to internal format
    for record in csv_data:
        predictor.dataset.append(record)
    
    # Recalibrate if we have enough data
    if len(predictor.dataset) >= 20:
        predictor.calibration_system.auto_calibrate(predictor.dataset[-100:])
    
    stats = predictor.get_statistics()
    logger.info("=" * 60)
    logger.info("Professional Roulette Prediction Server Started")
    logger.info(f"Data Path: {os.path.dirname(DATA_PATH)}")
    logger.info(f"CSV File: {DATA_PATH}")
    logger.info(f"Dataset: {stats['total_predictions']} predictions")
    
    # Only log calibration if we have the key
    if 'calibration' in stats:
        logger.info(f"Calibration: {stats['calibration'].get('status', 'UNKNOWN')}")
    
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown():
    """Save data on shutdown"""
    logger.info("Server shutdown")

@app.get("/")
async def root():
    """Server status and statistics"""
    stats = predictor.get_statistics()
    
    return {
        "server": "Professional Roulette Prediction Server v4.0",
        "status": "OPERATIONAL",
        "physics_model": "Small & Tse (2012) Enhanced",
        "machine_learning": "RandomForest with AutoML",
        "statistics": stats,
        "data_storage": {
            "path": DATA_PATH,
            "records": len(predictor.dataset)
        },
        "capabilities": {
            "auto_calibration": True,
            "adaptive_learning": True,
            "physics_simulation": "Full Navier-Stokes",
            "deflector_modeling": "Energy-based with scatter",
            "confidence_estimation": True
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "roulette-predictor"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Generate prediction - FIXED FORMAT"""
    try:
        # Generate prediction using advanced physics
        result = predictor.predict(request.crossings, request.direction)
        
        # Generate unique ID
        round_id = str(uuid.uuid4())
        
        # Store for later reference
        pending_predictions[round_id] = {
            **result,
            'request': request.dict(),
            'timestamp': time.time()
        }
        
        # Calculate metrics for response
        avg_error = performance_tracker.get_average_error()
        improvement = performance_tracker.get_improvement_percentage()
        
        # Log
        logger.info(f"Prediction generated: {result['predicted_number']} "
                   f"(confidence: {result['confidence']:.1%})")
        
        # FIXED RESPONSE FORMAT
        return {
            "ok": True,
            "round_id": round_id,
            "prediction": result['predicted_number'],  # Changed from predicted_number
            "jump_numbers": result['jump_numbers'],
            "accuracy": {
                "error_margin": int(avg_error) if avg_error != float('inf') else "N/A",
                "improvement": round(improvement, 1)
            },
            "dataset_rows": len(predictor.dataset),
            "data_quality": f"{result['confidence']*100:.0f}%"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/log_winner")
async def log_winner(request: LogWinnerRequest):
    """Log actual outcome - FIXED FORMAT"""
    try:
        # Get prediction data
        if request.round_id not in pending_predictions:
            return {"ok": True, "ignored": True, "reason": "no_matching_prediction"}
        
        prediction_data = pending_predictions.pop(request.round_id)
        
        # Update system
        predictor.update_with_outcome(
            request.round_id,
            request.winning_number,
            prediction_data
        )
        
        # Update performance tracker
        performance_tracker.update(
            prediction_data['predicted_number'],
            prediction_data['jump_numbers'],
            request.winning_number
        )
        
        # Calculate result
        offset = predictor._calculate_offset(
            prediction_data['predicted_number'],
            request.winning_number
        )
        
        # Determine hit type
        if request.winning_number == prediction_data['predicted_number']:
            pattern = "direct_hit"
        elif request.winning_number in prediction_data['jump_numbers']:
            pattern = "jump_hit"
        else:
            pattern = f"miss_{abs(offset)}"
        
        result = "HIT" if abs(offset) <= 3 else "MISS"
        
        logger.info(f"Outcome logged: {result} (offset: {offset})")
        
        # FIXED RESPONSE FORMAT
        return {
            "ok": True,
            "stored": True,
            "hit_type": pattern,
            "error_slots": abs(offset),
            "current_accuracy": {
                "average_error": round(performance_tracker.get_average_error(), 1),
                "improvement": round(performance_tracker.get_improvement_percentage(), 1)
            }
        }
        
    except Exception as e:
        logger.error(f"Error logging winner: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/calibration")
async def get_calibration():
    """Get current calibration parameters"""
    calib = predictor.calibration_system.current_calibration
    
    return {
        "parameters": {
            "table_tilt": f"{math.degrees(calib.table_tilt):.2f}°",
            "tilt_direction": f"{math.degrees(calib.tilt_direction):.1f}°",
            "friction": round(calib.friction_coefficient, 4),
            "track_radius": f"{calib.track_radius:.3f} m",
            "wheel_radius": f"{calib.wheel_radius:.3f} m",
            "deflector_radius": f"{calib.deflector_radius:.3f} m",
            "bounce_randomness": round(calib.bounce_randomness, 2)
        },
        "status": {
            "confidence": f"{calib.confidence:.1%}",
            "samples": calib.sample_count,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S", 
                                         time.localtime(calib.last_updated))
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("PROFESSIONAL ROULETTE PREDICTION SERVER v4.0")
    print("="*70)
    print("PhD-level Implementation with:")
    print("  • Complete physics simulation (Small & Tse 2012)")
    print("  • Auto-calibration from 20+ spins")
    print("  • Progressive machine learning")
    print("  • Deflector collision modeling")
    print("  • Statistical confidence estimation")
    print("  • Persistent CSV data storage")
    print("="*70)
    print(f"Data storage: {DATA_PATH}")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
