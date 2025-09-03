"""
Enhanced Roulette Prediction Server with Physics Model
Combines pattern matching with physics calculations for improved accuracy
"""

import os
import csv
import json
import math
import time
import logging
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('roulette_physics_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PhysicsRouletteServer")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Server configuration with physics constants"""
    
    # Grid parameters for hybrid approach
    POSITION_CELL_SIZE = 5   # pixels - more precise than before
    SPEED_CELL_SIZE = 10     # milliseconds
    
    # Data management
    MAX_RECORDS_PER_CELL = 200
    GLOBAL_MAX_RECORDS = 1000000
    PENDING_TIMEOUT_MINUTES = 10
    DATA_RETENTION_DAYS = 60
    
    # Physics constants
    GRAVITY = 9.81                # m/s²
    WHEEL_RADIUS = 0.38           # meters (standard roulette wheel)
    BALL_RADIUS = 0.01            # meters
    CRITICAL_VELOCITY = 0.8       # rad/s - velocity when ball drops
    BASE_FRICTION_COEFF = 0.025   # initial friction coefficient
    AIR_RESISTANCE_COEFF = 0.0001 # air drag coefficient
    
    # Prediction parameters
    MIN_MATCHES_FOR_PREDICTION = 5    # reduced since physics helps
    MIN_CONFIDENCE_THRESHOLD = 0.25   # lowered threshold
    SECTOR_SIZE = 8                   # predict 8-number sectors
    
    # Validation
    MIN_BALL_SPEED_MS = 300
    MAX_BALL_SPEED_MS = 3000
    MIN_TRAVELED_POCKETS = 5         # filter out very slow spins
    VALID_DIRECTIONS = ['CW', 'CCW']
    
    # European wheel layout (single zero, clockwise order)
    WHEEL_LAYOUT = [
        0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27,
        13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1,
        20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
    ]
    
    # Create reverse lookup
    POCKET_TO_INDEX = {num: idx for idx, num in enumerate(WHEEL_LAYOUT)}
    
    # CSV structure
    CSV_COLUMNS = [
        'timestamp', 'round_id', 'table_id',
        'pos_x', 'pos_y', 'speed_ms', 'traveled_pockets',
        'direction', 'number_t1', 'number_t2', 
        'winning_number', 'offset_from_t2',
        'predicted_physics', 'predicted_pattern', 'error_physics', 'error_pattern'
    ]

# ============================================================================
# DATA MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Incoming prediction request from client"""
    round_id: str = Field(..., min_length=5, max_length=50)
    pos_x: float = Field(..., ge=0, le=2000)
    pos_y: float = Field(..., ge=0, le=2000)
    speed_ms_total: int = Field(..., ge=100, le=5000)
    traveled_pockets: int = Field(default=7, ge=1, le=37)
    direction: str = Field(..., description="Ball direction: CW or CCW")
    number_at_t1: int = Field(..., ge=0, le=36)
    number_at_t2: int = Field(..., ge=0, le=36)
    table_id: str = Field(default="default", max_length=50)
    
    @field_validator('speed_ms_total')
    @classmethod
    def validate_speed(cls, v):
        if v < Config.MIN_BALL_SPEED_MS:
            raise ValueError(f"Speed {v}ms below minimum {Config.MIN_BALL_SPEED_MS}ms")
        return v
    
    @field_validator('traveled_pockets')
    @classmethod
    def validate_pockets(cls, v):
        if v < Config.MIN_TRAVELED_POCKETS:
            raise ValueError(f"Traveled pockets {v} below minimum {Config.MIN_TRAVELED_POCKETS}")
        return v
    
    @field_validator('direction')
    @classmethod
    def validate_direction(cls, v):
        if v not in Config.VALID_DIRECTIONS:
            raise ValueError('Direction must be CW or CCW')
        return v

class WinnerRequest(BaseModel):
    """Winning number notification from client"""
    round_id: str = Field(..., min_length=5, max_length=50)
    winning_number: int = Field(..., ge=0, le=36)

# ============================================================================
# PHYSICS ENGINE
# ============================================================================

class PhysicsEngine:
    """Physics calculations for ball trajectory prediction"""
    
    def __init__(self):
        self.friction_map = {}  # Position-based friction coefficients
        self.calibration_data = defaultdict(list)
    
    def calculate_prediction(self, request: PredictionRequest) -> Tuple[Optional[int], float, Dict]:
        """
        Calculate predicted number using physics model
        Returns: (predicted_number, confidence, debug_info)
        """
        # Convert to physical units
        pockets_per_second = request.traveled_pockets / (request.speed_ms_total / 1000.0)
        angular_velocity = pockets_per_second * (2 * math.pi / 37)  # rad/s
        
        # Get calibrated friction for this position
        friction = self.get_friction_coefficient(request.pos_x, request.pos_y)
        
        # Calculate deceleration (simplified model)
        deceleration = (friction * Config.GRAVITY) / Config.WHEEL_RADIUS
        
        # Add air resistance (proportional to velocity squared)
        air_drag = Config.AIR_RESISTANCE_COEFF * angular_velocity ** 2
        total_deceleration = deceleration + air_drag
        
        # Time until critical velocity
        if angular_velocity <= Config.CRITICAL_VELOCITY:
            return None, 0.0, {"error": "Initial velocity too low"}
        
        time_to_drop = (angular_velocity - Config.CRITICAL_VELOCITY) / total_deceleration
        
        # Distance traveled (in radians)
        distance_rad = (angular_velocity * time_to_drop - 
                       0.5 * total_deceleration * time_to_drop ** 2)
        
        # Convert to pockets
        pockets_to_travel = int((distance_rad * 37) / (2 * math.pi))
        
        # Account for wheel direction
        if request.direction == "CCW":
            pockets_to_travel = -pockets_to_travel
        
        # Calculate predicted number
        predicted_number = WheelPhysics.get_number_at_distance(
            request.number_at_t2,
            pockets_to_travel,
            request.direction
        )
        
        # Calculate confidence based on calibration quality
        confidence = self.calculate_confidence(request.pos_x, request.pos_y, friction)
        
        debug_info = {
            "angular_velocity": round(angular_velocity, 3),
            "friction": round(friction, 5),
            "deceleration": round(total_deceleration, 3),
            "time_to_drop": round(time_to_drop, 2),
            "pockets_to_travel": pockets_to_travel
        }
        
        return predicted_number, confidence, debug_info
    
    def get_friction_coefficient(self, x: float, y: float) -> float:
        """Get calibrated friction coefficient for position"""
        # Create position key
        grid_x = round(x / Config.POSITION_CELL_SIZE) * Config.POSITION_CELL_SIZE
        grid_y = round(y / Config.POSITION_CELL_SIZE) * Config.POSITION_CELL_SIZE
        pos_key = (grid_x, grid_y)
        
        if pos_key in self.friction_map:
            return self.friction_map[pos_key]
        
        return Config.BASE_FRICTION_COEFF
    
    def update_calibration(self, request_data: dict, actual_offset: int):
        """Update friction map based on actual results"""
        # Calculate what friction would have given correct result
        actual_pockets = actual_offset
        speed_ms = request_data['speed_ms_total']
        traveled = request_data['traveled_pockets']
        
        # Initial angular velocity
        pockets_per_sec = traveled / (speed_ms / 1000.0)
        angular_vel = pockets_per_sec * (2 * math.pi / 37)
        
        # Skip if velocity too low
        if angular_vel <= Config.CRITICAL_VELOCITY:
            return
        
        # Back-calculate required deceleration
        # Using simplified model: d = v*t - 0.5*a*t²
        # We know d (actual_pockets converted to radians)
        distance_rad = (actual_pockets * 2 * math.pi) / 37
        
        # Quadratic formula to find time
        # 0.5*a*t² - v*t + d = 0
        # This is approximate - real physics is more complex
        
        pos_key = (
            round(request_data['pos_x'] / Config.POSITION_CELL_SIZE) * Config.POSITION_CELL_SIZE,
            round(request_data['pos_y'] / Config.POSITION_CELL_SIZE) * Config.POSITION_CELL_SIZE
        )
        
        # Store calibration data
        self.calibration_data[pos_key].append({
            'velocity': angular_vel,
            'actual_pockets': actual_pockets,
            'timestamp': datetime.now()
        })
        
        # Update friction map with weighted average of recent data
        if len(self.calibration_data[pos_key]) >= 3:
            self.recalculate_friction(pos_key)
    
    def recalculate_friction(self, pos_key: Tuple[float, float]):
        """Recalculate friction coefficient from calibration data"""
        recent_data = self.calibration_data[pos_key][-20:]  # Last 20 spins
        
        if not recent_data:
            return
        
        # Simple average for now - could use more sophisticated methods
        friction_estimates = []
        
        for data in recent_data:
            # Simplified back-calculation
            # In reality, would solve the differential equation properly
            est_friction = Config.BASE_FRICTION_COEFF * (1 + data['actual_pockets'] / 100)
            friction_estimates.append(est_friction)
        
        # Update friction map
        self.friction_map[pos_key] = sum(friction_estimates) / len(friction_estimates)
        
        # Limit friction to reasonable range
        self.friction_map[pos_key] = max(0.01, min(0.1, self.friction_map[pos_key]))
    
    def calculate_confidence(self, x: float, y: float, friction: float) -> float:
        """Calculate confidence based on calibration quality"""
        pos_key = (
            round(x / Config.POSITION_CELL_SIZE) * Config.POSITION_CELL_SIZE,
            round(y / Config.POSITION_CELL_SIZE) * Config.POSITION_CELL_SIZE
        )
        
        # Base confidence
        confidence = 0.3
        
        # Increase if we have calibration data
        if pos_key in self.calibration_data:
            data_points = len(self.calibration_data[pos_key])
            if data_points >= 10:
                confidence = 0.5
            if data_points >= 20:
                confidence = 0.6
            
            # Check consistency of recent predictions
            if data_points >= 5:
                recent = self.calibration_data[pos_key][-5:]
                variance = self.calculate_variance(recent)
                if variance < 5:  # Low variance = high consistency
                    confidence += 0.1
        
        return min(0.7, confidence)  # Cap at 70%
    
    def calculate_variance(self, data_points: List[Dict]) -> float:
        """Calculate variance in actual pockets for consistency check"""
        if len(data_points) < 2:
            return 100.0
        
        pockets = [d['actual_pockets'] for d in data_points]
        mean = sum(pockets) / len(pockets)
        variance = sum((p - mean) ** 2 for p in pockets) / len(pockets)
        
        return variance

# ============================================================================
# WHEEL PHYSICS
# ============================================================================

class WheelPhysics:
    """Handle roulette wheel physics and calculations"""
    
    @staticmethod
    def calculate_pocket_distance(from_number: int, to_number: int, direction: str) -> int:
        """
        Calculate pocket distance between two numbers in given direction
        Returns positive value for forward movement
        """
        from_idx = Config.POCKET_TO_INDEX.get(from_number)
        to_idx = Config.POCKET_TO_INDEX.get(to_number)
        
        if from_idx is None or to_idx is None:
            logger.warning(f"Invalid pocket numbers: from={from_number}, to={to_number}")
            return 0
        
        if direction == "CW":
            distance = (to_idx - from_idx) % 37
        else:  # CCW
            distance = (from_idx - to_idx) % 37
        
        # Normalize to -18 to +18 range
        if distance > 18:
            distance = distance - 37
            
        return distance
    
    @staticmethod
    def get_number_at_distance(from_number: int, distance: int, direction: str) -> int:
        """Get pocket number at specified distance from reference"""
        from_idx = Config.POCKET_TO_INDEX.get(from_number)
        
        if from_idx is None:
            logger.warning(f"Invalid pocket number: {from_number}")
            return from_number
        
        if direction == "CW":
            target_idx = (from_idx + distance) % 37
        else:  # CCW
            target_idx = (from_idx - distance) % 37
            
        return Config.WHEEL_LAYOUT[target_idx]
    
    @staticmethod
    def get_sector(center_number: int, size: int = 8) -> List[int]:
        """Get sector of numbers around center"""
        center_idx = Config.POCKET_TO_INDEX.get(center_number, 0)
        sector = []
        
        half_size = size // 2
        for offset in range(-half_size, half_size + 1):
            idx = (center_idx + offset) % 37
            sector.append(Config.WHEEL_LAYOUT[idx])
            
        return sector

# ============================================================================
# HYBRID STORAGE SYSTEM
# ============================================================================

class HybridStorage:
    """Combined pattern matching and physics calibration storage"""
    
    def __init__(self):
        self.data_path = self._get_data_path()
        self.pending_rounds = {}
        self.pattern_database = defaultdict(list)
        self.physics_engine = PhysicsEngine()
        self.total_records = 0
        self.predictions_made = {}
        
        self._initialize_storage()
        self._load_existing_data()
    
    def _get_data_path(self) -> str:
        """Determine optimal data storage location"""
        candidates = [
            os.getenv("ROULETTE_DATA_PATH", ""),
            os.path.expanduser("~/.roulette_physics/database.csv"),
            "./roulette_physics_database.csv"
        ]
        
        for path in candidates:
            if not path:
                continue
            try:
                directory = os.path.dirname(path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                
                with open(path, 'a', encoding='utf-8'):
                    pass
                
                logger.info(f"Using data storage: {path}")
                return path
            except Exception as e:
                continue
                
        raise RuntimeError("No writable location found for database")
    
    def _initialize_storage(self):
        """Initialize CSV file with headers if needed"""
        if not os.path.exists(self.data_path) or os.path.getsize(self.data_path) == 0:
            with open(self.data_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=Config.CSV_COLUMNS)
                writer.writeheader()
            logger.info("Initialized new database")
    
    def _load_existing_data(self):
        """Load historical data for both pattern matching and physics calibration"""
        if not os.path.exists(self.data_path):
            return
            
        try:
            cutoff_date = datetime.now() - timedelta(days=Config.DATA_RETENTION_DAYS)
            
            with open(self.data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                loaded = 0
                
                for row in reader:
                    if not all(k in row for k in ['pos_x', 'pos_y', 'speed_ms', 'direction', 'offset_from_t2', 'timestamp']):
                        continue
                    
                    try:
                        row_timestamp = datetime.fromisoformat(row['timestamp'])
                        if row_timestamp < cutoff_date:
                            continue
                        
                        x = float(row['pos_x'])
                        y = float(row['pos_y'])
                        speed = int(row['speed_ms'])
                        direction = row['direction']
                        offset = int(row['offset_from_t2'])
                        
                        if direction not in Config.VALID_DIRECTIONS:
                            continue
                        
                        # Add to pattern database
                        cell_key = self._get_cell_key(x, y, speed, direction)
                        pattern_data = {
                            'offset': offset,
                            'timestamp': row['timestamp'],
                            'traveled_pockets': int(row.get('traveled_pockets', 7))
                        }
                        self.pattern_database[cell_key].append(pattern_data)
                        
                        # Update physics calibration
                        if 'winning_number' in row and row['winning_number']:
                            self.physics_engine.update_calibration(
                                {
                                    'pos_x': x,
                                    'pos_y': y,
                                    'speed_ms_total': speed,
                                    'traveled_pockets': int(row.get('traveled_pockets', 7))
                                },
                                offset
                            )
                        
                        loaded += 1
                        
                        # Enforce cell limit
                        if len(self.pattern_database[cell_key]) > Config.MAX_RECORDS_PER_CELL:
                            self.pattern_database[cell_key] = self.pattern_database[cell_key][-Config.MAX_RECORDS_PER_CELL:]
                    
                    except (ValueError, KeyError) as e:
                        continue
                
                self.total_records = loaded
                logger.info(f"Loaded {loaded} records into {len(self.pattern_database)} grid cells")
                logger.info(f"Physics calibration data for {len(self.physics_engine.friction_map)} positions")
                
        except Exception as e:
            logger.error(f"Error loading database: {e}")
    
    def _get_cell_key(self, x: float, y: float, speed: int, direction: str) -> Tuple:
        """Convert position and speed to grid cell coordinates"""
        x_cell = round(x / Config.POSITION_CELL_SIZE) * Config.POSITION_CELL_SIZE
        y_cell = round(y / Config.POSITION_CELL_SIZE) * Config.POSITION_CELL_SIZE
        speed_cell = round(speed / Config.SPEED_CELL_SIZE) * Config.SPEED_CELL_SIZE
        return (x_cell, y_cell, speed_cell, direction)
    
    def store_pending(self, request: PredictionRequest) -> bool:
        """Store round data temporarily until winning number arrives"""
        # Clear old pending rounds if new data arrives
        if self.pending_rounds:
            old_count = len(self.pending_rounds)
            self.pending_rounds.clear()
            self.predictions_made.clear()
            logger.info(f"Cleared {old_count} pending rounds - new round data received")
        
        # Store pending data
        self.pending_rounds[request.round_id] = {
            'timestamp': datetime.now(),
            'data': request.dict(),
            'physics_prediction': None,
            'pattern_prediction': None
        }
        
        logger.info(f"Stored pending round {request.round_id}")
        return True
    
    def get_hybrid_prediction(self, request: PredictionRequest) -> Dict[str, Any]:
        """Get prediction using both physics and pattern matching"""
        
        # 1. Physics-based prediction
        physics_number, physics_confidence, debug_info = self.physics_engine.calculate_prediction(request)
        
        # 2. Pattern-based prediction (existing method)
        pattern_number, pattern_confidence, matches = self.find_pattern_matches(request)
        
        # 3. Combine predictions
        if physics_number is not None and pattern_number is not None:
            # Both methods have predictions
            if physics_number == pattern_number:
                # Strong agreement
                final_number = physics_number
                final_confidence = min(0.8, physics_confidence + pattern_confidence)
                method = "both_agree"
            else:
                # Disagreement - use weighted average based on confidence
                if physics_confidence > pattern_confidence:
                    final_number = physics_number
                    final_confidence = physics_confidence
                    method = "physics_preferred"
                else:
                    final_number = pattern_number
                    final_confidence = pattern_confidence
                    method = "pattern_preferred"
        elif physics_number is not None:
            # Only physics has prediction
            final_number = physics_number
            final_confidence = physics_confidence
            method = "physics_only"
        elif pattern_number is not None:
            # Only pattern has prediction
            final_number = pattern_number
            final_confidence = pattern_confidence
            method = "pattern_only"
        else:
            # No prediction
            final_number = None
            final_confidence = 0.0
            method = "none"
        
        # Store predictions for accuracy tracking
        if final_number is not None:
            self.predictions_made[request.round_id] = {
                'physics': physics_number,
                'pattern': pattern_number,
                'final': final_number,
                'method': method
            }
        
        return {
            'predicted_number': final_number,
            'confidence': round(final_confidence, 3),
            'method': method,
            'physics': {
                'number': physics_number,
                'confidence': round(physics_confidence, 3) if physics_confidence else 0,
                'debug': debug_info
            },
            'pattern': {
                'number': pattern_number,
                'confidence': round(pattern_confidence, 3) if pattern_confidence else 0,
                'matches': matches
            },
            'dataset_rows': self.total_records
        }
    
    def find_pattern_matches(self, request: PredictionRequest) -> Tuple[Optional[int], float, int]:
        """Find matching patterns (existing method improved)"""
        cell_key = self._get_cell_key(
            request.pos_x,
            request.pos_y,
            request.speed_ms_total,
            request.direction
        )
        
        # Collect matches with tighter criteria
        offset_weights = defaultdict(float)
        total_matches = 0
        
        # Search in smaller radius for better precision
        for radius in range(2):  # Only 0 and 1
            cells_to_check = self._get_neighbor_cells(cell_key, radius)
            
            for check_cell in cells_to_check:
                if check_cell in self.pattern_database:
                    for pattern in self.pattern_database[check_cell]:
                        # Filter by traveled pockets similarity
                        if abs(pattern.get('traveled_pockets', 7) - request.traveled_pockets) > 1:
                            continue
                        
                        offset = pattern['offset']
                        weight = 1.0 / (1 + radius)
                        offset_weights[offset] += weight
                        total_matches += 1
        
        if total_matches < Config.MIN_MATCHES_FOR_PREDICTION:
            return None, 0.0, total_matches
        
        if not offset_weights:
            return None, 0.0, 0
        
        # Get the offset with highest weight
        best_offset = max(offset_weights.items(), key=lambda x: x[1])[0]
        
        # Convert offset to predicted number
        predicted_number = WheelPhysics.get_number_at_distance(
            request.number_at_t2,
            best_offset,
            request.direction
        )
        
        # Calculate confidence
        confidence = self._calculate_pattern_confidence(offset_weights, total_matches)
        
        return predicted_number, confidence, total_matches
    
    def _get_neighbor_cells(self, center_cell: Tuple, radius: int) -> List[Tuple]:
        """Get neighboring grid cells at specified radius"""
        if radius == 0:
            return [center_cell]
        
        x, y, speed, direction = center_cell
        neighbors = []
        
        for dx in [-radius, 0, radius]:
            for dy in [-radius, 0, radius]:
                for ds in [-radius, 0, radius]:
                    if max(abs(dx), abs(dy), abs(ds)) == radius:
                        neighbor = (
                            x + dx * Config.POSITION_CELL_SIZE,
                            y + dy * Config.POSITION_CELL_SIZE,
                            speed + ds * Config.SPEED_CELL_SIZE,
                            direction
                        )
                        neighbors.append(neighbor)
        
        return neighbors
    
    def _calculate_pattern_confidence(self, offset_weights: Dict[int, float], total_matches: int) -> float:
        """Calculate confidence score based on pattern consistency"""
        if total_matches < Config.MIN_MATCHES_FOR_PREDICTION:
            return 0.0
        
        if not offset_weights:
            return 0.0
        
        top_weight = max(offset_weights.values())
        total_weight = sum(offset_weights.values())
        
        if total_weight == 0:
            return 0.0
            
        consistency = top_weight / total_weight
        
        # Return confidence based on consistency and match count
        base_confidence = 0.3
        if consistency >= 0.7 and total_matches >= 10:
            base_confidence = 0.5
        elif consistency >= 0.6:
            base_confidence = 0.4
            
        return base_confidence
    
    def finalize_round(self, round_id: str, winning_number: int) -> Dict[str, Any]:
        """Complete round with winning number and update both systems"""
        if round_id not in self.pending_rounds:
            return {
                "success": False,
                "error": "Round not found in pending storage"
            }
        
        pending = self.pending_rounds[round_id]
        round_data = pending['data']
        
        # Calculate offset from T2 to winning number
        offset = WheelPhysics.calculate_pocket_distance(
            round_data['number_at_t2'],
            winning_number,
            round_data['direction']
        )
        
        # Get prediction errors if we made predictions
        physics_error = None
        pattern_error = None
        
        if round_id in self.predictions_made:
            pred = self.predictions_made[round_id]
            
            if pred['physics'] is not None:
                physics_error = WheelPhysics.calculate_pocket_distance(
                    pred['physics'],
                    winning_number,
                    round_data['direction']
                )
            
            if pred['pattern'] is not None:
                pattern_error = WheelPhysics.calculate_pocket_distance(
                    pred['pattern'],
                    winning_number,
                    round_data['direction']
                )
        
        # Prepare complete record
        complete_record = {
            'timestamp': pending['timestamp'].isoformat(),
            'round_id': round_id,
            'table_id': round_data['table_id'],
            'pos_x': round_data['pos_x'],
            'pos_y': round_data['pos_y'],
            'speed_ms': round_data['speed_ms_total'],
            'traveled_pockets': round_data['traveled_pockets'],
            'direction': round_data['direction'],
            'number_t1': round_data['number_at_t1'],
            'number_t2': round_data['number_at_t2'],
            'winning_number': winning_number,
            'offset_from_t2': offset,
            'predicted_physics': self.predictions_made.get(round_id, {}).get('physics'),
            'predicted_pattern': self.predictions_made.get(round_id, {}).get('pattern'),
            'error_physics': physics_error,
            'error_pattern': pattern_error
        }
        
        # Save to database
        self._save_to_csv(complete_record)
        
        # Update pattern database
        cell_key = self._get_cell_key(
            round_data['pos_x'],
            round_data['pos_y'],
            round_data['speed_ms_total'],
            round_data['direction']
        )
        
        pattern_data = {
            'offset': offset,
            'timestamp': complete_record['timestamp'],
            'traveled_pockets': round_data['traveled_pockets']
        }
        
        self.pattern_database[cell_key].append(pattern_data)
        
        # Update physics calibration
        self.physics_engine.update_calibration(round_data, offset)
        
        # Enforce limits
        if len(self.pattern_database[cell_key]) > Config.MAX_RECORDS_PER_CELL:
            self.pattern_database[cell_key] = self.pattern_database[cell_key][-Config.MAX_RECORDS_PER_CELL:]
        
        self.total_records += 1
        
        # Clean up
        del self.pending_rounds[round_id]
        if round_id in self.predictions_made:
            del self.predictions_made[round_id]
        
        # Prepare response
        response = {
            "success": True,
            "offset": offset,
            "total_records": self.total_records
        }
        
        if physics_error is not None:
            response["physics_error"] = abs(physics_error)
            response["physics_accuracy"] = "accurate" if abs(physics_error) <= 4 else "needs_improvement"
        
        if pattern_error is not None:
            response["pattern_error"] = abs(pattern_error)
            response["pattern_accuracy"] = "accurate" if abs(pattern_error) <= 4 else "needs_improvement"
        
        logger.info(f"Finalized round {round_id}: winning={winning_number}, "
                   f"physics_error={physics_error}, pattern_error={pattern_error}")
        
        return response
    
    def _save_to_csv(self, record: Dict[str, Any]):
        """Append record to CSV database"""
        try:
            with open(self.data_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=Config.CSV_COLUMNS)
                writer.writerow(record)
        except Exception as e:
            logger.error(f"Failed to save record: {e}")
    
    def cleanup_pending_rounds(self):
        """Remove pending rounds older than timeout"""
        now = datetime.now()
        expired = []
        
        for round_id, data in self.pending_rounds.items():
            if now - data['timestamp'] > timedelta(minutes=Config.PENDING_TIMEOUT_MINUTES):
                expired.append(round_id)
        
        for round_id in expired:
            del self.pending_rounds[round_id]
            if round_id in self.predictions_made:
                del self.predictions_made[round_id]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired pending rounds")

# ============================================================================
# API SERVER
# ============================================================================

app = FastAPI(
    title="Enhanced Roulette Physics Server",
    description="Hybrid physics and pattern matching prediction system",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Initialize storage
storage = HybridStorage()

@app.on_event("startup")
async def startup_event():
    """Run maintenance tasks on server startup"""
    storage.cleanup_pending_rounds()
    logger.info("Server started with hybrid physics + pattern matching engine")

@app.get("/")
async def status():
    """Server status and statistics"""
    return {
        "server": "Enhanced Roulette Physics Server",
        "version": "3.0.0",
        "status": "operational",
        "engine": "hybrid_physics_pattern",
        "statistics": {
            "total_records": storage.total_records,
            "pattern_cells": len(storage.pattern_database),
            "physics_positions": len(storage.physics_engine.friction_map),
            "pending_rounds": len(storage.pending_rounds),
            "active_predictions": len(storage.predictions_made)
        },
        "configuration": {
            "min_speed_ms": Config.MIN_BALL_SPEED_MS,
            "max_speed_ms": Config.MAX_BALL_SPEED_MS,
            "position_precision": Config.POSITION_CELL_SIZE,
            "speed_precision": Config.SPEED_CELL_SIZE,
            "physics_enabled": True,
            "pattern_matching_enabled": True
        }
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Process prediction request using hybrid approach"""
    try:
        # Validate and store pending round
        if not storage.store_pending(request):
            return {
                "predicted_number": None,
                "confidence": 0,
                "error": "Invalid data - round rejected",
                "dataset_rows": storage.total_records
            }
        
        # Get hybrid prediction
        result = storage.get_hybrid_prediction(request)
        
        # Format response
        if result['predicted_number'] is not None:
            # Calculate predicted sector for better accuracy representation
            sector = WheelPhysics.get_sector(result['predicted_number'], Config.SECTOR_SIZE)
            
            response = {
                "predicted_number": result['predicted_number'],
                "confidence": result['confidence'],
                "predicted_sector": sector,
                "method": result['method'],
                "dataset_rows": result['dataset_rows'],
                "physics_prediction": result['physics']['number'],
                "pattern_prediction": result['pattern']['number'],
                "matches_found": result['pattern']['matches']
            }
            
            # Add confidence assessment
            if result['confidence'] >= 0.6:
                response['confidence_level'] = "high"
            elif result['confidence'] >= 0.4:
                response['confidence_level'] = "medium"
            else:
                response['confidence_level'] = "low"
            
            return response
        else:
            return {
                "predicted_number": None,
                "confidence": 0,
                "error": "Insufficient data for prediction",
                "dataset_rows": result['dataset_rows'],
                "matches_found": result['pattern']['matches'],
                "physics_status": "calculating" if storage.total_records < 100 else "ready"
            }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_winner")
async def log_winner(request: WinnerRequest):
    """Log winning number for completed round"""
    try:
        result = storage.finalize_round(request.round_id, request.winning_number)
        
        if result["success"]:
            response = {
                "ok": True,
                "stored": True,
                "dataset_rows": result["total_records"],
                "offset_recorded": result["offset"]
            }
            
            # Add accuracy metrics if available
            if "physics_error" in result:
                response["physics_error"] = result["physics_error"]
                response["physics_accuracy"] = result["physics_accuracy"]
            
            if "pattern_error" in result:
                response["pattern_error"] = result["pattern_error"]
                response["pattern_accuracy"] = result["pattern_accuracy"]
            
            # Overall accuracy assessment
            if "physics_error" in result and "pattern_error" in result:
                avg_error = (result["physics_error"] + result["pattern_error"]) / 2
                response["average_error"] = round(avg_error, 1)
                response["overall_accuracy"] = "accurate" if avg_error <= 4 else "improving"
            
            return response
        else:
            return {
                "ok": False,
                "error": result["error"],
                "dataset_rows": storage.total_records
            }
    
    except Exception as e:
        logger.error(f"Winner logging error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def statistics():
    """Detailed system statistics"""
    # Calculate physics calibration coverage
    physics_coverage = len(storage.physics_engine.friction_map)
    
    # Get accuracy statistics from recent predictions
    recent_errors_physics = []
    recent_errors_pattern = []
    
    # This would need to be implemented to track recent prediction errors
    
    return {
        "total_records": storage.total_records,
        "pattern_cells": len(storage.pattern_database),
        "physics_calibrated_positions": physics_coverage,
        "pending_rounds": len(storage.pending_rounds),
        "predictions_tracking": len(storage.predictions_made),
        "database_path": storage.data_path,
        "engine_type": "hybrid_physics_pattern",
        "physics_constants": {
            "wheel_radius": Config.WHEEL_RADIUS,
            "critical_velocity": Config.CRITICAL_VELOCITY,
            "base_friction": Config.BASE_FRICTION_COEFF
        }
    }

@app.get("/calibration/{table_id}")
async def get_calibration(table_id: str = "default"):
    """Get physics calibration data for specific table"""
    friction_data = {}
    
    for pos_key, friction in storage.physics_engine.friction_map.items():
        key_str = f"{pos_key[0]},{pos_key[1]}"
        friction_data[key_str] = {
            "friction_coefficient": round(friction, 5),
            "calibration_points": len(storage.physics_engine.calibration_data.get(pos_key, []))
        }
    
    return {
        "table_id": table_id,
        "calibrated_positions": len(friction_data),
        "friction_map": friction_data
    }

@app.delete("/clear_pending")
async def clear_pending():
    """Clear all pending rounds (admin function)"""
    count = len(storage.pending_rounds)
    storage.pending_rounds.clear()
    storage.predictions_made.clear()
    return {"cleared": count, "message": f"Cleared {count} pending rounds"}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("PHYSICS-ONLY ROULETTE PREDICTION SERVER v2.0.0")
    print("Physics-based trajectory calculation engine")
    print("="*70)
    print(f"Database: {storage.data_path}")
    print(f"Records loaded: {storage.total_records}")
    print(f"Physics positions: {len(storage.physics_engine.friction_map)}")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
