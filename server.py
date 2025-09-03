"""
Enhanced Roulette Prediction Server with Corrected Physics Model
Three-point measurement system with proper ball/rotor velocity calculations
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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('roulette_enhanced_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedRouletteServer")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Server configuration with corrected physics constants"""
    
    # Data management
    MAX_RECORDS = 2000000
    PENDING_TIMEOUT_MINUTES = 10
    DATA_RETENTION_DAYS = 90
    
    # Physics constants
    GRAVITY = 9.81                # m/s²
    WHEEL_RADIUS = 0.38           # meters (standard roulette wheel)
    BALL_RADIUS = 0.01            # meters
    CRITICAL_VELOCITY = 0.8       # rad/s - velocity when ball drops
    AIR_RESISTANCE_COEFF = 0.0001 # air drag coefficient
    
    # Prediction parameters
    MIN_CONFIDENCE_THRESHOLD = 0.4   # Raised threshold
    SECTOR_SIZE = 9                  # predict 9-number sectors (center + 4 each side)
    MIN_DECELERATION = 0.1          # Minimum valid deceleration
    MAX_TIME_TO_DROP = 10.0         # Maximum reasonable time to drop
    
    # Validation
    MIN_LAP_TIME_MS = 200
    MAX_LAP_TIME_MS = 3000
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
        'pos_x', 'pos_y', 'lap1_ms', 'lap2_ms',
        'rotor_shift', 'rotor_direction', 'ball_direction',
        'number_t1', 'number_t2', 'number_t3', 'phase_t2',
        'winning_number', 'offset_from_t3',
        'predicted_physics', 'predicted_sector', 
        'error_physics', 'confidence'
    ]

# ============================================================================
# DATA MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Enhanced prediction request with three-point data"""
    round_id: str = Field(..., min_length=5, max_length=50)
    pos_x: float = Field(..., ge=0, le=2000)
    pos_y: float = Field(..., ge=0, le=2000)
    lap1_ms: int = Field(..., ge=100, le=5000, description="Time for first lap (T1 to T2)")
    lap2_ms: int = Field(..., ge=100, le=5000, description="Time for second lap (T2 to T3)")
    rotor_shift: int = Field(..., ge=0, le=18, description="Rotor movement in pockets during lap1")
    rotor_direction: str = Field(..., description="Rotor direction: CW or CCW")
    direction: str = Field(..., description="Ball direction: CW or CCW")
    number_at_t1: int = Field(..., ge=0, le=36)
    number_at_t2: int = Field(..., ge=0, le=36)
    number_at_t3: int = Field(..., ge=0, le=36)
    phase_at_t2: float = Field(default=0.0, ge=0, lt=1, description="Fractional position within pocket")
    table_id: str = Field(default="default", max_length=50)
    
    @field_validator('lap1_ms', 'lap2_ms')
    @classmethod
    def validate_lap_times(cls, v):
        if v < Config.MIN_LAP_TIME_MS or v > Config.MAX_LAP_TIME_MS:
            raise ValueError(f"Lap time {v}ms out of valid range [{Config.MIN_LAP_TIME_MS}, {Config.MAX_LAP_TIME_MS}]")
        return v
    
    @field_validator('direction', 'rotor_direction')
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
    """Enhanced physics calculations using three-point measurement"""
    
    def __init__(self):
        self.scatter_models = defaultdict(list)  # Position-based scatter patterns
        self.calibration_data = defaultdict(list)
    
    def calculate_prediction(self, request: PredictionRequest) -> Tuple[Optional[int], float, List[int], Dict]:
        """
        Calculate predicted number using corrected physics model
        Returns: (predicted_number, confidence, predicted_sector, debug_info)
        """
        try:
            # 1. Calculate absolute ball velocities (rad/s)
            omega_ball_t2 = (2 * math.pi) / (request.lap1_ms / 1000.0)
            omega_ball_t3 = (2 * math.pi) / (request.lap2_ms / 1000.0)
            
            # 2. Calculate ball deceleration
            time_between_measurements = request.lap2_ms / 1000.0
            ball_deceleration = (omega_ball_t3 - omega_ball_t2) / time_between_measurements
            
            # Validate deceleration
            if ball_deceleration >= -Config.MIN_DECELERATION:
                return None, 0.0, [], {"error": "Insufficient deceleration", "decel": round(ball_deceleration, 3)}
            
            # 3. Calculate rotor velocity
            rotor_pockets_per_sec = request.rotor_shift / (request.lap1_ms / 1000.0)
            omega_rotor = (2 * math.pi / 37) * rotor_pockets_per_sec
            
            # Apply rotor direction
            if request.rotor_direction == "CCW":
                omega_rotor = -omega_rotor
            
            # 4. Time to critical velocity (when ball drops)
            if omega_ball_t3 <= Config.CRITICAL_VELOCITY:
                return None, 0.0, [], {"error": "Already below critical velocity"}
            
            time_to_drop = (omega_ball_t3 - Config.CRITICAL_VELOCITY) / (-ball_deceleration)
            
            if time_to_drop > Config.MAX_TIME_TO_DROP:
                return None, 0.0, [], {"error": "Time to drop too large", "ttd": round(time_to_drop, 2)}
            
            # 5. Calculate ball position at drop (in radians)
            ball_angle_traveled = (omega_ball_t3 * time_to_drop + 
                                 0.5 * ball_deceleration * time_to_drop ** 2)
            
            # 6. Calculate rotor position at drop
            rotor_angle_traveled = omega_rotor * time_to_drop
            
            # 7. Relative position (ball - rotor) in radians
            relative_angle = ball_angle_traveled - rotor_angle_traveled
            
            # 8. Convert to pockets
            pockets_offset = relative_angle * 37 / (2 * math.pi)
            
            # Account for phase within pocket at T2
            pockets_offset += request.phase_at_t2
            
            # Round to get discrete pocket offset
            pockets_offset_int = int(round(pockets_offset))
            
            # 9. Calculate predicted number
            predicted_number = WheelPhysics.get_number_at_distance(
                request.number_at_t3,
                pockets_offset_int,
                request.direction
            )
            
            # 10. Calculate confidence based on measurement quality
            confidence = self.calculate_confidence(
                ball_deceleration,
                time_to_drop,
                request.lap1_ms,
                request.lap2_ms
            )
            
            # 11. Get predicted sector (with scatter model)
            predicted_sector = self.get_predicted_sector(
                predicted_number,
                request.pos_x,
                request.pos_y
            )
            
            debug_info = {
                "omega_ball_t2": round(omega_ball_t2, 3),
                "omega_ball_t3": round(omega_ball_t3, 3),
                "deceleration": round(ball_deceleration, 3),
                "omega_rotor": round(omega_rotor, 3),
                "time_to_drop": round(time_to_drop, 2),
                "pockets_offset": round(pockets_offset, 2),
                "phase_contribution": round(request.phase_at_t2, 3)
            }
            
            return predicted_number, confidence, predicted_sector, debug_info
            
        except Exception as e:
            logger.error(f"Physics calculation error: {e}")
            return None, 0.0, [], {"error": str(e)}
    
    def calculate_confidence(self, deceleration: float, time_to_drop: float, 
                           lap1_ms: int, lap2_ms: int) -> float:
        """Calculate confidence based on measurement quality"""
        confidence = 0.5  # Base confidence
        
        # Deceleration stability
        if -2.0 < deceleration < -0.3:
            confidence += 0.2
        
        # Time to drop reasonableness
        if 0.5 < time_to_drop < 5.0:
            confidence += 0.1
        
        # Lap time consistency
        lap_ratio = lap2_ms / lap1_ms
        if 1.05 < lap_ratio < 1.5:  # Ball should be slowing down
            confidence += 0.1
        
        # Cap confidence
        return min(0.8, max(0.3, confidence))
    
    def get_predicted_sector(self, center_number: int, pos_x: float, pos_y: float) -> List[int]:
        """Get sector of numbers accounting for scatter"""
        # Get scatter model for this position
        scatter_data = self.get_scatter_distribution(pos_x, pos_y)
        
        # Default sector size
        sector_size = Config.SECTOR_SIZE
        
        # Adjust based on scatter data if available
        if scatter_data and len(scatter_data) > 10:
            # Calculate standard deviation of scatter
            scatter_std = np.std([d['offset'] for d in scatter_data])
            # Adjust sector size based on scatter
            sector_size = min(13, max(7, int(2 * scatter_std + 1)))
        
        return WheelPhysics.get_sector(center_number, sector_size)
    
    def get_scatter_distribution(self, pos_x: float, pos_y: float) -> List[Dict]:
        """Get historical scatter data for position"""
        # Grid position
        grid_x = round(pos_x / 50) * 50
        grid_y = round(pos_y / 50) * 50
        pos_key = (grid_x, grid_y)
        
        return self.scatter_models.get(pos_key, [])
    
    def update_scatter_model(self, request_data: dict, predicted_offset: int, actual_offset: int):
        """Update scatter model with actual results"""
        scatter = actual_offset - predicted_offset
        
        grid_x = round(request_data['pos_x'] / 50) * 50
        grid_y = round(request_data['pos_y'] / 50) * 50
        pos_key = (grid_x, grid_y)
        
        self.scatter_models[pos_key].append({
            'offset': scatter,
            'timestamp': datetime.now()
        })
        
        # Keep only recent data
        if len(self.scatter_models[pos_key]) > 100:
            self.scatter_models[pos_key] = self.scatter_models[pos_key][-100:]

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
    def get_sector(center_number: int, size: int = 9) -> List[int]:
        """Get sector of numbers around center"""
        center_idx = Config.POCKET_TO_INDEX.get(center_number, 0)
        sector = []
        
        half_size = size // 2
        for offset in range(-half_size, half_size + 1):
            idx = (center_idx + offset) % 37
            sector.append(Config.WHEEL_LAYOUT[idx])
            
        return sector

# ============================================================================
# ENHANCED STORAGE SYSTEM
# ============================================================================

class EnhancedStorage:
    """Storage system for three-point measurement data"""
    
    def __init__(self):
        self.data_path = self._get_data_path()
        self.pending_rounds = {}
        self.physics_engine = PhysicsEngine()
        self.total_records = 0
        self.predictions_made = {}
        
        self._initialize_storage()
        self._load_existing_data()
    
    def _get_data_path(self) -> str:
        """Determine optimal data storage location"""
        candidates = [
            os.getenv("ROULETTE_DATA_PATH", ""),
            os.path.expanduser("~/.roulette_enhanced/database.csv"),
            "./roulette_enhanced_database.csv"
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
        """Load historical data for scatter model training"""
        if not os.path.exists(self.data_path):
            return
            
        try:
            cutoff_date = datetime.now() - timedelta(days=Config.DATA_RETENTION_DAYS)
            loaded = 0
            
            with open(self.data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        row_timestamp = datetime.fromisoformat(row['timestamp'])
                        if row_timestamp < cutoff_date:
                            continue
                        
                        # Update scatter model if we have prediction and result
                        if all(k in row and row[k] for k in ['predicted_physics', 'winning_number', 'number_t3']):
                            predicted_num = int(row['predicted_physics'])
                            winning_num = int(row['winning_number'])
                            t3_num = int(row['number_t3'])
                            
                            predicted_offset = WheelPhysics.calculate_pocket_distance(
                                t3_num, predicted_num, row['ball_direction']
                            )
                            actual_offset = WheelPhysics.calculate_pocket_distance(
                                t3_num, winning_num, row['ball_direction']
                            )
                            
                            self.physics_engine.update_scatter_model(
                                {'pos_x': float(row['pos_x']), 'pos_y': float(row['pos_y'])},
                                predicted_offset,
                                actual_offset
                            )
                        
                        loaded += 1
                        
                    except (ValueError, KeyError) as e:
                        continue
            
            self.total_records = loaded
            logger.info(f"Loaded {loaded} records for scatter model training")
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
    
    def store_pending(self, request: PredictionRequest) -> bool:
        """Store round data temporarily until winning number arrives"""
        # Store pending data
        self.pending_rounds[request.round_id] = {
            'timestamp': datetime.now(),
            'data': request.dict(),
            'physics_prediction': None,
            'physics_confidence': None,
            'predicted_sector': None
        }
        
        logger.info(f"Stored pending round {request.round_id} with three-point data")
        return True
    
    def get_prediction(self, request: PredictionRequest) -> Dict[str, Any]:
        """Get prediction using enhanced physics"""
        
        # Physics-based prediction
        physics_number, confidence, predicted_sector, debug_info = self.physics_engine.calculate_prediction(request)
        
        # Store prediction for accuracy tracking
        if physics_number is not None and confidence >= Config.MIN_CONFIDENCE_THRESHOLD:
            self.predictions_made[request.round_id] = {
                'physics': physics_number,
                'confidence': confidence,
                'sector': predicted_sector
            }
            
            # Update pending round
            if request.round_id in self.pending_rounds:
                self.pending_rounds[request.round_id]['physics_prediction'] = physics_number
                self.pending_rounds[request.round_id]['physics_confidence'] = confidence
                self.pending_rounds[request.round_id]['predicted_sector'] = predicted_sector
        
        return {
            'predicted_number': physics_number if confidence >= Config.MIN_CONFIDENCE_THRESHOLD else None,
            'confidence': round(confidence, 3),
            'predicted_sector': predicted_sector if physics_number else [],
            'dataset_rows': self.total_records,
            'physics': {
                'confidence': round(confidence, 3),
                'debug': debug_info
            }
        }
    
    def finalize_round(self, round_id: str, winning_number: int) -> Dict[str, Any]:
        """Complete round with winning number and update models"""
        if round_id not in self.pending_rounds:
            return {
                "success": False,
                "error": "Round not found in pending storage"
            }
        
        pending = self.pending_rounds[round_id]
        round_data = pending['data']
        
        # Calculate offset from T3 to winning number
        offset = WheelPhysics.calculate_pocket_distance(
            round_data['number_at_t3'],
            winning_number,
            round_data['direction']
        )
        
        # Calculate prediction error if we made a prediction
        physics_error = None
        if pending['physics_prediction'] is not None:
            physics_error = WheelPhysics.calculate_pocket_distance(
                pending['physics_prediction'],
                winning_number,
                round_data['direction']
            )
            
            # Update scatter model
            predicted_offset = WheelPhysics.calculate_pocket_distance(
                round_data['number_at_t3'],
                pending['physics_prediction'],
                round_data['direction']
            )
            self.physics_engine.update_scatter_model(
                round_data,
                predicted_offset,
                offset
            )
        
        # Prepare complete record
        complete_record = {
            'timestamp': pending['timestamp'].isoformat(),
            'round_id': round_id,
            'table_id': round_data['table_id'],
            'pos_x': round_data['pos_x'],
            'pos_y': round_data['pos_y'],
            'lap1_ms': round_data['lap1_ms'],
            'lap2_ms': round_data['lap2_ms'],
            'rotor_shift': round_data['rotor_shift'],
            'rotor_direction': round_data['rotor_direction'],
            'ball_direction': round_data['direction'],
            'number_t1': round_data['number_at_t1'],
            'number_t2': round_data['number_at_t2'],
            'number_t3': round_data['number_at_t3'],
            'phase_t2': round_data['phase_at_t2'],
            'winning_number': winning_number,
            'offset_from_t3': offset,
            'predicted_physics': pending['physics_prediction'],
            'predicted_sector': json.dumps(pending['predicted_sector']) if pending['predicted_sector'] else None,
            'error_physics': physics_error,
            'confidence': pending['physics_confidence']
        }
        
        # Save to database
        self._save_to_csv(complete_record)
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
            response["overall_accuracy"] = "accurate" if abs(physics_error) <= 4 else "improving"
        
        logger.info(f"Finalized round {round_id}: winning={winning_number}, "
                   f"physics_error={physics_error}")
        
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
    description="Three-point measurement prediction system with corrected physics",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Initialize storage
storage = EnhancedStorage()

@app.on_event("startup")
async def startup_event():
    """Run maintenance tasks on server startup"""
    storage.cleanup_pending_rounds()
    logger.info("Server started with enhanced three-point measurement system")
    logger.info(f"Total records: {storage.total_records}")

@app.get("/")
async def status():
    """Server status and statistics"""
    return {
        "server": "Enhanced Roulette Physics Server",
        "version": "5.0.0",
        "status": "operational",
        "engine": "three_point_physics",
        "statistics": {
            "total_records": storage.total_records,
            "pending_rounds": len(storage.pending_rounds),
            "active_predictions": len(storage.predictions_made),
            "scatter_positions": len(storage.physics_engine.scatter_models)
        },
        "configuration": {
            "measurement_points": 3,
            "physics_model": "absolute_velocity",
            "scatter_modeling": True,
            "sector_size": Config.SECTOR_SIZE,
            "confidence_threshold": Config.MIN_CONFIDENCE_THRESHOLD
        }
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Process prediction request using enhanced physics"""
    try:
        # Validate and store pending round
        if not storage.store_pending(request):
            return {
                "predicted_number": None,
                "confidence": 0,
                "error": "Invalid data - round rejected",
                "dataset_rows": storage.total_records
            }
        
        # Get physics prediction
        result = storage.get_prediction(request)
        
        # Format response
        if result['predicted_number'] is not None:
            response = {
                "predicted_number": result['predicted_number'],
                "confidence": result['confidence'],
                "predicted_sector": result['predicted_sector'],
                "dataset_rows": result['dataset_rows'],
                "physics_confidence": result['physics']['confidence'],
                "debug": result['physics']['debug']
            }
            
            # Add confidence assessment
            if result['confidence'] >= 0.7:
                response['confidence_level'] = "high"
                response['recommendation'] = f"Bet on sector: {result['predicted_sector']}"
            elif result['confidence'] >= 0.5:
                response['confidence_level'] = "medium"
                response['recommendation'] = f"Consider sector: {result['predicted_sector']}"
            else:
                response['confidence_level'] = "low"
                response['recommendation'] = "No bet - insufficient confidence"
            
            logger.info(f"Prediction for round {request.round_id}: "
                       f"Number={result['predicted_number']}, "
                       f"Confidence={result['confidence']}")
            
            return response
        else:
            reason = result['physics']['debug'].get('error', 'Insufficient confidence')
            return {
                "predicted_number": None,
                "confidence": 0,
                "error": f"No prediction: {reason}",
                "dataset_rows": result['dataset_rows'],
                "debug": result['physics']['debug']
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
                response["overall_accuracy"] = result["overall_accuracy"]
            
            logger.info(f"Winner logged for round {request.round_id}: "
                       f"Number={request.winning_number}")
            
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
    scatter_stats = {}
    for pos_key, scatter_data in storage.physics_engine.scatter_models.items():
        if scatter_data:
            offsets = [d['offset'] for d in scatter_data]
            scatter_stats[f"{pos_key[0]},{pos_key[1]}"] = {
                "samples": len(scatter_data),
                "mean_scatter": round(np.mean(offsets), 2),
                "std_scatter": round(np.std(offsets), 2)
            }
    
    return {
        "total_records": storage.total_records,
        "pending_rounds": len(storage.pending_rounds),
        "active_predictions": len(storage.predictions_made),
        "database_path": storage.data_path,
        "engine_type": "three_point_physics",
        "scatter_models": scatter_stats,
        "physics_constants": {
            "critical_velocity": Config.CRITICAL_VELOCITY,
            "min_deceleration": Config.MIN_DECELERATION,
            "sector_size": Config.SECTOR_SIZE
        }
    }

@app.get("/calibration/{table_id}")
async def get_calibration(table_id: str = "default"):
    """Get scatter calibration data for specific table"""
    scatter_data = {}
    
    for pos_key, data in storage.physics_engine.scatter_models.items():
        if data:
            key_str = f"{pos_key[0]},{pos_key[1]}"
            offsets = [d['offset'] for d in data]
            scatter_data[key_str] = {
                "calibration_points": len(data),
                "mean_scatter": round(np.mean(offsets), 2),
                "std_scatter": round(np.std(offsets), 2),
                "recommended_sector_size": min(13, max(7, int(2 * np.std(offsets) + 1)))
            }
    
    return {
        "table_id": table_id,
        "calibrated_positions": len(scatter_data),
        "scatter_data": scatter_data
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
    print("ENHANCED ROULETTE PHYSICS SERVER v5.0.0")
    print("Three-point measurement system with corrected physics")
    print("="*70)
    print(f"Database: {storage.data_path}")
    print(f"Records loaded: {storage.total_records}")
    print(f"Scatter models: {len(storage.physics_engine.scatter_models)}")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
