"""
Professional Roulette Prediction Server
High-accuracy pattern matching system for Evolution Gaming roulette
"""

import os
import csv
import json
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

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('roulette_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RouletteServer")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Server configuration parameters optimized for accuracy"""
    # Grid cell sizes for pattern matching - tighter grid for better accuracy
    POSITION_CELL_SIZE = 15  # pixels - reduced from 5 for more precise matching
    SPEED_CELL_SIZE = 15     # milliseconds - wider tolerance for timing variations
    
    # Data management
    MAX_RECORDS_PER_CELL = 100
    GLOBAL_MAX_RECORDS = 500000
    PENDING_TIMEOUT_MINUTES = 10
    DATA_RETENTION_DAYS = 30  # Keep only recent data for predictions
    
    # Search parameters
    MAX_SEARCH_RADIUS = 1  # Reduced from 3 - only look at very close matches
    MIN_MATCHES_FOR_PREDICTION = 10  # Increased from 5 - need more confidence
    
    # Validation
    MIN_BALL_SPEED_MS = 300  # Reduced from 400
    MAX_BALL_SPEED_MS = 3000
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
        'winning_number', 'offset_from_t2'
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
    direction_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    
    @field_validator('speed_ms_total')
    @classmethod
    def validate_speed(cls, v):
        if v < Config.MIN_BALL_SPEED_MS:
            raise ValueError(f"Speed {v}ms below minimum {Config.MIN_BALL_SPEED_MS}ms")
        return v
    
    @field_validator('direction')
    @classmethod
    def validate_direction(cls, v):
        if v not in ['CW', 'CCW']:
            raise ValueError('Direction must be CW or CCW')
        return v

class WinnerRequest(BaseModel):
    """Winning number notification from client"""
    round_id: str = Field(..., min_length=5, max_length=50)
    winning_number: int = Field(..., ge=0, le=36)

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

class WheelPhysics:
    """Handle roulette wheel physics and calculations"""
    
    @staticmethod
    def calculate_pocket_distance(from_number: int, to_number: int, direction: str) -> int:
        """
        Calculate pocket distance between two numbers in given direction
        Returns positive value for forward movement, negative for backward
        """
        from_idx = Config.POCKET_TO_INDEX.get(from_number)
        to_idx = Config.POCKET_TO_INDEX.get(to_number)
        
        # FIXED: Handle invalid pocket numbers gracefully
        if from_idx is None or to_idx is None:
            logger.warning(f"Invalid pocket numbers: from={from_number}, to={to_number}")
            return 0  # Return 0 instead of raising exception
        
        if direction == "CW":
            # Clockwise: calculate forward distance
            distance = (to_idx - from_idx) % 37
        else:  # CCW
            # Counter-clockwise: calculate backward distance
            distance = (from_idx - to_idx) % 37
        
        # Normalize to -18 to +18 range (shortest path)
        if distance > 18:
            distance = distance - 37
            
        return distance
    
    @staticmethod
    def get_number_at_distance(from_number: int, distance: int, direction: str) -> int:
        """Get pocket number at specified distance from reference"""
        from_idx = Config.POCKET_TO_INDEX.get(from_number)
        
        # FIXED: Handle invalid pocket number gracefully
        if from_idx is None:
            logger.warning(f"Invalid pocket number: {from_number}")
            return from_number  # Return same number instead of error
        
        if direction == "CW":
            target_idx = (from_idx + distance) % 37
        else:  # CCW
            target_idx = (from_idx - distance) % 37
            
        return Config.WHEEL_LAYOUT[target_idx]

class GridSystem:
    """Grid-based indexing for pattern matching"""
    
    @staticmethod
    def get_cell_key(x: float, y: float, speed: int, direction: str) -> Tuple:
        """Convert position and speed to grid cell coordinates"""
        x_cell = round(x / Config.POSITION_CELL_SIZE) * Config.POSITION_CELL_SIZE
        y_cell = round(y / Config.POSITION_CELL_SIZE) * Config.POSITION_CELL_SIZE
        speed_cell = round(speed / Config.SPEED_CELL_SIZE) * Config.SPEED_CELL_SIZE
        return (x_cell, y_cell, speed_cell, direction)

# ============================================================================
# DATA STORAGE
# ============================================================================

class DataStorage:
    """Professional data storage and pattern matching system"""
    
    def __init__(self):
        self.data_path = self._get_data_path()
        self.pending_rounds = {}  # Temporary storage for incomplete rounds
        self.pattern_database = defaultdict(list)  # Grid cells with historical data
        self.total_records = 0
        self.predictions_made = {}  # Track predictions for accuracy analysis
        
        self._initialize_storage()
        self._load_existing_data()
        
    def _get_data_path(self) -> str:
        """Determine optimal data storage location"""
        candidates = [
            os.getenv("ROULETTE_DATA_PATH", ""),
            os.path.expanduser("~/.roulette_server/database.csv"),
            "/var/lib/roulette/database.csv",
            "./roulette_database.csv"
        ]
        
        for path in candidates:
            if not path:
                continue
            try:
                directory = os.path.dirname(path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                
                # Test write permissions
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
        """Load historical data into memory for pattern matching - only recent data"""
        if not os.path.exists(self.data_path):
            return
            
        try:
            cutoff_date = datetime.now() - timedelta(days=Config.DATA_RETENTION_DAYS)
            
            with open(self.data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                loaded = 0
                
                for row in reader:
                    # Validate row data
                    if not all(k in row for k in ['pos_x', 'pos_y', 'speed_ms', 'direction', 'offset_from_t2', 'timestamp']):
                        continue
                    
                    try:
                        # Check if data is recent enough
                        row_timestamp = datetime.fromisoformat(row['timestamp'])
                        if row_timestamp < cutoff_date:
                            continue  # Skip old data
                        
                        # Parse and validate
                        x = float(row['pos_x'])
                        y = float(row['pos_y'])
                        speed = int(row['speed_ms'])
                        direction = row['direction']
                        traveled_pockets = int(row.get('traveled_pockets', 7))
                        
                        if direction not in Config.VALID_DIRECTIONS:
                            continue
                        
                        # Add to grid cell
                        cell_key = GridSystem.get_cell_key(x, y, speed, direction)
                        
                        # Keep only essential data in memory
                        pattern_data = {
                            'offset': int(row['offset_from_t2']),
                            'timestamp': row['timestamp'],
                            'confidence': float(row.get('confidence', 1.0)),
                            'traveled_pockets': traveled_pockets
                        }
                        
                        self.pattern_database[cell_key].append(pattern_data)
                        loaded += 1
                        
                        # Enforce cell limit
                        if len(self.pattern_database[cell_key]) > Config.MAX_RECORDS_PER_CELL:
                            self.pattern_database[cell_key] = self.pattern_database[cell_key][-Config.MAX_RECORDS_PER_CELL:]
                    
                    except (ValueError, KeyError) as e:
                        continue
                
                self.total_records = loaded
                logger.info(f"Loaded {loaded} recent records (last {Config.DATA_RETENTION_DAYS} days) into {len(self.pattern_database)} grid cells")
                
        except Exception as e:
            logger.error(f"Error loading database: {e}")
    
    def store_pending(self, request: PredictionRequest) -> bool:
        """Store round data temporarily until winning number arrives"""
        # If there are pending rounds waiting, clear them all
        # New data means previous round ended without logging winner
        if self.pending_rounds:
            old_count = len(self.pending_rounds)
            self.pending_rounds.clear()
            # Also clear any predictions made for those rounds
            self.predictions_made.clear()
            logger.info(f"Cleared {old_count} pending rounds - new round data received")
        
        # Validate direction
        if request.direction not in Config.VALID_DIRECTIONS:
            logger.warning(f"Invalid direction {request.direction} - rejecting round {request.round_id}")
            return False
        
        # Store pending data
        self.pending_rounds[request.round_id] = {
            'timestamp': datetime.now(),
            'data': request.dict(),
            'prediction': None
        }
        
        logger.info(f"Stored pending round {request.round_id}, direction: {request.direction}, speed: {request.speed_ms_total}ms")
        return True
    
    def finalize_round(self, round_id: str, winning_number: int) -> Dict[str, Any]:
        """Complete round with winning number and store in database"""
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
            'offset_from_t2': offset
        }
        
        # Save to database
        self._save_to_csv(complete_record)
        
        # Add to pattern database
        cell_key = GridSystem.get_cell_key(
            round_data['pos_x'],
            round_data['pos_y'],
            round_data['speed_ms_total'],
            round_data['direction']
        )
        
        pattern_data = {
            'offset': offset,
            'timestamp': complete_record['timestamp'],
            'confidence': round_data.get('direction_confidence', 1.0),
            'traveled_pockets': round_data['traveled_pockets']
        }
        
        self.pattern_database[cell_key].append(pattern_data)
        
        # Enforce limits
        if len(self.pattern_database[cell_key]) > Config.MAX_RECORDS_PER_CELL:
            self.pattern_database[cell_key] = self.pattern_database[cell_key][-Config.MAX_RECORDS_PER_CELL:]
        
        self.total_records += 1
        
        # Calculate prediction accuracy if we made one
        prediction_error = None
        if round_id in self.predictions_made:
            predicted_offset = self.predictions_made[round_id]['main_offset']
            prediction_error = abs(predicted_offset - offset)
            
            if prediction_error >= 16:  # Half-wheel error
                logger.warning(f"Large prediction error ({prediction_error} pockets) for round {round_id}")
            
            del self.predictions_made[round_id]
        
        # Clean up
        del self.pending_rounds[round_id]
        
        logger.info(f"Finalized round {round_id}: winning={winning_number}, offset={offset}, error={prediction_error}")
        
        return {
            "success": True,
            "offset": offset,
            "prediction_error": prediction_error,
            "total_records": self.total_records
        }
    
    def find_pattern_matches(self, request: PredictionRequest) -> Tuple[Optional[int], float, int]:
        """
        Find matching patterns and predict single best number
        Returns: (predicted_number, confidence, match_count)
        """
        cell_key = GridSystem.get_cell_key(
            request.pos_x,
            request.pos_y,
            request.speed_ms_total,
            request.direction
        )
        
        # Collect all matches with weights
        offset_weights = defaultdict(float)
        total_matches = 0
        
        # Search in limited radius for more precise matching
        for radius in range(Config.MAX_SEARCH_RADIUS + 1):
            cells_to_check = self._get_neighbor_cells(cell_key, radius)
            
            for check_cell in cells_to_check:
                if check_cell in self.pattern_database:
                    for pattern in self.pattern_database[check_cell]:
                        # Filter old data even from memory
                        pattern_time = datetime.fromisoformat(pattern['timestamp'])
                        if datetime.now() - pattern_time > timedelta(days=Config.DATA_RETENTION_DAYS):
                            continue
                        
                        # Filter by traveled pockets similarity
                        if abs(pattern.get('traveled_pockets', 7) - request.traveled_pockets) > 2:
                            continue
                        
                        offset = pattern['offset']
                        
                        # FIXED: Ensure confidence is never None
                        confidence_value = pattern.get('confidence', 1.0)
                        if confidence_value is None:
                            confidence_value = 1.0
                            
                        # Weight by distance and pattern confidence
                        weight = (1.0 / (1 + radius)) * confidence_value
                        offset_weights[offset] += weight
                        total_matches += 1
        
        if total_matches < Config.MIN_MATCHES_FOR_PREDICTION:
            return None, 0.0, total_matches
        
        # Find the most probable offset
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
        
        # Calculate confidence based on consistency
        confidence = self._calculate_confidence(offset_weights, total_matches)
        
        # Store prediction for accuracy tracking
        if predicted_number is not None:
            self.predictions_made[request.round_id] = {
                'main_offset': best_offset,
                'predicted_number': predicted_number
            }
        
        return predicted_number, confidence, total_matches
    
    def _get_neighbor_cells(self, center_cell: Tuple, radius: int) -> List[Tuple]:
        """Get neighboring grid cells at specified radius"""
        if radius == 0:
            return [center_cell]
        
        x, y, speed, direction = center_cell
        neighbors = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for ds in range(-radius, radius + 1):
                    # Check if on radius boundary
                    if max(abs(dx), abs(dy), abs(ds)) == radius:
                        neighbor = (
                            x + dx * Config.POSITION_CELL_SIZE,
                            y + dy * Config.POSITION_CELL_SIZE,
                            speed + ds * Config.SPEED_CELL_SIZE,
                            direction
                        )
                        neighbors.append(neighbor)
        
        return neighbors
    
    def _calculate_confidence(self, offset_weights: Dict[int, float], total_matches: int) -> float:
        """Calculate confidence score based on pattern consistency"""
        if total_matches < Config.MIN_MATCHES_FOR_PREDICTION:
            return 0.0
        
        if not offset_weights:
            return 0.0
        
        # Get the highest weight
        top_weight = max(offset_weights.values())
        total_weight = sum(offset_weights.values())
        
        # FIXED: Ensure no division by zero
        if total_weight == 0:
            return 0.0
            
        # Calculate consistency - what percentage of matches point to the same result
        consistency = top_weight / total_weight
        
        # Only return high confidence if 60%+ matches agree on the same offset
        if consistency >= 0.6:
            return consistency
        else:
            return 0.3  # Low confidence even if we have matches
    
    def _save_to_csv(self, record: Dict[str, Any]):
        """Append record to CSV database"""
        try:
            with open(self.data_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=Config.CSV_COLUMNS)
                writer.writerow(record)
        except Exception as e:
            logger.error(f"Failed to save record: {e}")
    
    def cleanup_old_csv_data(self):
        """Remove old records from CSV file while preserving recent data"""
        try:
            # Only cleanup if we have enough data
            if self.total_records < 50000:
                return
            
            cutoff_date = datetime.now() - timedelta(days=60)  # Keep 60 days in file
            temp_file = self.data_path + '.tmp'
            kept_records = 0
            removed_records = 0
            
            # Read and filter data
            with open(self.data_path, 'r', encoding='utf-8') as old_file:
                with open(temp_file, 'w', newline='', encoding='utf-8') as new_file:
                    reader = csv.DictReader(old_file)
                    writer = csv.DictWriter(new_file, fieldnames=Config.CSV_COLUMNS)
                    writer.writeheader()
                    
                    for row in reader:
                        try:
                            row_time = datetime.fromisoformat(row['timestamp'])
                            if row_time >= cutoff_date:
                                writer.writerow(row)
                                kept_records += 1
                            else:
                                removed_records += 1
                        except:
                            # Keep records with invalid timestamps
                            writer.writerow(row)
                            kept_records += 1
            
            # Replace old file with cleaned one
            os.replace(temp_file, self.data_path)
            logger.info(f"CSV cleanup complete: kept {kept_records}, removed {removed_records} old records")
            
        except Exception as e:
            logger.error(f"Error during CSV cleanup: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
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
    title="Professional Roulette Prediction Server",
    description="High-accuracy pattern matching for Evolution Gaming roulette",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Initialize storage
storage = DataStorage()

# Run cleanup on startup (and optionally schedule periodic cleanup)
@app.on_event("startup")
async def startup_event():
    """Run maintenance tasks on server startup"""
    storage.cleanup_old_csv_data()
    storage.cleanup_pending_rounds()

@app.get("/")
async def status():
    """Server status and statistics"""
    return {
        "server": "Professional Roulette Prediction Server",
        "version": "2.0.0",
        "status": "operational",
        "statistics": {
            "total_records": storage.total_records,
            "pattern_cells": len(storage.pattern_database),
            "pending_rounds": len(storage.pending_rounds),
            "active_predictions": len(storage.predictions_made)
        },
        "configuration": {
            "min_speed_ms": Config.MIN_BALL_SPEED_MS,
            "max_speed_ms": Config.MAX_BALL_SPEED_MS,
            "cell_size_position": Config.POSITION_CELL_SIZE,
            "cell_size_speed": Config.SPEED_CELL_SIZE,
            "min_matches_required": Config.MIN_MATCHES_FOR_PREDICTION
        }
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Process prediction request and return single best number"""
    try:
        # Store pending round
        if not storage.store_pending(request):
            return {
                "predicted_number": None,
                "confidence": 0,
                "error": "Invalid data - round rejected",
                "dataset_rows": storage.total_records
            }
        
        # Find pattern matches
        predicted_number, confidence, matches = storage.find_pattern_matches(request)
        
        if predicted_number is not None and confidence > 0.3:
            return {
                "predicted_number": predicted_number,
                "confidence": round(confidence, 3),
                "matches_found": matches,
                "dataset_rows": storage.total_records,
                "accuracy_metrics": {
                    "confidence_level": "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "low",
                    "pattern_strength": f"{int(confidence * 100)}%"
                }
            }
        else:
            return {
                "predicted_number": None,
                "confidence": 0,
                "error": f"Insufficient pattern matches. Found {matches}, need {Config.MIN_MATCHES_FOR_PREDICTION}+",
                "dataset_rows": storage.total_records,
                "matches_found": matches
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
            
            if result["prediction_error"] is not None:
                response["prediction_error"] = result["prediction_error"]
                response["accuracy"] = "accurate" if result["prediction_error"] <= 3 else "needs_improvement"
            
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
    cell_stats = {}
    for cell_key, patterns in storage.pattern_database.items():
        x, y, speed, direction = cell_key
        cell_stats[f"{direction}_{speed}ms_({x},{y})"] = len(patterns)
    
    return {
        "total_records": storage.total_records,
        "pattern_cells": len(storage.pattern_database),
        "pending_rounds": len(storage.pending_rounds),
        "predictions_tracking": len(storage.predictions_made),
        "cell_distribution": cell_stats,
        "database_path": storage.data_path,
        "data_retention_days": Config.DATA_RETENTION_DAYS
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
    print("PROFESSIONAL ROULETTE PREDICTION SERVER v2.0.0")
    print("High-accuracy pattern matching for Evolution Gaming")
    print("="*70)
    print(f"Database: {storage.data_path}")
    print(f"Records loaded: {storage.total_records}")
    print(f"Pattern cells: {len(storage.pattern_database)}")
    print(f"Data retention: {Config.DATA_RETENTION_DAYS} days")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
