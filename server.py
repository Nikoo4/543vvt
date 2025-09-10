"""
Simple Roulette Prediction Server - Enhanced with TS3
Based on exact pattern matching method for automated roulette
Version: 2.0.0
"""

import os
import csv
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("RouletteServer")

# ============================================================================
# CONFIGURATION
# ============================================================================

# European wheel layout (clockwise)
EUROPEAN_WHEEL = [
    0, 26, 3, 35, 12, 28, 7, 29, 18, 22, 9, 31, 14, 20, 1, 33, 
    16, 24, 5, 10, 23, 8, 30, 11, 36, 13, 27, 6, 34, 17, 25, 2, 
    21, 4, 19, 15, 32
]

# Database configuration
DATABASE_FILE = "roulette_patterns_v2.csv"
MAX_RECORDS = 100000

# CSV columns - Enhanced with TS3 data and rotor direction
CSV_COLUMNS = [
    'timestamp', 'round_id', 
    'ball_speed_ms', 'period1_ms', 'period2_ms', 'deceleration',
    'traveled_pockets', 'number_at_ts2', 'number_at_ts3', 
    'direction', 'winning_number', 'pockets_to_win',
    'green_angle_ts1', 'green_angle_ts2', 'green_angle_ts3', 
    'wheel_speed', 'rotor_direction'  # CHANGE 1: Added rotor_direction
]

# Matching tolerances
wheel_speed_tolerance = 5  # degrees tolerance for wheel speed matching

# ============================================================================
# DATA MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request for prediction based on ball measurements"""
    round_id: str = Field(..., min_length=5, max_length=50)
    ball_speed_ms: int = Field(..., ge=200, le=10000, description="Ball rotation time in ms")
    period1_ms: Optional[int] = Field(None, description="Time between TS1 and TS2")
    period2_ms: Optional[int] = Field(None, description="Time between TS2 and TS3")
    deceleration: Optional[int] = Field(None, description="Ball deceleration (period2 - period1)")
    traveled_pockets: int = Field(..., ge=0, le=37, description="Pockets traveled between timestamps")
    number_at_ts2: int = Field(..., ge=0, le=36, description="Number at second timestamp")
    number_at_ts3: Optional[int] = Field(None, ge=0, le=36, description="Number at third timestamp")
    direction: str = Field(..., pattern="^(CW|CCW)$", description="Ball direction")
    table_id: str = Field(default="auto_roulette")
    green_angle_ts1: Optional[float] = Field(None, description="Green marker angle at TS1")
    green_angle_ts2: Optional[float] = Field(None, description="Green marker angle at TS2")
    green_angle_ts3: Optional[float] = Field(None, description="Green marker angle at TS3")
    rotor_direction: Optional[str] = Field(None, pattern="^(CW|CCW)$", description="Rotor direction")  # CHANGE 2: Added field

class WinnerRequest(BaseModel):
    """Log winning number for completed round"""
    round_id: str = Field(..., min_length=5, max_length=50)
    winning_number: int = Field(..., ge=0, le=36)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_pocket_distance(from_number: int, to_number: int, direction: str) -> int:
    """Calculate distance between two numbers on wheel"""
    try:
        from_idx = EUROPEAN_WHEEL.index(from_number)
        to_idx = EUROPEAN_WHEEL.index(to_number)
    except ValueError:
        return 0
    
    if direction == "CW":
        distance = (to_idx - from_idx) % 37
    else:  # CCW
        distance = (from_idx - to_idx) % 37
    
    return distance

def get_number_at_distance(from_number: int, distance: int, direction: str) -> int:
    """Get number at specified distance from reference number"""
    try:
        from_idx = EUROPEAN_WHEEL.index(from_number)
    except ValueError:
        return from_number
    
    if direction == "CW":
        target_idx = (from_idx + distance) % 37
    else:  # CCW
        target_idx = (from_idx - distance) % 37
    
    return EUROPEAN_WHEEL[target_idx]

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class PatternDatabase:
    """Manages pattern storage and matching"""
    
    def __init__(self):
        self.pending_rounds = {}  # Temporary storage for rounds awaiting results
        self.total_records = 0
        self._initialize_database()
        self._load_record_count()
    
    def _initialize_database(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(DATABASE_FILE):
            with open(DATABASE_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
            logger.info(f"Created new database: {DATABASE_FILE}")
    
    def _load_record_count(self):
        """Count existing records in database"""
        try:
            with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
                self.total_records = sum(1 for _ in f) - 1  # Subtract header
                if self.total_records < 0:
                    self.total_records = 0
            logger.info(f"Loaded database with {self.total_records} records")
        except Exception as e:
            logger.error(f"Error counting records: {e}")
            self.total_records = 0
    
    def find_exact_match(self, request: PredictionRequest, wheel_speed: Optional[float]) -> Optional[int]:
        """
        Find EXACT match in database with TS3 data if available
        Returns pockets_to_win if found, None otherwise
        """
        if not os.path.exists(DATABASE_FILE):
            return None
        
        # Determine which reference number to use based on TS3 availability
        reference_number = request.number_at_ts3 if request.number_at_ts3 is not None else request.number_at_ts2
        
        try:
            with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Check for EXACT match on primary parameters
                    if (int(row['ball_speed_ms']) == request.ball_speed_ms and
                        int(row['traveled_pockets']) == request.traveled_pockets and
                        row['direction'] == request.direction and
                        row.get('rotor_direction') == request.rotor_direction and  # CHANGE 3: Check rotor direction
                        row['pockets_to_win']):  # Must have result
                        
                        # CHANGE 4: If wheel speed data is available, use percentage-based matching
                        if wheel_speed is not None and row.get('wheel_speed'):
                            try:
                                row_wheel_speed = float(row['wheel_speed'])
                                relative_diff = abs(row_wheel_speed - wheel_speed) / wheel_speed
                                if relative_diff > 0.03:  # 3% tolerance
                                    continue
                            except (ValueError, TypeError, ZeroDivisionError):
                                continue
                        
                        # If we have TS3 data, also match on deceleration
                        if request.deceleration is not None and row.get('deceleration'):
                            try:
                                row_decel = int(row['deceleration'])
                                if abs(row_decel - request.deceleration) > 50:  # 50ms tolerance
                                    continue
                            except (ValueError, TypeError):
                                pass
                        
                        logger.info(f"Found exact match: speed={request.ball_speed_ms}, "
                                  f"traveled={request.traveled_pockets}, offset={row['pockets_to_win']}")
                        if wheel_speed is not None:
                            logger.info(f"Wheel speed matched within tolerance")
                        if request.deceleration is not None:
                            logger.info(f"Deceleration matched: {request.deceleration}ms")
                        return int(row['pockets_to_win'])
                
                logger.info(f"No exact match found for: speed={request.ball_speed_ms}, "
                          f"traveled={request.traveled_pockets}, direction={request.direction}, "
                          f"wheel_speed={wheel_speed}, deceleration={request.deceleration}")
                return None
                
        except Exception as e:
            logger.error(f"Error searching database: {e}")
            return None
    
    def store_pending(self, request: PredictionRequest):
        """Store round data temporarily until winning number arrives"""
        
        # CHANGE 5: Calculate wheel speed in degrees per second
        wheel_speed = None
        if request.green_angle_ts1 is not None and request.green_angle_ts2 is not None:
            angle_diff = request.green_angle_ts2 - request.green_angle_ts1
            if angle_diff < 0:
                angle_diff += 360
            wheel_speed = angle_diff / (request.ball_speed_ms / 1000.0)  # degrees per second
            logger.info(f"Calculated wheel speed: {wheel_speed:.1f} degrees/second")
        
        self.pending_rounds[request.round_id] = {
            'timestamp': datetime.now().isoformat(),
            'ball_speed_ms': request.ball_speed_ms,
            'period1_ms': request.period1_ms,
            'period2_ms': request.period2_ms,
            'deceleration': request.deceleration,
            'traveled_pockets': request.traveled_pockets,
            'number_at_ts2': request.number_at_ts2,
            'number_at_ts3': request.number_at_ts3,
            'direction': request.direction,
            'green_angle_ts1': request.green_angle_ts1,
            'green_angle_ts2': request.green_angle_ts2,
            'green_angle_ts3': request.green_angle_ts3,
            'wheel_speed': wheel_speed,
            'rotor_direction': request.rotor_direction  # CHANGE 6: Store rotor direction
        }
        logger.info(f"Stored pending round: {request.round_id}, wheel_speed: {wheel_speed}")
        if request.deceleration:
            logger.info(f"With deceleration: {request.deceleration}ms")
    
    def complete_round(self, round_id: str, winning_number: int) -> Dict[str, Any]:
        """Complete round with winning number and save to database"""
        
        if round_id not in self.pending_rounds:
            return {
                "success": False,
                "error": "Round not found in pending storage"
            }
        
        pending = self.pending_rounds[round_id]
        
        # Calculate pockets from reference number to winning number
        # Use TS3 if available, otherwise TS2
        reference_number = pending['number_at_ts3'] if pending['number_at_ts3'] is not None else pending['number_at_ts2']
        pockets_to_win = calculate_pocket_distance(
            reference_number,
            winning_number,
            pending['direction']
        )
        
        # Prepare complete record
        record = {
            'timestamp': pending['timestamp'],
            'round_id': round_id,
            'ball_speed_ms': pending['ball_speed_ms'],
            'period1_ms': pending['period1_ms'],
            'period2_ms': pending['period2_ms'],
            'deceleration': pending['deceleration'],
            'traveled_pockets': pending['traveled_pockets'],
            'number_at_ts2': pending['number_at_ts2'],
            'number_at_ts3': pending['number_at_ts3'],
            'direction': pending['direction'],
            'winning_number': winning_number,
            'pockets_to_win': pockets_to_win,
            'green_angle_ts1': pending['green_angle_ts1'],
            'green_angle_ts2': pending['green_angle_ts2'],
            'green_angle_ts3': pending['green_angle_ts3'],
            'wheel_speed': pending['wheel_speed'],
            'rotor_direction': pending['rotor_direction']  # CHANGE 7: Save rotor direction
        }
        
        # Save to CSV
        try:
            with open(DATABASE_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(record)
            
            self.total_records += 1
            del self.pending_rounds[round_id]
            
            logger.info(f"Completed round {round_id}: winning={winning_number}, "
                       f"offset={pockets_to_win}, total_records={self.total_records}")
            if pending['number_at_ts3'] is not None:
                logger.info(f"Using TS3 reference: {pending['number_at_ts3']}")
            
            return {
                "success": True,
                "pockets_to_win": pockets_to_win,
                "total_records": self.total_records
            }
            
        except Exception as e:
            logger.error(f"Error saving record: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_prediction(self, request: PredictionRequest) -> Optional[int]:
        """Get prediction based on exact pattern match"""
        
        # First, store the pending round
        self.store_pending(request)
        
        # Calculate wheel speed for matching
        wheel_speed = None
        if request.green_angle_ts1 is not None and request.green_angle_ts2 is not None:
            angle_diff = request.green_angle_ts2 - request.green_angle_ts1
            if angle_diff < 0:
                angle_diff += 360
            wheel_speed = angle_diff / (request.ball_speed_ms / 1000.0)  # degrees per second
        
        # Look for exact match in database
        offset = self.find_exact_match(request, wheel_speed)
        
        if offset is not None:
            # Calculate predicted number from reference
            reference_number = request.number_at_ts3 if request.number_at_ts3 is not None else request.number_at_ts2
            predicted = get_number_at_distance(
                reference_number,
                offset,
                request.direction
            )
            
            logger.info(f"Prediction for round {request.round_id}: "
                       f"number={predicted} (offset={offset} from {reference_number})")
            return predicted
        
        return None

# ============================================================================
# API SERVER
# ============================================================================

app = FastAPI(
    title="Simple Roulette Pattern Server v2",
    description="Exact pattern matching prediction system with TS3 support",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Initialize database
db = PatternDatabase()

@app.get("/")
async def status():
    """Server status and statistics"""
    return {
        "server": "Simple Roulette Pattern Server",
        "version": "2.0.0",
        "status": "operational",
        "method": "exact_pattern_matching_with_ts3",
        "features": ["3-point measurement", "deceleration tracking", "wheel speed matching", "rotor direction"],
        "statistics": {
            "total_records": db.total_records,
            "pending_rounds": len(db.pending_rounds)
        }
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Get prediction based on exact pattern match"""
    
    try:
        # Get prediction
        predicted_number = db.get_prediction(request)
        
        if predicted_number is not None:
            # Found exact match - return prediction
            return {
                "predicted_number": predicted_number,
                "confidence": 1.0,  # Exact match = 100% confidence in pattern
                "dataset_rows": db.total_records,
                "method": "exact_match_ts3" if request.number_at_ts3 else "exact_match_ts2",
                "matches_found": 1,
                "accuracy_metrics": {
                    "confidence_level": "high" if request.deceleration else "medium"
                }
            }
        else:
            # No exact match found
            return {
                "predicted_number": None,
                "confidence": 0.0,
                "dataset_rows": db.total_records,
                "matches_found": 0,
                "message": f"No exact match for speed={request.ball_speed_ms}ms, "
                          f"traveled={request.traveled_pockets}, "
                          f"deceleration={request.deceleration}ms"
            }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_winner")
async def log_winner(request: WinnerRequest):
    """Log winning number for completed round"""
    
    try:
        result = db.complete_round(request.round_id, request.winning_number)
        
        if result["success"]:
            return {
                "stored": True,
                "dataset_rows": result["total_records"],
                "pockets_recorded": result["pockets_to_win"]
            }
        else:
            return {
                "stored": False,
                "error": result["error"],
                "dataset_rows": db.total_records
            }
    
    except Exception as e:
        logger.error(f"Winner logging error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def statistics():
    """Get detailed statistics"""
    
    # Count patterns by speed/pockets combination
    pattern_counts = {}
    ts3_count = 0
    
    try:
        with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                key = f"{row['ball_speed_ms']}ms_{row['traveled_pockets']}p_{row['direction']}"
                pattern_counts[key] = pattern_counts.get(key, 0) + 1
                
                # Count TS3 records
                if row.get('number_at_ts3'):
                    ts3_count += 1
    except:
        pass
    
    return {
        "total_records": db.total_records,
        "ts3_records": ts3_count,
        "ts2_only_records": db.total_records - ts3_count,
        "pending_rounds": len(db.pending_rounds),
        "unique_patterns": len(pattern_counts),
        "pattern_distribution": dict(sorted(
            pattern_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20])  # Top 20 patterns
    }

@app.delete("/clear_pending")
async def clear_pending():
    """Clear all pending rounds (admin function)"""
    count = len(db.pending_rounds)
    db.pending_rounds.clear()
    return {"cleared": count}

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("SIMPLE ROULETTE PATTERN SERVER v2.0.0")
    print("Exact Pattern Matching Method with TS3 Support")
    print("="*60)
    print(f"Database: {DATABASE_FILE}")
    print(f"Records: {db.total_records}")
    print("="*60)
    print("\nServer starting on http://0.0.0.0:8000")
    print("\nWhat's new in v2.0.0:")
    print("- Collects 3 timestamps for better accuracy")
    print("- Tracks ball deceleration between periods")
    print("- Improved wheel speed matching (3% tolerance)")
    print("- Rotor direction tracking")
    print("- Reference from TS3 when available (more accurate)")
    print("\nHow it works:")
    print("1. Collects ball_speed_ms, traveled_pockets, and deceleration")
    print("2. Searches for EXACT match in database")
    print("3. If found, predicts same offset to winning number")
    print("4. No match = no prediction (need more data)")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
