"""
Roulette Final Memory Server
Direct final number prediction based on historical matches
Version: 2.0.0
"""

import os
import csv
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import Counter, defaultdict

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
DATABASE_FILE = "roulette_patterns.csv"

# CSV columns with new fields
CSV_COLUMNS = [
    'timestamp', 'round_id', 'table_id', 'ball_speed_ms', 'traveled_pockets',
    'number_at_ts2', 'direction', 'winning_number', 'pockets_to_win',
    'green_angle_ts1', 'green_angle_ts2', 'wheel_speed'
]

# Tolerance settings (Updated for matching)
MS_TOLERANCE = 10  # ±10 milliseconds tolerance
POCKETS_TOLERANCE = 0  # Exact pockets match
ANGLE_TOLERANCE = 0.3  # ±0.3 degrees tolerance

# Maximum valid speed filter
MAX_VALID_SPEED_MS = 1500  # Maximum valid ball speed
TIME_WINDOW_SECONDS = 30  # Time window for grouping same spin

# ============================================================================
# DATA MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request for prediction based on ball measurements"""
    round_id: str = Field(..., min_length=5, max_length=50)
    ball_speed_ms: int = Field(..., ge=200, le=10000, description="Ball rotation time in ms")
    traveled_pockets: int = Field(..., ge=0, le=37, description="Pockets traveled between timestamps")
    number_at_ts2: int = Field(..., ge=0, le=36, description="Number at second timestamp")
    direction: str = Field(..., pattern="^(CW|CCW)$", description="Ball direction")
    table_id: str = Field(default="auto_roulette")
    green_angle_ts1: Optional[float] = Field(None, description="Green marker angle at TS1")
    green_angle_ts2: Optional[float] = Field(None, description="Green marker angle at TS2")

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

def circular_angle_difference(angle1: float, angle2: float) -> float:
    """Calculate minimum angular difference in degrees (0-180)"""
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class PatternDatabase:
    """Manages pattern storage and matching with final memory approach"""
    
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
    
    def find_final_matches(self, ball_speed: int, traveled_pockets: int, 
                          direction: str, green_angle_ts2: Optional[float], 
                          table_id: str) -> List[int]:
        """
        Find ALL matches and return list of winning numbers
        Uses tolerance matching for speed and angle
        """
        if not os.path.exists(DATABASE_FILE):
            return []
        
        winning_numbers = []
        
        try:
            with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Skip if different table
                    if row.get('table_id') != table_id:
                        continue
                    
                    # Check match with tolerances
                    if (abs(int(row['ball_speed_ms']) - ball_speed) > MS_TOLERANCE or
                        abs(int(row['traveled_pockets']) - traveled_pockets) > POCKETS_TOLERANCE or
                        row['direction'] != direction):
                        continue
                    
                    # Check angle match if provided
                    if green_angle_ts2 is not None and row.get('green_angle_ts2'):
                        try:
                            row_angle = float(row['green_angle_ts2'])
                            angle_diff = circular_angle_difference(green_angle_ts2, row_angle)
                            if angle_diff > ANGLE_TOLERANCE:
                                continue
                        except (ValueError, TypeError):
                            continue
                    
                    # This is a match - add winning number
                    if row.get('winning_number'):
                        winning_numbers.append(int(row['winning_number']))
                
                if winning_numbers:
                    logger.info(f"Found {len(winning_numbers)} matches for "
                              f"speed={ball_speed}ms, pockets={traveled_pockets}, "
                              f"angle={green_angle_ts2:.1f}°" if green_angle_ts2 else "")
                
                return winning_numbers
                
        except Exception as e:
            logger.error(f"Error searching database: {e}")
            return []
    
    def store_pending(self, request: PredictionRequest):
        """Store round data temporarily until winning number arrives"""
        
        # Filter out invalid speeds
        if request.ball_speed_ms > MAX_VALID_SPEED_MS:
            logger.warning(f"Rejected round {request.round_id}: speed {request.ball_speed_ms}ms > {MAX_VALID_SPEED_MS}ms")
            return
        
        # Calculate wheel speed if both angles are provided
        wheel_speed = None
        if request.green_angle_ts1 is not None and request.green_angle_ts2 is not None:
            angle_diff = request.green_angle_ts2 - request.green_angle_ts1
            if angle_diff < 0:
                angle_diff += 360
            wheel_speed = angle_diff  # degrees traveled during ball_speed_ms time
            logger.info(f"Calculated wheel speed: {wheel_speed:.1f} degrees in {request.ball_speed_ms}ms")
        
        # Store with precise timestamp for sorting
        self.pending_rounds[request.round_id] = {
            'timestamp': datetime.now().isoformat(),
            'received_at': datetime.now(),  # For precise sorting
            'table_id': request.table_id,
            'ball_speed_ms': request.ball_speed_ms,
            'traveled_pockets': request.traveled_pockets,
            'number_at_ts2': request.number_at_ts2,
            'direction': request.direction,
            'green_angle_ts1': request.green_angle_ts1,
            'green_angle_ts2': request.green_angle_ts2,
            'wheel_speed': wheel_speed
        }
        logger.info(f"Stored pending round: {request.round_id}")
    
    def complete_round(self, round_id: str, winning_number: int) -> Dict[str, Any]:
        """Complete round with winning number and save to database"""
        
        # Find the reference round
        if round_id not in self.pending_rounds:
            # Try to find any round from the same spin
            current_time = datetime.now()
            time_window_start = current_time - timedelta(seconds=TIME_WINDOW_SECONDS)
            
            # Find all rounds in time window
            matching_rounds = []
            for rid, data in self.pending_rounds.items():
                if data['received_at'] >= time_window_start:
                    matching_rounds.append((rid, data))
            
            if not matching_rounds:
                return {
                    "success": False,
                    "error": "No pending rounds found in time window"
                }
            
            # Group by table_id
            table_rounds = defaultdict(list)
            for rid, data in matching_rounds:
                table_rounds[data['table_id']].append((rid, data))
            
            # Find the most likely table (with most entries)
            max_table = max(table_rounds.items(), key=lambda x: len(x[1]))
            table_id, rounds = max_table
            
            # Sort by received time and take the first
            rounds.sort(key=lambda x: x[1]['received_at'])
            selected_round_id, selected_data = rounds[0]
            
            logger.info(f"Selected first round {selected_round_id} from {len(rounds)} rounds on table {table_id}")
        else:
            # Direct match found
            selected_round_id = round_id
            selected_data = self.pending_rounds[round_id]
            table_id = selected_data['table_id']
            
            # Still need to find related rounds to delete them
            current_time = datetime.now()
            time_window_start = current_time - timedelta(seconds=TIME_WINDOW_SECONDS)
            
            rounds = [(rid, data) for rid, data in self.pending_rounds.items()
                     if data['table_id'] == table_id and data['received_at'] >= time_window_start]
        
        # Calculate pockets from TS2 to winning number
        pockets_to_win = calculate_pocket_distance(
            selected_data['number_at_ts2'],
            winning_number,
            selected_data['direction']
        )
        
        # Prepare complete record
        record = {
            'timestamp': selected_data['timestamp'],
            'round_id': selected_round_id,
            'table_id': selected_data['table_id'],
            'ball_speed_ms': selected_data['ball_speed_ms'],
            'traveled_pockets': selected_data['traveled_pockets'],
            'number_at_ts2': selected_data['number_at_ts2'],
            'direction': selected_data['direction'],
            'winning_number': winning_number,
            'pockets_to_win': pockets_to_win,
            'green_angle_ts1': selected_data['green_angle_ts1'],
            'green_angle_ts2': selected_data['green_angle_ts2'],
            'wheel_speed': selected_data['wheel_speed']
        }
        
        # Save to CSV
        try:
            with open(DATABASE_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(record)
            
            self.total_records += 1
            
            # Delete ALL rounds from the same spin
            deleted_count = 0
            for rid, data in list(self.pending_rounds.items()):
                if (data['table_id'] == table_id and 
                    data['received_at'] >= time_window_start):
                    del self.pending_rounds[rid]
                    deleted_count += 1
            
            logger.info(f"Completed round {selected_round_id}: winning={winning_number}, "
                       f"deleted {deleted_count} pending rounds, total_records={self.total_records}")
            
            return {
                "success": True,
                "pockets_to_win": pockets_to_win,
                "total_records": self.total_records,
                "deleted_pending": deleted_count
            }
            
        except Exception as e:
            logger.error(f"Error saving record: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_prediction(self, request: PredictionRequest) -> Optional[Dict[str, Any]]:
        """Get prediction using final memory approach"""
        
        # First, store the pending round
        self.store_pending(request)
        
        # Skip prediction if speed is invalid
        if request.ball_speed_ms > MAX_VALID_SPEED_MS:
            return None
        
        # Find all matching winning numbers
        winning_numbers = self.find_final_matches(
            request.ball_speed_ms,
            request.traveled_pockets,
            request.direction,
            request.green_angle_ts2,
            request.table_id
        )
        
        if not winning_numbers:
            return None
        
        # Count occurrences of each number
        number_counts = Counter(winning_numbers)
        
        # Get most common number and its count
        most_common_number, count = number_counts.most_common(1)[0]
        
        # Calculate confidence as percentage of total matches
        confidence = count / len(winning_numbers)
        
        # Get top 3 predictions with their percentages
        top3 = []
        for number, cnt in number_counts.most_common(3):
            percentage = cnt / len(winning_numbers)
            top3.append([number, round(percentage, 2)])
        
        logger.info(f"Prediction for round {request.round_id}: "
                   f"number={most_common_number} (confidence={confidence:.2f}, "
                   f"based on {len(winning_numbers)} matches)")
        
        return {
            "predicted_number": most_common_number,
            "confidence": round(confidence, 2),
            "matches_found": len(winning_numbers),
            "top3": top3
        }

# ============================================================================
# API SERVER
# ============================================================================

app = FastAPI(
    title="Roulette Final Memory Server",
    description="Direct final number prediction based on historical matches with filtering",
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
        "server": "Roulette Final Memory Server",
        "version": "2.0.0",
        "status": "operational",
        "method": "final_memory_with_filtering",
        "statistics": {
            "total_records": db.total_records,
            "pending_rounds": len(db.pending_rounds)
        },
        "tolerances": {
            "speed_ms": MS_TOLERANCE,
            "pockets": POCKETS_TOLERANCE,
            "angle_degrees": ANGLE_TOLERANCE,
            "max_valid_speed_ms": MAX_VALID_SPEED_MS
        }
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Get prediction based on final memory with tolerance matching"""
    
    try:
        # Get prediction using final memory approach
        result = db.get_prediction(request)
        
        if result:
            # Found matches - return prediction
            return {
                "predicted_number": result["predicted_number"],
                "confidence": result["confidence"],
                "matches_found": result["matches_found"],
                "dataset_rows": db.total_records,
                "top3": result["top3"],
                "method": "final_memory"
            }
        else:
            # No matches found or invalid data
            message = f"No matches for speed={request.ball_speed_ms}ms"
            if request.ball_speed_ms > MAX_VALID_SPEED_MS:
                message = f"Invalid speed: {request.ball_speed_ms}ms > {MAX_VALID_SPEED_MS}ms"
            
            return {
                "predicted_number": None,
                "confidence": 0.0,
                "matches_found": 0,
                "dataset_rows": db.total_records,
                "message": message
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
                "pockets_recorded": result["pockets_to_win"],
                "deleted_pending": result.get("deleted_pending", 1)
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
    """Get detailed statistics about collected data"""
    
    # Count patterns by parameter combinations
    pattern_stats = defaultdict(lambda: {"count": 0, "winning_distribution": defaultdict(int)})
    
    try:
        with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Create pattern key
                key = f"{row['ball_speed_ms']}ms_{row['traveled_pockets']}p_{row['direction']}"
                pattern_stats[key]["count"] += 1
                
                # Track winning number distribution
                if row.get('winning_number'):
                    pattern_stats[key]["winning_distribution"][row['winning_number']] += 1
    except:
        pass
    
    # Convert to list and calculate entropy for each pattern
    pattern_list = []
    for pattern, data in pattern_stats.items():
        total = data["count"]
        distribution = data["winning_distribution"]
        
        # Find most common winning number
        if distribution:
            most_common = max(distribution.items(), key=lambda x: x[1])
            confidence = most_common[1] / total
        else:
            most_common = (None, 0)
            confidence = 0
        
        pattern_list.append({
            "pattern": pattern,
            "occurrences": total,
            "most_common_winner": most_common[0],
            "confidence": round(confidence, 2),
            "unique_outcomes": len(distribution)
        })
    
    # Sort by occurrences
    pattern_list.sort(key=lambda x: x["occurrences"], reverse=True)
    
    return {
        "total_records": db.total_records,
        "pending_rounds": len(db.pending_rounds),
        "unique_patterns": len(pattern_stats),
        "top_patterns": pattern_list[:20],  # Top 20 patterns
        "time_window_seconds": TIME_WINDOW_SECONDS,
        "max_valid_speed_ms": MAX_VALID_SPEED_MS
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
    print("ROULETTE FINAL MEMORY SERVER v2.0.0")
    print("Direct Final Number Prediction Method with Filtering")
    print("="*60)
    print(f"Database: {DATABASE_FILE}")
    print(f"Records: {db.total_records}")
    print("="*60)
    print("\nTolerance settings:")
    print(f"- Speed: ±{MS_TOLERANCE}ms tolerance")
    print(f"- Pockets: ±{POCKETS_TOLERANCE} (exact match)")
    print(f"- Angle: ±{ANGLE_TOLERANCE}° tolerance")
    print(f"- Max valid speed: {MAX_VALID_SPEED_MS}ms")
    print(f"- Time window: {TIME_WINDOW_SECONDS} seconds")
    print("="*60)
    print("\nHow it works:")
    print("1. Filters data by speed (<1500ms)")
    print("2. Groups rounds by table_id and time window")
    print("3. Stores only first round from each group")
    print("4. Uses tolerances for prediction matching")
    print("5. Deletes all pending rounds after recording winner")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
