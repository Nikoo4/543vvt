"""
Simple Roulette Prediction Server - Direct Implementation from Document
This server implements the exact logic described in the prediction document
"""

import os
import csv
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("SimpleRouletteServer")

# ============================================================================
# CONFIGURATION
# ============================================================================

# European roulette wheel layout (37 pockets: 0-36)
WHEEL_LAYOUT = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27,
    13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1,
    20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
]

# CSV columns for data storage
CSV_COLUMNS = [
    'timestamp', 'round_id', 'table_id',
    'pos_x', 'pos_y', 'speed_ms', 'traveled_pockets',
    'direction', 'number_t1', 'number_t2', 
    'winning_number', 'pockets_from_winning'
]

# ============================================================================
# DATA MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request for prediction - exactly as described in document"""
    round_id: str = Field(..., description="Unique round identifier")
    pos_x: float = Field(..., description="X position where ball was measured")
    pos_y: float = Field(..., description="Y position where ball was measured")
    speed_ms_total: int = Field(..., description="Time in ms for ball to return to same position")
    traveled_pockets: int = Field(..., description="Pockets traveled between TS1 and TS2")
    direction: str = Field(..., description="Ball direction: CW or CCW")
    number_at_t1: int = Field(..., description="Number under ball at TS1")
    number_at_t2: int = Field(..., description="Number under ball at TS2")
    table_id: str = Field(default="default", description="Table identifier")

class WinnerRequest(BaseModel):
    """Winning number notification"""
    round_id: str
    winning_number: int = Field(..., ge=0, le=36)

# ============================================================================
# SIMPLE DATA STORAGE
# ============================================================================

class SimpleDataStorage:
    """Simple storage implementing exact logic from document"""
    
    def __init__(self):
        self.data_path = self._get_data_path()
        self.pending_rounds = {}  # Temporary storage for incomplete rounds
        self.historical_data = []  # All completed rounds for matching
        
        self._initialize_storage()
        self._load_historical_data()
    
    def _get_data_path(self) -> str:
        """Get path for CSV storage"""
        # Try to find writable location
        for path in ["./roulette_data.csv", os.path.expanduser("~/roulette_data.csv")]:
            try:
                directory = os.path.dirname(path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)
                # Test write access
                with open(path, 'a'):
                    pass
                logger.info(f"Using data file: {path}")
                return path
            except:
                continue
        raise RuntimeError("Cannot find writable location for data")
    
    def _initialize_storage(self):
        """Create CSV with headers if doesn't exist"""
        if not os.path.exists(self.data_path):
            with open(self.data_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
            logger.info("Created new data file")
    
    def _load_historical_data(self):
        """Load all historical data into memory for fast searching"""
        self.historical_data = []
        
        if not os.path.exists(self.data_path):
            return
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Only load complete records with winning numbers
                    if row.get('winning_number') and row['winning_number'] != '':
                        self.historical_data.append({
                            'pos_x': float(row['pos_x']),
                            'pos_y': float(row['pos_y']),
                            'speed_ms': int(row['speed_ms']),
                            'traveled_pockets': int(row['traveled_pockets']),
                            'direction': row['direction'],
                            'pockets_from_winning': int(row['pockets_from_winning'])
                        })
            
            logger.info(f"Loaded {len(self.historical_data)} historical records")
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def store_pending(self, request: PredictionRequest) -> None:
        """Store round data until winning number arrives"""
        # Store the pending round data
        self.pending_rounds[request.round_id] = {
            'timestamp': datetime.now(),
            'data': request.dict()
        }
        
        logger.info(f"Stored pending round {request.round_id}")
    
    def finalize_round(self, round_id: str, winning_number: int) -> bool:
        """Complete round with winning number and calculate pockets_from_winning"""
        if round_id not in self.pending_rounds:
            logger.warning(f"Round {round_id} not found in pending storage")
            return False
        
        pending = self.pending_rounds[round_id]
        round_data = pending['data']
        
        # Calculate pockets from TS2 to winning number
        # This is the key value we need for predictions
        number_at_t2 = round_data['number_at_t2']
        pockets_from_winning = self._calculate_pockets_between(
            number_at_t2, 
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
            'pockets_from_winning': pockets_from_winning
        }
        
        # Save to CSV
        self._save_to_csv(complete_record)
        
        # Add to historical data for future predictions
        self.historical_data.append({
            'pos_x': round_data['pos_x'],
            'pos_y': round_data['pos_y'],
            'speed_ms': round_data['speed_ms_total'],
            'traveled_pockets': round_data['traveled_pockets'],
            'direction': round_data['direction'],
            'pockets_from_winning': pockets_from_winning
        })
        
        # Clean up
        del self.pending_rounds[round_id]
        
        logger.info(f"Finalized round {round_id}: winning={winning_number}, "
                   f"pockets_from_TS2={pockets_from_winning}")
        
        return True
    
    def find_matches_and_predict(self, request: PredictionRequest) -> Optional[int]:
        """
        Find matching historical data and predict winning number
        This implements the exact logic from the document:
        - Find records with same position and speed
        - Use their pockets_from_winning to predict
        """
        if not self.historical_data:
            return None
        
        # Find exact matches (as described in document)
        # "if position is 480,115 and ball speed is 820"
        matches = []
        
        for record in self.historical_data:
            # Check if position matches (with small tolerance for floating point)
            pos_match = (abs(record['pos_x'] - request.pos_x) < 0.1 and 
                        abs(record['pos_y'] - request.pos_y) < 0.1)
            
            # Check if speed matches (exact as in document)
            speed_match = record['speed_ms'] == request.speed_ms_total
            
            # Check if traveled pockets match
            pockets_match = record['traveled_pockets'] == request.traveled_pockets
            
            # Check direction
            direction_match = record['direction'] == request.direction
            
            if pos_match and speed_match and pockets_match and direction_match:
                matches.append(record['pockets_from_winning'])
        
        if not matches:
            logger.info(f"No matches found for position ({request.pos_x}, {request.pos_y}), "
                       f"speed {request.speed_ms_total}ms")
            return None
        
        # Use the most common pockets_from_winning value
        # (In document example, it's assumed to be consistent)
        most_common_offset = max(set(matches), key=matches.count)
        
        # Calculate predicted number
        # "if number under timestamp2 is 9, winning number is 14 pockets away"
        predicted_number = self._add_pockets_to_number(
            request.number_at_t2,
            most_common_offset,
            request.direction
        )
        
        logger.info(f"Found {len(matches)} matches, predicting {predicted_number} "
                   f"({most_common_offset} pockets from {request.number_at_t2})")
        
        return predicted_number
    
    def _calculate_pockets_between(self, from_number: int, to_number: int, direction: str) -> int:
        """Calculate pockets between two numbers in given direction"""
        try:
            from_idx = WHEEL_LAYOUT.index(from_number)
            to_idx = WHEEL_LAYOUT.index(to_number)
        except ValueError:
            logger.error(f"Invalid numbers: {from_number} or {to_number}")
            return 0
        
        if direction == "CW":
            # Clockwise distance
            distance = (to_idx - from_idx) % 37
        else:  # CCW
            # Counter-clockwise distance
            distance = (from_idx - to_idx) % 37
        
        return distance
    
    def _add_pockets_to_number(self, start_number: int, pockets: int, direction: str) -> int:
        """Add pockets to a number in given direction"""
        try:
            start_idx = WHEEL_LAYOUT.index(start_number)
        except ValueError:
            logger.error(f"Invalid start number: {start_number}")
            return start_number
        
        if direction == "CW":
            # Move clockwise
            result_idx = (start_idx + pockets) % 37
        else:  # CCW
            # Move counter-clockwise
            result_idx = (start_idx - pockets) % 37
        
        return WHEEL_LAYOUT[result_idx]
    
    def _save_to_csv(self, record: Dict):
        """Append record to CSV file"""
        try:
            with open(self.data_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(record)
        except Exception as e:
            logger.error(f"Failed to save record: {e}")

# ============================================================================
# API SERVER
# ============================================================================

app = FastAPI(
    title="Simple Roulette Prediction Server",
    description="Direct implementation of prediction method from document",
    version="1.0.0"
)

# Enable CORS for browser extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Initialize storage
storage = SimpleDataStorage()

@app.get("/")
async def root():
    """Server status and info"""
    return {
        "server": "Simple Roulette Prediction Server",
        "version": "1.0.0",
        "status": "running",
        "total_records": len(storage.historical_data),
        "pending_rounds": len(storage.pending_rounds),
        "description": "This server implements exact logic from prediction document"
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Make prediction based on historical data
    Following exact logic: same position + same speed = same result
    """
    try:
        # Store the pending round first
        storage.store_pending(request)
        
        # Try to find matches and predict
        predicted_number = storage.find_matches_and_predict(request)
        
        if predicted_number is not None:
            return {
                "predicted_number": predicted_number,
                "confidence": 1.0,  # Simple binary: found match or not
                "matches_found": 1,  # Simplified for document logic
                "dataset_rows": len(storage.historical_data)
            }
        else:
            return {
                "predicted_number": None,
                "confidence": 0,
                "error": "No matching data found",
                "dataset_rows": len(storage.historical_data),
                "matches_found": 0
            }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_winner")
async def log_winner(request: WinnerRequest):
    """Log the winning number to complete a round"""
    try:
        success = storage.finalize_round(request.round_id, request.winning_number)
        
        if success:
            return {
                "ok": True,
                "stored": True,
                "dataset_rows": len(storage.historical_data)
            }
        else:
            return {
                "ok": False,
                "error": "Round not found in pending storage",
                "dataset_rows": len(storage.historical_data)
            }
    
    except Exception as e:
        logger.error(f"Winner logging error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data_summary")
async def data_summary():
    """Get summary of collected data"""
    return {
        "total_records": len(storage.historical_data),
        "pending_rounds": len(storage.pending_rounds),
        "data_file": storage.data_path,
        "sample_data": storage.historical_data[:5] if storage.historical_data else []
    }

@app.delete("/clear_pending")
async def clear_pending():
    """Clear all pending rounds (admin function)"""
    count = len(storage.pending_rounds)
    storage.pending_rounds.clear()
    return {"cleared": count}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("SIMPLE ROULETTE PREDICTION SERVER")
    print("Direct implementation from document")
    print("="*60)
    print(f"Data file: {storage.data_path}")
    print(f"Records loaded: {len(storage.historical_data)}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
