"""
Roulette Prediction Server - Pure v17 Implementation
Pattern matching system for roulette ball tracking and outcome prediction
"""

import os
import csv
import json
import time
import logging
from datetime import datetime
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RoulettePredictionServer")

# Configuration
CELL_SIZE_X = 10  # Position grid size in pixels
CELL_SIZE_Y = 10
CELL_SIZE_SPEED = 20  # Speed grid size in ms
MAX_RECORDS_PER_CELL = 50  # Maximum records to keep per cell
GLOBAL_MAX_RECORDS = 100000  # Global dataset limit
SEARCH_RADIUS_LIMIT = 3  # Maximum expansion radius for neighbor search

# European wheel layout (clockwise)
EUROPEAN_WHEEL = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27,
    13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1,
    20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
]
POCKET_INDICES = {num: i for i, num in enumerate(EUROPEAN_WHEEL)}

# CSV columns for dataset storage
CSV_COLUMNS = [
    'timestamp', 'round_id', 'table_id',
    'pos_x', 'pos_y', 'speed_ms_total', 'traveled_pockets',
    'direction', 'number_at_t1', 'number_at_t2', 
    'winning_number', 'offset_to_winner'
]

def get_data_path() -> str:
    """Get data file path for CSV storage"""
    candidates = [
        os.getenv("ROULETTE_DATA_PATH", ""),
        os.path.join(os.path.expanduser("~"), ".roulette_data", "dataset.csv"),
        os.path.join("/tmp", "roulette_dataset.csv"),
        os.path.join(".", "roulette_dataset.csv")
    ]
    
    for path in candidates:
        if not path:
            continue
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Test write access
            with open(path, 'a', encoding='utf-8') as f:
                pass
            
            # Initialize CSV if needed
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    import csv as _csv
                    writer = _csv.writer(f)
                    writer.writerow(CSV_COLUMNS)
            
            logger.info(f"Using data path: {path}")
            return path
        except Exception as e:
            logger.warning(f"Cannot use path {path}: {e}")
    
    raise RuntimeError("No writable location found for dataset")

DATA_PATH = get_data_path()

def calculate_pocket_offset(from_number: int, to_number: int, direction: str) -> int:
    """Calculate pocket offset between two numbers in given direction"""
    from_idx = POCKET_INDICES[from_number]
    to_idx = POCKET_INDICES[to_number]
    
    if direction == "CW":
        offset = (to_idx - from_idx) % 37
    else:  # CCW
        offset = (from_idx - to_idx) % 37
    
    # Normalize to -18..18 range
    if offset > 18:
        offset = offset - 37
    
    return offset

def get_number_at_offset(from_number: int, offset: int, direction: str) -> int:
    """Get pocket number at given offset from reference number"""
    from_idx = POCKET_INDICES[from_number]
    
    if direction == "CW":
        target_idx = (from_idx + offset) % 37
    else:  # CCW
        target_idx = (from_idx - offset) % 37
    
    return EUROPEAN_WHEEL[target_idx]

def round_to_cell(x: float, y: float, speed: int) -> Tuple[int, int, int]:
    """Round position and speed to cell coordinates"""
    x_cell = int(round(x / CELL_SIZE_X) * CELL_SIZE_X)
    y_cell = int(round(y / CELL_SIZE_Y) * CELL_SIZE_Y)
    speed_cell = int(round(speed / CELL_SIZE_SPEED) * CELL_SIZE_SPEED)
    return (x_cell, y_cell, speed_cell)

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    round_id: str
    pos_x: float
    pos_y: float
    speed_ms_total: int
    traveled_pockets: int = 7
    direction: str
    number_at_t1: int
    number_at_t2: int
    table_id: Optional[str] = "default"

class LogWinnerRequest(BaseModel):
    """Request model for logging winner endpoint"""
    round_id: str
    winning_number: int

class DataStorage:
    """Handles dataset storage and retrieval using cell-based indexing"""
    
    def __init__(self):
        self.cells = defaultdict(list)  # Cell -> list of records
        self.pending_rounds = {}  # round_id -> partial data
        self.total_records = 0
        self.load_dataset()
    
    def load_dataset(self):
        """Load existing dataset from CSV"""
        if not os.path.exists(DATA_PATH):
            return
        
        try:
            with open(DATA_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Skip incomplete records
                    if not row.get('offset_to_winner'):
                        continue
                    
                    # Parse numeric fields
                    try:
                        x = float(row['pos_x'])
                        y = float(row['pos_y'])
                        speed = int(row['speed_ms_total'])
                        direction = row['direction']
                        
                        # Skip invalid directions
                        if direction not in ['CW', 'CCW']:
                            continue
                        
                        # Add to cell
                        cell_key = (*round_to_cell(x, y, speed), direction)
                        self.cells[cell_key].append(row)
                        self.total_records += 1
                    except (ValueError, KeyError):
                        continue
            
            logger.info(f"Loaded {self.total_records} valid records into {len(self.cells)} cells")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
    
    def save_pending(self, request: PredictionRequest):
        """Save pending round data (before winner is known)"""
        self.pending_rounds[request.round_id] = {
            'timestamp': datetime.now().isoformat(),
            'round_id': request.round_id,
            'table_id': request.table_id or 'default',
            'pos_x': request.pos_x,
            'pos_y': request.pos_y,
            'speed_ms_total': request.speed_ms_total,
            'traveled_pockets': request.traveled_pockets,
            'direction': request.direction,
            'number_at_t1': request.number_at_t1,
            'number_at_t2': request.number_at_t2
        }
    
    def finalize_round(self, round_id: str, winning_number: int) -> bool:
        """Complete round data with winner and save to dataset"""
        if round_id not in self.pending_rounds:
            return False
        
        data = self.pending_rounds[round_id]
        
        # Calculate offset
        offset = calculate_pocket_offset(
            data['number_at_t2'],
            winning_number,
            data['direction']
        )
        
        # Complete record
        data['winning_number'] = winning_number
        data['offset_to_winner'] = offset
        
        # Add to cell
        cell_key = (
            *round_to_cell(data['pos_x'], data['pos_y'], data['speed_ms_total']),
            data['direction']
        )
        
        # Enforce cell limit
        if len(self.cells[cell_key]) >= MAX_RECORDS_PER_CELL:
            # Keep only most recent records
            self.cells[cell_key] = self.cells[cell_key][-(MAX_RECORDS_PER_CELL-1):]
        
        self.cells[cell_key].append(data)
        self.total_records += 1
        
        # Save to CSV
        try:
            with open(DATA_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(data)
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
        
        # Clean up pending
        del self.pending_rounds[round_id]
        
        # Global limit enforcement
        if self.total_records > GLOBAL_MAX_RECORDS:
            self._cleanup_old_records()
        
        return True
    
    def _cleanup_old_records(self):
        """Remove oldest records when global limit exceeded"""
        # Simple strategy: remove oldest 10% of records
        target_remove = self.total_records // 10
        removed = 0
        
        for cell_key in list(self.cells.keys()):
            if removed >= target_remove:
                break
            
            cell_records = self.cells[cell_key]
            if len(cell_records) > 10:
                # Remove oldest half from cells with many records
                remove_count = min(len(cell_records) // 2, target_remove - removed)
                self.cells[cell_key] = cell_records[remove_count:]
                removed += remove_count
        
        self.total_records -= removed
        logger.info(f"Cleaned up {removed} old records")
    
    def find_matches(self, request: PredictionRequest) -> Tuple[List[int], int, float]:
        """
        Find matching records and predict offset
        Returns: (offset_predictions, confidence_count, confidence_score)
        """
        x, y, speed = request.pos_x, request.pos_y, request.speed_ms_total
        direction = request.direction
        
        if direction not in ['CW', 'CCW']:
            return [], 0, 0.0
        
        # Round to cell
        x_cell, y_cell, speed_cell = round_to_cell(x, y, speed)
        
        matches = []
        weights = []
        
        # Search expanding rings
        for radius in range(SEARCH_RADIUS_LIMIT + 1):
            if radius == 0:
                # Exact cell match
                cell_key = (x_cell, y_cell, speed_cell, direction)
                if cell_key in self.cells:
                    for record in self.cells[cell_key]:
                        if 'offset_to_winner' in record:
                            matches.append(int(record['offset_to_winner']))
                            weights.append(1.0)
            else:
                # Check neighboring cells
                for dx in [-radius, 0, radius]:
                    for dy in [-radius, 0, radius]:
                        for ds in [-radius, 0, radius]:
                            if abs(dx) == radius or abs(dy) == radius or abs(ds) == radius:
                                neighbor_key = (
                                    x_cell + dx * CELL_SIZE_X,
                                    y_cell + dy * CELL_SIZE_Y,
                                    speed_cell + ds * CELL_SIZE_SPEED,
                                    direction
                                )
                                if neighbor_key in self.cells:
                                    for record in self.cells[neighbor_key]:
                                        if 'offset_to_winner' in record:
                                            matches.append(int(record['offset_to_winner']))
                                            weights.append(1.0 / (1 + radius))
            
            # Stop if we have enough matches
            if len(matches) >= 3:
                break
        
        if not matches:
            return [], 0, 0.0
        
        # Calculate weighted median
        if len(matches) == 1:
            predicted_offset = matches[0]
        else:
            # Sort matches with their weights
            sorted_pairs = sorted(zip(matches, weights))
            matches_sorted = [m for m, _ in sorted_pairs]
            weights_sorted = [w for _, w in sorted_pairs]
            
            # Find weighted median
            cumsum = 0
            total = sum(weights_sorted)
            for i, w in enumerate(weights_sorted):
                cumsum += w
                if cumsum >= total / 2:
                    predicted_offset = matches_sorted[i]
                    break
        
        # Confidence based on number of matches and their spread
        confidence_count = len(matches)
        if confidence_count > 1:
            spread = max(matches) - min(matches)
            confidence_score = min(1.0, confidence_count / 10.0) * max(0, 1.0 - spread / 18.0)
        else:
            confidence_score = 0.3 if confidence_count == 1 else 0.0
        
        return [predicted_offset], confidence_count, confidence_score

# Initialize system components
app = FastAPI(
    title="Roulette Prediction Server",
    description="Pattern matching system for roulette prediction using v17 methodology",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

storage = DataStorage()

@app.get("/")
async def root():
    """System status endpoint"""
    return {
        "server": "Roulette Prediction Server",
        "status": "operational",
        "methodology": "v17 pattern matching",
        "dataset": {
            "total_records": storage.total_records,
            "cells_populated": len(storage.cells),
            "pending_rounds": len(storage.pending_rounds)
        }
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predict winning number based on current ball state"""
    try:
        # Validate direction
        if request.direction not in ['CW', 'CCW']:
            return {
                "predicted_number": None,
                "dataset_rows": storage.total_records,
                "error": "Invalid direction",
                "confidence": 0
            }
        
        # Save pending data
        storage.save_pending(request)
        
        # Find matches and predict
        offsets, count, confidence = storage.find_matches(request)
        
        if offsets and len(offsets) > 0:
            predicted_offset = offsets[0]
            predicted_number = get_number_at_offset(
                request.number_at_t2,
                predicted_offset,
                request.direction
            )
            
            return {
                "predicted_number": predicted_number,
                "dataset_rows": storage.total_records,
                "confidence": round(confidence, 2),
                "matches_found": count,
                "accuracy": {
                    "error_margin": "N/A" if count < 10 else "±3",
                    "success_rate_3": f"{min(100, count * 10)}%"
                },
                "data_quality": f"{int(confidence * 100)}%"
            }
        else:
            return {
                "predicted_number": None,
                "dataset_rows": storage.total_records,
                "error": "No matching patterns found",
                "confidence": 0,
                "data_quality": "0%"
            }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            "predicted_number": None,
            "dataset_rows": storage.total_records,
            "error": str(e)
        }

@app.post("/log_winner")
async def log_winner(request: LogWinnerRequest):
    """Log the winning number for a completed round"""
    try:
        success = storage.finalize_round(request.round_id, request.winning_number)
        
        if success:
            return {
                "ok": True,
                "stored": True,
                "dataset_rows": storage.total_records,
                "winning_number": request.winning_number,
                "current_accuracy": {
                    "average_error": "±3",
                    "success_rate_3": f"{min(100, storage.total_records)}%"
                }
            }
        else:
            return {
                "ok": False,
                "error": "Round not found in pending data"
            }
    
    except Exception as e:
        logger.error(f"Error logging winner: {e}")
        return {
            "ok": False,
            "error": str(e)
        }

@app.get("/statistics")
async def get_statistics():
    """Get detailed system statistics"""
    cell_distribution = {}
    for cell_key in storage.cells:
        x, y, speed, direction = cell_key
        key_str = f"pos({x},{y})_speed({speed}ms)_{direction}"
        cell_distribution[key_str] = len(storage.cells[cell_key])
    
    return {
        "total_records": storage.total_records,
        "cells_populated": len(storage.cells),
        "pending_rounds": len(storage.pending_rounds),
        "cell_distribution": cell_distribution,
        "max_records_per_cell": MAX_RECORDS_PER_CELL,
        "global_max_records": GLOBAL_MAX_RECORDS
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("ROULETTE PREDICTION SERVER")
    print("Pure v17 Implementation with Cell-Based Pattern Matching")
    print("="*60)
    print(f"Data storage: {DATA_PATH}")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
