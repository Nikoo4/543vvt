"""
Roulette Prediction Server v17
Based on ball speed and traveled pockets method
"""

import os
import csv
import json
import time
import uuid
import logging
from datetime import datetime
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional, Tuple
from statistics import mean, stdev

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RouletteV17")

# ─────────────────────── CSV Configuration ───────────────────────

CSV_COLUMNS = [
    'timestamp', 'round_id', 'ball_speed_ms', 'traveled_pockets',
    'pockets_from_timestamp2_to_winner', 'ball_direction',
    'timestamp1_number', 'timestamp2_number', 'winning_number',
    'timestamp1_position_x', 'timestamp1_position_y',
    'predicted_number', 'prediction_error', 'pattern_matches'
]

# Dataset limits
MIN_DATA_FOR_PREDICTION = 20    # Minimum records to start predictions
OPTIMAL_DATA_SIZE = 500         # Target for good accuracy
MAX_DATASET_SIZE = 5000         # Maximum records to keep

# Matching parameters
SPEED_TOLERANCE_MS = 50         # ±50ms for speed matching
POSITION_TOLERANCE_PX = 30      # ±30px for position matching

# Learning parameters
IMPROVEMENT_WINDOW = 100        # Records to compare for improvement
MIN_IMPROVEMENT = 0.5          # Minimum improvement in pockets
ERROR_THRESHOLD = 4            # Stop learning when average error <= 4

# ─────────────────────── File Management ───────────────────────

def get_data_path() -> str:
    """
    Determine the best available path for data storage
    Priority: ENV > user_home > /tmp > current_dir
    """
    candidates = [
        os.getenv("ROULETTE_DATA_PATH", ""),
        os.path.join(os.path.expanduser("~"), ".roulette_v17", "roulette_v17_data.csv"),
        os.path.join("/tmp", "roulette_v17_data.csv"),
        os.path.join(".", "roulette_v17_data.csv")
    ]
    
    for path in candidates:
        if not path:
            continue
        
        try:
            # Create directory if needed
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Test write access
            with open(path, 'a', encoding='utf-8') as f:
                pass
            
            # Initialize with headers if empty
            if os.path.getsize(path) == 0:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_COLUMNS)
            
            logger.info(f"Using data path: {path}")
            return path
            
        except Exception as e:
            logger.warning(f"Cannot use path {path}: {e}")
            continue
    
    raise RuntimeError("No writable location found for CSV data")

DATA_PATH = get_data_path()

def load_csv_data() -> List[Dict]:
    """Load all records from CSV file"""
    if not os.path.exists(DATA_PATH):
        return []
    
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = []
            for row in reader:
                # Convert numeric fields
                for field in ['ball_speed_ms', 'traveled_pockets', 
                             'pockets_from_timestamp2_to_winner',
                             'timestamp1_position_x', 'timestamp1_position_y',
                             'timestamp1_number', 'timestamp2_number', 
                             'winning_number', 'predicted_number',
                             'prediction_error', 'pattern_matches']:
                    if field in row and row[field]:
                        try:
                            row[field] = int(row[field])
                        except ValueError:
                            pass
                data.append(row)
            return data
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return []

def append_csv_record(record: Dict):
    """Append a single record to CSV file"""
    try:
        with open(DATA_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(record)
    except Exception as e:
        logger.error(f"Error appending to CSV: {e}")

def rewrite_csv_data(data: List[Dict]):
    """Rewrite entire CSV file with filtered data"""
    try:
        with open(DATA_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"Rewrote CSV with {len(data)} records")
    except Exception as e:
        logger.error(f"Error rewriting CSV: {e}")

# ─────────────────────── Wheel Configuration ───────────────────────

EUROPEAN_WHEEL = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27,
    13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1,
    20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
]

POCKET_INDICES = {num: i for i, num in enumerate(EUROPEAN_WHEEL)}

def calculate_pocket_distance(from_number: int, to_number: int, direction: str) -> int:
    """Calculate distance between two pockets in given direction"""
    from_idx = POCKET_INDICES[from_number]
    to_idx = POCKET_INDICES[to_number]
    
    if direction.upper() == "CW":
        distance = (to_idx - from_idx) % 37
    else:  # CCW
        distance = (from_idx - to_idx) % 37
    
    return distance

def get_number_at_offset(from_number: int, offset: int, direction: str) -> int:
    """Get pocket number at given offset from starting number"""
    from_idx = POCKET_INDICES[from_number]
    
    if direction.upper() == "CW":
        target_idx = (from_idx + offset) % 37
    else:  # CCW
        target_idx = (from_idx - offset) % 37
    
    return EUROPEAN_WHEEL[target_idx]

# ─────────────────────── Request/Response Models ───────────────────────

class PredictionRequest(BaseModel):
    round_id: str
    ball_speed_ms: int = Field(..., description="Ball rotation time in milliseconds")
    traveled_pockets: int = Field(..., description="Pockets between timestamp1 and timestamp2")
    ball_direction: str = Field(..., description="CW or CCW")
    timestamp1_number: int = Field(..., description="Number under ball at release")
    timestamp2_number: int = Field(..., description="Number under ball after one rotation")
    total_rotations: Optional[float] = Field(None, description="Total rotations if available")

class LogRoundDataRequest(BaseModel):
    round_id: str
    timestamp1: Dict[str, Any]
    timestamp2: Dict[str, Any]
    ball_speed_ms: int
    traveled_pockets: int
    pockets_from_timestamp2_to_winner: int
    ball_direction: str
    winning_number: int
    timestamp: Optional[int] = None

# ─────────────────────── Pattern Matching Engine ───────────────────────

class PatternMatcher:
    """Find and analyze matching patterns in historical data"""
    
    def __init__(self):
        self.dataset = []
        self.pattern_cache = defaultdict(list)
        
    def load_dataset(self, data: List[Dict]):
        """Load dataset and build pattern cache"""
        self.dataset = data
        self._build_cache()
    
    def _build_cache(self):
        """Build cache for faster pattern matching"""
        self.pattern_cache.clear()
        
        for record in self.dataset:
            if all(key in record for key in ['ball_speed_ms', 'traveled_pockets', 'ball_direction']):
                # Create cache key
                speed_bucket = record['ball_speed_ms'] // 50  # 50ms buckets
                key = f"{speed_bucket}_{record['traveled_pockets']}_{record['ball_direction']}"
                self.pattern_cache[key].append(record)
    
    def find_matches(self, speed_ms: int, traveled_pockets: int, direction: str, 
                    position: Optional[Tuple[int, int]] = None) -> List[Dict]:
        """Find matching patterns in dataset"""
        matches = []
        
        # Try exact bucket first
        speed_bucket = speed_ms // 50
        key = f"{speed_bucket}_{traveled_pockets}_{direction}"
        
        # Check exact bucket and adjacent buckets
        for bucket_offset in [0, -1, 1]:
            check_bucket = speed_bucket + bucket_offset
            check_key = f"{check_bucket}_{traveled_pockets}_{direction}"
            
            if check_key in self.pattern_cache:
                for record in self.pattern_cache[check_key]:
                    # Verify speed is within tolerance
                    if abs(record['ball_speed_ms'] - speed_ms) <= SPEED_TOLERANCE_MS:
                        # Optional position check
                        if position and 'timestamp1_position_x' in record:
                            pos_x_match = abs(record['timestamp1_position_x'] - position[0]) <= POSITION_TOLERANCE_PX
                            pos_y_match = abs(record['timestamp1_position_y'] - position[1]) <= POSITION_TOLERANCE_PX
                            if not (pos_x_match and pos_y_match):
                                continue
                        
                        matches.append(record)
        
        return matches
    
    def predict_offset(self, matches: List[Dict]) -> Tuple[int, float]:
        """Calculate predicted offset from matches"""
        if not matches:
            return 0, 0.0
        
        offsets = [m['pockets_from_timestamp2_to_winner'] for m in matches]
        
        # Calculate average and confidence
        avg_offset = mean(offsets)
        
        if len(offsets) >= 3:
            std_dev = stdev(offsets)
            # Confidence based on consistency
            confidence = max(0, 1 - (std_dev / 10)) * min(len(matches) / 10, 1.0)
        else:
            confidence = len(matches) / 10  # Low confidence with few matches
        
        return int(round(avg_offset)), confidence

# ─────────────────────── Performance Tracking ───────────────────────

class PerformanceTracker:
    """Track prediction accuracy and improvement over time"""
    
    def __init__(self):
        self.error_history = deque(maxlen=50)
        self.prediction_count = 0
        self.direct_hits = 0
        self.within_3 = 0
        self.within_5 = 0
        self.baseline_error = None
        
    def update(self, predicted: int, actual: int, direction: str):
        """Update metrics with new prediction result"""
        error = calculate_pocket_distance(predicted, actual, direction)
        
        # Normalize error to shortest distance
        if error > 18:
            error = 37 - error
        
        self.error_history.append(error)
        self.prediction_count += 1
        
        if error == 0:
            self.direct_hits += 1
        if error <= 3:
            self.within_3 += 1
        if error <= 5:
            self.within_5 += 1
        
        # Set baseline after first 20 predictions
        if self.prediction_count == 20 and self.error_history:
            self.baseline_error = mean(self.error_history)
    
    def get_average_error(self) -> float:
        """Calculate current average error"""
        if not self.error_history:
            return float('inf')
        return mean(self.error_history)
    
    def get_improvement(self) -> float:
        """Calculate improvement percentage from baseline"""
        if not self.baseline_error or self.prediction_count < 30:
            return 0.0
        
        current_error = self.get_average_error()
        if self.baseline_error == 0:
            return 0.0
        
        improvement = ((self.baseline_error - current_error) / self.baseline_error) * 100
        return max(0, improvement)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        avg_error = self.get_average_error()
        
        return {
            "total_predictions": self.prediction_count,
            "average_error": round(avg_error, 1) if avg_error != float('inf') else "N/A",
            "improvement": round(self.get_improvement(), 1),
            "accuracy_rates": {
                "direct_hits": round(self.direct_hits / max(1, self.prediction_count) * 100, 1),
                "within_3": round(self.within_3 / max(1, self.prediction_count) * 100, 1),
                "within_5": round(self.within_5 / max(1, self.prediction_count) * 100, 1)
            }
        }

# ─────────────────────── Learning Control System ───────────────────────

class LearningController:
    """Intelligent system to manage dataset quality and learning"""
    
    def __init__(self):
        self.performance_windows = deque(maxlen=10)  # Track 10 windows of 100 records
        
    def should_stop_learning(self, dataset: List[Dict]) -> Tuple[bool, str]:
        """Determine if learning should stop"""
        
        # Need minimum data
        if len(dataset) < MIN_DATA_FOR_PREDICTION:
            return False, "Collecting initial data"
        
        # Calculate recent performance
        recent_errors = []
        for record in dataset[-50:]:
            if 'prediction_error' in record and record['prediction_error'] is not None:
                recent_errors.append(record['prediction_error'])
        
        if not recent_errors:
            return False, "No prediction data yet"
        
        avg_error = mean(recent_errors)
        
        # Stop if error threshold reached
        if avg_error <= ERROR_THRESHOLD:
            return True, f"Target accuracy reached (avg error: {avg_error:.1f})"
        
        # Check for improvement plateau
        if len(dataset) >= 400:
            older_window = dataset[-400:-200]
            newer_window = dataset[-200:]
            
            older_errors = [r['prediction_error'] for r in older_window 
                          if 'prediction_error' in r and r['prediction_error'] is not None]
            newer_errors = [r['prediction_error'] for r in newer_window 
                          if 'prediction_error' in r and r['prediction_error'] is not None]
            
            if older_errors and newer_errors:
                old_avg = mean(older_errors)
                new_avg = mean(newer_errors)
                improvement = old_avg - new_avg
                
                if improvement < MIN_IMPROVEMENT:
                    return True, f"Learning plateau (improvement: {improvement:.1f})"
        
        # Check dataset size limit
        if len(dataset) >= MAX_DATASET_SIZE - 100:
            return True, "Approaching dataset size limit"
        
        return False, "Active learning"
    
    def filter_poor_quality_data(self, dataset: List[Dict]) -> List[Dict]:
        """Remove poor quality data that doesn't contribute to learning"""
        
        if len(dataset) < 200:
            return dataset  # Don't filter early data
        
        # Analyze pattern reliability
        pattern_performance = defaultdict(list)
        
        for record in dataset:
            if all(key in record for key in ['ball_speed_ms', 'traveled_pockets', 
                                             'ball_direction', 'prediction_error']):
                pattern_key = f"{record['ball_speed_ms']//50}_{record['traveled_pockets']}_{record['ball_direction']}"
                if record['prediction_error'] is not None:
                    pattern_performance[pattern_key].append(record['prediction_error'])
        
        # Identify consistently poor patterns
        poor_patterns = set()
        for pattern, errors in pattern_performance.items():
            if len(errors) >= 5:  # Need enough samples
                avg_error = mean(errors)
                if avg_error > 10:  # Consistently bad predictions
                    poor_patterns.add(pattern)
        
        # Filter dataset
        filtered = []
        removed = 0
        
        for record in dataset:
            pattern_key = f"{record.get('ball_speed_ms', 0)//50}_{record.get('traveled_pockets', 0)}_{record.get('ball_direction', '')}"
            
            # Keep if not a poor pattern or no prediction yet
            if pattern_key not in poor_patterns or 'prediction_error' not in record:
                filtered.append(record)
            else:
                removed += 1
        
        if removed > 0:
            logger.info(f"Filtered {removed} poor quality records")
        
        return filtered

# ─────────────────────── Main Predictor System ───────────────────────

class RouletteV17Predictor:
    """Main prediction system for v17 method"""
    
    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.performance_tracker = PerformanceTracker()
        self.learning_controller = LearningController()
        self.pending_predictions = {}
        
    def initialize(self):
        """Initialize system with existing data"""
        dataset = load_csv_data()
        self.pattern_matcher.load_dataset(dataset)
        
        # Rebuild performance metrics
        for record in dataset:
            if all(key in record for key in ['predicted_number', 'winning_number', 'ball_direction']):
                self.performance_tracker.update(
                    record['predicted_number'],
                    record['winning_number'],
                    record['ball_direction']
                )
        
        logger.info(f"Initialized with {len(dataset)} records")
        logger.info(f"Average error: {self.performance_tracker.get_average_error():.1f}")
    
    def predict(self, request: PredictionRequest) -> Dict[str, Any]:
        """Generate prediction based on pattern matching"""
        
        # Find matching patterns
        matches = self.pattern_matcher.find_matches(
            request.ball_speed_ms,
            request.traveled_pockets,
            request.ball_direction
        )
        
        if not matches:
            # No matches - can't predict
            return {
                "ok": False,
                "error": "No matching patterns found",
                "dataset_rows": len(self.pattern_matcher.dataset)
            }
        
        # Calculate predicted offset
        offset, confidence = self.pattern_matcher.predict_offset(matches)
        
        # Calculate predicted number
        predicted_number = get_number_at_offset(
            request.timestamp2_number,
            offset,
            request.ball_direction
        )
        
        # Store prediction for validation
        self.pending_predictions[request.round_id] = {
            "predicted_number": predicted_number,
            "request": request.dict(),
            "matches": len(matches),
            "timestamp": time.time()
        }
        
        # Get current performance metrics
        stats = self.performance_tracker.get_stats()
        
        # Format response exactly like screenshot
        return {
            "ok": True,
            "round_id": request.round_id,
            "prediction": predicted_number,  # Note: "prediction" not "predicted_number"
            "dataset_rows": len(self.pattern_matcher.dataset),
            "accuracy": {
                "error_margin": int(stats["average_error"]) if stats["average_error"] != "N/A" else "N/A",
                "improvement": stats["improvement"]
            },
            "data_quality": f"{int(confidence * 100)}%"
        }
    
    def log_round_data(self, request: LogRoundDataRequest) -> Dict[str, Any]:
        """Log complete round data and update learning"""
        
        # Check if learning should stop
        dataset = self.pattern_matcher.dataset
        should_stop, reason = self.learning_controller.should_stop_learning(dataset)
        
        if should_stop:
            logger.info(f"Learning stopped: {reason}")
            return {
                "ok": True,
                "stored": False,
                "ignored": True,
                "reason": "learning_complete",
                "message": reason,
                "dataset_rows": len(dataset)
            }
        
        # Check for incomplete data
        if not request.timestamp2 or request.traveled_pockets == 0:
            return {
                "ok": True,
                "stored": False,
                "ignored": True,
                "reason": "incomplete_data",
                "dataset_rows": len(dataset)
            }
        
        # Get prediction if exists
        prediction_data = self.pending_predictions.pop(request.round_id, None)
        predicted_number = prediction_data["predicted_number"] if prediction_data else None
        
        # Calculate prediction error if we made a prediction
        prediction_error = None
        if predicted_number is not None:
            error = calculate_pocket_distance(predicted_number, request.winning_number, request.ball_direction)
            if error > 18:
                error = 37 - error
            prediction_error = error
            
            # Update performance tracker
            self.performance_tracker.update(predicted_number, request.winning_number, request.ball_direction)
        
        # Create record
        record = {
            'timestamp': datetime.now().isoformat(),
            'round_id': request.round_id,
            'ball_speed_ms': request.ball_speed_ms,
            'traveled_pockets': request.traveled_pockets,
            'pockets_from_timestamp2_to_winner': request.pockets_from_timestamp2_to_winner,
            'ball_direction': request.ball_direction,
            'timestamp1_number': request.timestamp1['number'],
            'timestamp2_number': request.timestamp2['number'],
            'winning_number': request.winning_number,
            'timestamp1_position_x': request.timestamp1['position']['x'],
            'timestamp1_position_y': request.timestamp1['position']['y'],
            'predicted_number': predicted_number if predicted_number is not None else '',
            'prediction_error': prediction_error if prediction_error is not None else '',
            'pattern_matches': prediction_data["matches"] if prediction_data else ''
        }
        
        # Append to CSV
        append_csv_record(record)
        
        # Reload dataset
        new_dataset = load_csv_data()
        
        # Filter poor quality data periodically
        if len(new_dataset) % 100 == 0:
            filtered_dataset = self.learning_controller.filter_poor_quality_data(new_dataset)
            if len(filtered_dataset) < len(new_dataset):
                rewrite_csv_data(filtered_dataset)
                new_dataset = filtered_dataset
        
        # Maintain size limit
        if len(new_dataset) > MAX_DATASET_SIZE:
            new_dataset = new_dataset[-MAX_DATASET_SIZE:]
            rewrite_csv_data(new_dataset)
        
        # Update pattern matcher
        self.pattern_matcher.load_dataset(new_dataset)
        
        # Get current stats
        stats = self.performance_tracker.get_stats()
        
        return {
            "ok": True,
            "stored": True,
            "dataset_rows": len(new_dataset),
            "current_accuracy": {
                "average_error": stats["average_error"],
                "improvement": stats["improvement"]
            }
        }

# ─────────────────────── FastAPI Application ───────────────────────

app = FastAPI(
    title="Roulette Prediction Server v17",
    description="Ball speed and traveled pockets based prediction",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Initialize predictor
predictor = RouletteV17Predictor()

@app.on_event("startup")
async def startup():
    """Initialize server on startup"""
    predictor.initialize()
    stats = predictor.performance_tracker.get_stats()
    
    logger.info("="*60)
    logger.info("Roulette Prediction Server v17 Started")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Dataset size: {len(predictor.pattern_matcher.dataset)}")
    logger.info(f"Average error: {stats['average_error']}")
    logger.info(f"Improvement: {stats['improvement']}%")
    logger.info("="*60)

@app.get("/")
async def root():
    """Server status and statistics"""
    stats = predictor.performance_tracker.get_stats()
    dataset_size = len(predictor.pattern_matcher.dataset)
    should_stop, reason = predictor.learning_controller.should_stop_learning(
        predictor.pattern_matcher.dataset
    )
    
    return {
        "server": "Roulette Prediction Server v17",
        "status": "operational",
        "method": "Ball speed + traveled pockets pattern matching",
        "statistics": stats,
        "dataset": {
            "current_size": dataset_size,
            "min_for_prediction": MIN_DATA_FOR_PREDICTION,
            "optimal_size": OPTIMAL_DATA_SIZE,
            "max_size": MAX_DATASET_SIZE
        },
        "learning": {
            "active": not should_stop,
            "status": reason
        }
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Generate prediction endpoint"""
    try:
        result = predictor.predict(request)
        
        # Log prediction
        if result.get("ok"):
            logger.info(f"Prediction: {result['prediction']} "
                       f"(quality: {result.get('data_quality', 'N/A')}, "
                       f"error margin: {result['accuracy']['error_margin']})")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"ok": False, "error": str(e)}

@app.post("/log_winner")
async def log_winner(request: LogRoundDataRequest):
    """Log round data endpoint"""
    try:
        result = predictor.log_round_data(request)
        
        if result.get("stored"):
            logger.info(f"Round logged - Winner: {request.winning_number}, "
                       f"Dataset: {result['dataset_rows']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error logging round: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/statistics")
async def get_statistics():
    """Detailed statistics endpoint"""
    stats = predictor.performance_tracker.get_stats()
    dataset = predictor.pattern_matcher.dataset
    
    # Analyze patterns
    pattern_stats = defaultdict(lambda: {"count": 0, "avg_error": []})
    
    for record in dataset:
        if 'ball_speed_ms' in record and 'traveled_pockets' in record:
            key = f"{record['ball_speed_ms']//100}00ms_{record['traveled_pockets']}pockets"
            pattern_stats[key]["count"] += 1
            
            if 'prediction_error' in record and record['prediction_error'] is not None:
                pattern_stats[key]["avg_error"].append(record['prediction_error'])
    
    # Calculate averages
    for pattern in pattern_stats.values():
        if pattern["avg_error"]:
            pattern["avg_error"] = round(mean(pattern["avg_error"]), 1)
        else:
            pattern["avg_error"] = "N/A"
    
    return {
        "performance": stats,
        "dataset_size": len(dataset),
        "pattern_distribution": dict(pattern_stats),
        "learning_status": predictor.learning_controller.should_stop_learning(dataset)[1]
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("ROULETTE PREDICTION SERVER v17")
    print("="*70)
    print("Method: Ball speed + traveled pockets pattern matching")
    print("Features:")
    print("  • Automatic pattern matching with speed tolerance")
    print("  • Intelligent learning control (stops at ≤4 pockets error)")
    print("  • Poor quality data filtering")
    print("  • Real-time performance tracking")
    print(f"Data storage: {DATA_PATH}")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
