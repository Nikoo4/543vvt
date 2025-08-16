"""
Roulette Prediction Server v17 - Professional Edition
Advanced pattern matching with intelligent data management
"""

import os
import csv
import json
import time
import logging
from datetime import datetime
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional, Tuple
from statistics import mean, median, stdev

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RouletteV17Pro")

# ═══════════════════════ Configuration ═══════════════════════

# Dataset parameters
MIN_DATA_FOR_PREDICTION = 100   # Minimum records to start predictions
OPTIMAL_DATA_SIZE = 1500        # Target for good accuracy
MAX_DATASET_SIZE = 3000         # Maximum records to keep
CLEANUP_THRESHOLD = 2500        # Start cleanup when reaching this

# Matching parameters
SPEED_TOLERANCE_MS = 50         # ±50ms for speed matching
POSITION_TOLERANCE_PX = 30      # ±30px for position matching

# Quality parameters
TARGET_ERROR = 3.0              # Target error in pockets
MAX_ACCEPTABLE_ERROR = 10.0     # Maximum acceptable error
MIN_PATTERN_CONFIDENCE = 0.6    # Minimum confidence to predict

# Learning parameters
MAINTENANCE_INTERVAL = 100      # Check quality every N records
PATTERN_MIN_SAMPLES = 5         # Minimum samples to evaluate pattern
HISTORY_SIZE = 100              # Size of prediction history

# CSV Configuration
CSV_COLUMNS = [
    'timestamp', 'round_id', 'ball_speed_ms', 'traveled_pockets',
    'pockets_to_winner', 'ball_direction',
    'timestamp1_number', 'timestamp2_number', 'winning_number',
    'timestamp1_position_x', 'timestamp1_position_y',
    'predicted_number', 'prediction_error', 'confidence'
]

# ═══════════════════════ File Management ═══════════════════════

def get_data_path() -> str:
    """Get optimal path for data storage"""
    candidates = [
        os.getenv("ROULETTE_DATA_PATH", ""),
        os.path.join(os.path.expanduser("~"), ".roulette_v17", "roulette_v17_pro.csv"),
        os.path.join("/tmp", "roulette_v17_pro.csv"),
        os.path.join(".", "roulette_v17_pro.csv")
    ]
    
    for path in candidates:
        if not path:
            continue
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            with open(path, 'a', encoding='utf-8') as f:
                pass
            
            if os.path.getsize(path) == 0:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_COLUMNS)
            
            logger.info(f"Using data path: {path}")
            return path
        except Exception as e:
            logger.warning(f"Cannot use path {path}: {e}")
    
    raise RuntimeError("No writable location found for CSV data")

DATA_PATH = get_data_path()
BACKUP_PATH = DATA_PATH + ".backup"

# ═══════════════════════ Wheel Configuration ═══════════════════════

EUROPEAN_WHEEL = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27,
    13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1,
    20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
]

POCKET_INDICES = {num: i for i, num in enumerate(EUROPEAN_WHEEL)}

def calculate_pocket_distance(from_number: int, to_number: int, direction: str) -> int:
    """Calculate distance between two pockets"""
    from_idx = POCKET_INDICES[from_number]
    to_idx = POCKET_INDICES[to_number]
    
    if direction.upper() == "CW":
        distance = (to_idx - from_idx) % 37
    else:
        distance = (from_idx - to_idx) % 37
    
    return distance

def get_number_at_offset(from_number: int, offset: int, direction: str) -> int:
    """Get pocket number at given offset"""
    from_idx = POCKET_INDICES[from_number]
    
    if direction.upper() == "CW":
        target_idx = (from_idx + offset) % 37
    else:
        target_idx = (from_idx - offset) % 37
    
    return EUROPEAN_WHEEL[target_idx]

# ═══════════════════════ Request/Response Models ═══════════════════════

class PredictionRequest(BaseModel):
    round_id: str
    ball_speed_ms: int
    traveled_pockets: int
    ball_direction: str
    timestamp1_number: int
    timestamp2_number: int
    timestamp1_position: Optional[Dict[str, float]] = None
    timestamp2_position: Optional[Dict[str, float]] = None

class LogWinnerRequest(BaseModel):
    round_id: str
    winning_number: int

# ═══════════════════════ Data Storage System ═══════════════════════

class DataStorage:
    """Manages data storage and retrieval"""
    
    def __init__(self):
        self.active_dataset = []
        self.pending_round = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load dataset from CSV"""
        if not os.path.exists(DATA_PATH):
            self.active_dataset = []
            return
        
        try:
            with open(DATA_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.active_dataset = []
                for row in reader:
                    # Convert numeric fields
                    for field in ['ball_speed_ms', 'traveled_pockets', 'pockets_to_winner',
                                 'timestamp1_number', 'timestamp2_number', 'winning_number',
                                 'predicted_number', 'prediction_error']:
                        if field in row and row[field]:
                            try:
                                row[field] = int(row[field])
                            except ValueError:
                                pass
                    
                    for field in ['confidence', 'timestamp1_position_x', 'timestamp1_position_y']:
                        if field in row and row[field]:
                            try:
                                row[field] = float(row[field])
                            except ValueError:
                                pass
                    
                    self.active_dataset.append(row)
            
            logger.info(f"Loaded {len(self.active_dataset)} records from CSV")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            self.active_dataset = []
    
    def save_record(self, record: Dict):
        """Append single record to CSV"""
        try:
            with open(DATA_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(record)
            self.active_dataset.append(record)
        except Exception as e:
            logger.error(f"Error saving record: {e}")
    
    def rewrite_dataset(self):
        """Rewrite entire CSV with current dataset"""
        try:
            # Create backup first
            if os.path.exists(DATA_PATH):
                import shutil
                shutil.copy2(DATA_PATH, BACKUP_PATH)
            
            with open(DATA_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(self.active_dataset)
            
            logger.info(f"Rewrote CSV with {len(self.active_dataset)} records")
        except Exception as e:
            logger.error(f"Error rewriting CSV: {e}")

# ═══════════════════════ Pattern Matching Engine ═══════════════════════

class PatternEngine:
    """Advanced pattern matching with caching"""
    
    def __init__(self):
        self.pattern_cache = defaultdict(list)
        self.quality_index = {}
        
    def rebuild_cache(self, dataset: List[Dict]):
        """Rebuild pattern cache for fast lookup"""
        self.pattern_cache.clear()
        
        for record in dataset:
            if all(key in record for key in ['ball_speed_ms', 'traveled_pockets', 'ball_direction']):
                # Create cache key with speed buckets
                speed_bucket = record['ball_speed_ms'] // 50
                key = f"{speed_bucket}_{record['traveled_pockets']}_{record['ball_direction']}"
                self.pattern_cache[key].append(record)
    
    def find_matches(self, speed: int, pockets: int, direction: str) -> Tuple[List[Dict], float]:
        """Find matching patterns and calculate confidence"""
        matches = []
        speed_bucket = speed // 50
        
        # Check exact and adjacent buckets
        for bucket_offset in [0, -1, 1]:
            check_bucket = speed_bucket + bucket_offset
            key = f"{check_bucket}_{pockets}_{direction}"
            
            if key in self.pattern_cache:
                for record in self.pattern_cache[key]:
                    if abs(record['ball_speed_ms'] - speed) <= SPEED_TOLERANCE_MS:
                        matches.append(record)
        
        if not matches:
            return [], 0.0
        
        # Calculate confidence based on sample size and consistency
        confidence = min(len(matches) / 10, 1.0)  # More samples = higher confidence
        
        if len(matches) >= 3:
            offsets = [m['pockets_to_winner'] for m in matches if 'pockets_to_winner' in m]
            if offsets and len(offsets) >= 3:
                std_dev = stdev(offsets) if len(offsets) > 1 else 0
                consistency_factor = max(0, 1 - (std_dev / 10))
                confidence *= consistency_factor
        
        return matches, confidence
    
    def predict_offset(self, matches: List[Dict]) -> int:
        """Calculate weighted average offset"""
        if not matches:
            return 0
        
        offsets = []
        weights = []
        
        for match in matches:
            if 'pockets_to_winner' in match:
                offset = match['pockets_to_winner']
                offsets.append(offset)
                
                # Weight by inverse of error if available
                if 'prediction_error' in match and match['prediction_error'] is not None:
                    error = max(1, match['prediction_error'])
                    weight = 1.0 / error
                else:
                    weight = 1.0
                
                weights.append(weight)
        
        if not offsets:
            return 0
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            weighted_sum = sum(o * w for o, w in zip(offsets, weights))
            return int(round(weighted_sum / total_weight))
        
        return int(round(mean(offsets)))

# ═══════════════════════ Quality Management System ═══════════════════════

class QualityManager:
    """Manages dataset quality and optimization"""
    
    def __init__(self):
        self.last_maintenance = 0
        
    def optimize_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Optimize dataset by removing poor quality data"""
        if len(dataset) < MIN_DATA_FOR_PREDICTION:
            return dataset
        
        # Group by patterns
        pattern_performance = defaultdict(list)
        
        for record in dataset:
            if all(key in record for key in ['ball_speed_ms', 'traveled_pockets', 'ball_direction']):
                pattern_key = f"{record['ball_speed_ms']//50}_{record['traveled_pockets']}_{record['ball_direction']}"
                
                if 'prediction_error' in record and record['prediction_error'] is not None:
                    pattern_performance[pattern_key].append({
                        'error': record['prediction_error'],
                        'record': record
                    })
        
        # Identify and remove poor patterns
        cleaned_dataset = []
        removed_count = 0
        
        for record in dataset:
            pattern_key = f"{record.get('ball_speed_ms', 0)//50}_{record.get('traveled_pockets', 0)}_{record.get('ball_direction', '')}"
            
            # Check if pattern is poor
            if pattern_key in pattern_performance:
                pattern_data = pattern_performance[pattern_key]
                if len(pattern_data) >= PATTERN_MIN_SAMPLES:
                    avg_error = mean([p['error'] for p in pattern_data])
                    
                    # Remove if error too high
                    if avg_error > MAX_ACCEPTABLE_ERROR:
                        removed_count += 1
                        continue
            
            cleaned_dataset.append(record)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} poor quality records")
        
        # If dataset still too large, keep only best records
        if len(cleaned_dataset) > MAX_DATASET_SIZE:
            # Sort by prediction error (best first)
            cleaned_dataset.sort(
                key=lambda x: x.get('prediction_error', MAX_ACCEPTABLE_ERROR)
            )
            cleaned_dataset = cleaned_dataset[:MAX_DATASET_SIZE]
        
        return cleaned_dataset
    
    def should_optimize(self, dataset: List[Dict]) -> bool:
        """Check if optimization is needed"""
        if len(dataset) < MIN_DATA_FOR_PREDICTION:
            return False
        
        # Optimize every MAINTENANCE_INTERVAL records
        if len(dataset) - self.last_maintenance >= MAINTENANCE_INTERVAL:
            self.last_maintenance = len(dataset)
            return True
        
        # Force optimization if approaching limits
        if len(dataset) >= CLEANUP_THRESHOLD:
            return True
        
        return False

# ═══════════════════════ Real-time Analytics ═══════════════════════

class Analytics:
    """Track real-time performance and statistics"""
    
    def __init__(self):
        self.prediction_history = deque(maxlen=HISTORY_SIZE)
        self.pattern_stats = defaultdict(lambda: {'total': 0, 'errors': []})
    
    def track_prediction(self, predicted: int, actual: int, pattern_key: str, direction: str):
        """Track prediction accuracy"""
        error = calculate_pocket_distance(predicted, actual, direction)
        
        # Normalize to shortest distance
        if error > 18:
            error = 37 - error
        
        self.prediction_history.append({
            'predicted': predicted,
            'actual': actual,
            'error': error,
            'timestamp': time.time()
        })
        
        self.pattern_stats[pattern_key]['total'] += 1
        self.pattern_stats[pattern_key]['errors'].append(error)
        
        return error
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        if not self.prediction_history:
            return {
                'average_error': 'N/A',
                'success_rate_3': 'N/A',
                'success_rate_5': 'N/A',
                'trend': 'collecting'
            }
        
        recent = list(self.prediction_history)
        errors = [p['error'] for p in recent]
        
        # Calculate trend
        if len(errors) >= 20:
            first_half = errors[:len(errors)//2]
            second_half = errors[len(errors)//2:]
            trend = 'improving' if mean(second_half) < mean(first_half) else 'stable'
        else:
            trend = 'collecting'
        
        return {
            'average_error': round(mean(errors), 1),
            'median_error': round(median(errors), 1),
            'success_rate_3': round(len([e for e in errors if e <= 3]) / len(errors) * 100, 1),
            'success_rate_5': round(len([e for e in errors if e <= 5]) / len(errors) * 100, 1),
            'best': min(errors),
            'worst': max(errors),
            'trend': trend,
            'total_predictions': len(self.prediction_history)
        }

# ═══════════════════════ Main Prediction System ═══════════════════════

class PredictionSystem:
    """Main prediction system orchestrator"""
    
    def __init__(self):
        self.storage = DataStorage()
        self.pattern_engine = PatternEngine()
        self.quality_manager = QualityManager()
        self.analytics = Analytics()
        
        # Rebuild pattern cache
        self.pattern_engine.rebuild_cache(self.storage.active_dataset)
        
        # Track pending predictions
        self.pending_predictions = {}
    
    def predict(self, request: PredictionRequest) -> Dict[str, Any]:
        """Generate prediction"""
        
        # Handle pending data
        if self.storage.pending_round and self.storage.pending_round['round_id'] != request.round_id:
            logger.warning(f"Dropping incomplete round {self.storage.pending_round['round_id']} - no winner received")
            self.storage.pending_round = None
        
        # Store new pending round
        self.storage.pending_round = {
            'round_id': request.round_id,
            'ball_speed_ms': request.ball_speed_ms,
            'traveled_pockets': request.traveled_pockets,
            'ball_direction': request.ball_direction,
            'timestamp1_number': request.timestamp1_number,
            'timestamp2_number': request.timestamp2_number,
            'timestamp1_position_x': request.timestamp1_position.get('x') if request.timestamp1_position else None,
            'timestamp1_position_y': request.timestamp1_position.get('y') if request.timestamp1_position else None,
            'timestamp': time.time()
        }
        
        # Check if we have enough data
        dataset_size = len(self.storage.active_dataset)
        
        if dataset_size < MIN_DATA_FOR_PREDICTION:
            return {
                'predicted_number': None,
                'dataset_rows': dataset_size,
                'error': f'Need {MIN_DATA_FOR_PREDICTION - dataset_size} more samples',
                'accuracy': {'error_margin': 'N/A', 'success_rate_3': 'N/A'},
                'data_quality': '0%'
            }
        
        # Find matching patterns
        matches, confidence = self.pattern_engine.find_matches(
            request.ball_speed_ms,
            request.traveled_pockets,
            request.ball_direction
        )
        
        if not matches or confidence < MIN_PATTERN_CONFIDENCE:
            return {
                'predicted_number': None,
                'dataset_rows': dataset_size,
                'error': 'No reliable pattern found',
                'accuracy': self.analytics.get_statistics(),
                'data_quality': f'{int(confidence * 100)}%'
            }
        
        # Calculate prediction
        offset = self.pattern_engine.predict_offset(matches)
        predicted_number = get_number_at_offset(
            request.timestamp2_number,
            offset,
            request.ball_direction
        )
        
        # Store prediction for validation
        self.pending_predictions[request.round_id] = {
            'predicted': predicted_number,
            'pattern_matches': len(matches),
            'confidence': confidence
        }
        
        # Get statistics
        stats = self.analytics.get_statistics()
        
        return {
            'predicted_number': predicted_number,
            'dataset_rows': dataset_size,
            'accuracy': {
                'error_margin': stats['average_error'],
                'success_rate_3': f"{stats['success_rate_3']}%",
                'trend': stats['trend']
            },
            'data_quality': f'{int(confidence * 100)}%',
            'confidence': round(confidence, 2)
        }
    
    def log_winner(self, round_id: str, winning_number: int) -> Dict[str, Any]:
        """Log winning number and complete round data"""
        
        # Check for pending round
        if not self.storage.pending_round or self.storage.pending_round['round_id'] != round_id:
            return {
                'ok': False,
                'error': 'No pending data for this round'
            }
        
        # Get pending data
        pending = self.storage.pending_round
        
        # Calculate pockets to winner
        pockets_to_winner = calculate_pocket_distance(
            pending['timestamp2_number'],
            winning_number,
            pending['ball_direction']
        )
        
        # Get prediction if exists
        prediction_data = self.pending_predictions.pop(round_id, None)
        predicted_number = prediction_data['predicted'] if prediction_data else None
        
        # Calculate error if we made a prediction
        prediction_error = None
        if predicted_number is not None:
            prediction_error = self.analytics.track_prediction(
                predicted_number,
                winning_number,
                f"{pending['ball_speed_ms']//50}_{pending['traveled_pockets']}_{pending['ball_direction']}",
                pending['ball_direction']
            )
        
        # Create complete record
        record = {
            'timestamp': datetime.now().isoformat(),
            'round_id': round_id,
            'ball_speed_ms': pending['ball_speed_ms'],
            'traveled_pockets': pending['traveled_pockets'],
            'pockets_to_winner': pockets_to_winner,
            'ball_direction': pending['ball_direction'],
            'timestamp1_number': pending['timestamp1_number'],
            'timestamp2_number': pending['timestamp2_number'],
            'winning_number': winning_number,
            'timestamp1_position_x': pending.get('timestamp1_position_x', ''),
            'timestamp1_position_y': pending.get('timestamp1_position_y', ''),
            'predicted_number': predicted_number if predicted_number is not None else '',
            'prediction_error': prediction_error if prediction_error is not None else '',
            'confidence': prediction_data['confidence'] if prediction_data else ''
        }
        
        # Save to dataset
        self.storage.save_record(record)
        
        # Clear pending
        self.storage.pending_round = None
        
        # Check if optimization needed
        if self.quality_manager.should_optimize(self.storage.active_dataset):
            logger.info("Running dataset optimization...")
            self.storage.active_dataset = self.quality_manager.optimize_dataset(
                self.storage.active_dataset
            )
            self.storage.rewrite_dataset()
            self.pattern_engine.rebuild_cache(self.storage.active_dataset)
        
        # Get updated stats
        stats = self.analytics.get_statistics()
        
        return {
            'ok': True,
            'stored': True,
            'dataset_rows': len(self.storage.active_dataset),
            'winning_number': winning_number,
            'predicted_number': predicted_number,
            'error': prediction_error if prediction_error is not None else 'N/A',
            'current_accuracy': {
                'average_error': stats['average_error'],
                'success_rate_3': f"{stats['success_rate_3']}%"
            }
        }

# ═══════════════════════ FastAPI Application ═══════════════════════

app = FastAPI(
    title="Roulette Prediction Server v17 Professional",
    description="Advanced pattern matching with intelligent data management",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Initialize prediction system
predictor = PredictionSystem()

@app.on_event("startup")
async def startup():
    """Initialize server on startup"""
    stats = predictor.analytics.get_statistics()
    
    logger.info("="*70)
    logger.info("ROULETTE PREDICTION SERVER V17 PROFESSIONAL")
    logger.info("="*70)
    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Dataset size: {len(predictor.storage.active_dataset)}")
    logger.info(f"Minimum for predictions: {MIN_DATA_FOR_PREDICTION}")
    logger.info(f"Current accuracy: {stats['average_error']}")
    logger.info("="*70)

@app.get("/")
async def root():
    """Server status and statistics"""
    stats = predictor.analytics.get_statistics()
    dataset_size = len(predictor.storage.active_dataset)
    
    return {
        "server": "Roulette Prediction Server v17 Professional",
        "status": "operational",
        "statistics": stats,
        "dataset": {
            "current_size": dataset_size,
            "min_for_prediction": MIN_DATA_FOR_PREDICTION,
            "optimal_size": OPTIMAL_DATA_SIZE,
            "max_size": MAX_DATASET_SIZE
        },
        "pending_round": predictor.storage.pending_round['round_id'] if predictor.storage.pending_round else None
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Generate prediction endpoint"""
    try:
        result = predictor.predict(request)
        
        if result.get('predicted_number'):
            logger.info(f"Prediction: {result['predicted_number']} "
                       f"(confidence: {result.get('confidence', 'N/A')}, "
                       f"quality: {result.get('data_quality', 'N/A')})")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            "predicted_number": None,
            "error": str(e)
        }

@app.post("/log_winner")
async def log_winner(request: LogWinnerRequest):
    """Log winning number endpoint"""
    try:
        result = predictor.log_winner(request.round_id, request.winning_number)
        
        if result.get('stored'):
            logger.info(f"Winner logged: {request.winning_number}, "
                       f"Predicted: {result.get('predicted_number', 'N/A')}, "
                       f"Error: {result.get('error', 'N/A')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error logging winner: {e}")
        return {
            "ok": False,
            "error": str(e)
        }

@app.get("/statistics")
async def get_statistics():
    """Detailed statistics endpoint"""
    stats = predictor.analytics.get_statistics()
    dataset = predictor.storage.active_dataset
    
    # Pattern analysis
    pattern_distribution = defaultdict(int)
    for record in dataset:
        if 'ball_speed_ms' in record and 'traveled_pockets' in record:
            key = f"{record['ball_speed_ms']//100}00ms_{record['traveled_pockets']}pockets"
            pattern_distribution[key] += 1
    
    return {
        "performance": stats,
        "dataset_size": len(dataset),
        "pattern_distribution": dict(pattern_distribution),
        "pending_rounds": len(predictor.pending_predictions),
        "last_maintenance": predictor.quality_manager.last_maintenance
    }

@app.delete("/clear_pending")
async def clear_pending():
    """Clear all pending predictions"""
    predictor.storage.pending_round = None
    predictor.pending_predictions.clear()
    return {"status": "cleared"}

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("ROULETTE PREDICTION SERVER V17 PROFESSIONAL")
    print("="*70)
    print("Features:")
    print("  • Two-phase data collection (predict → log_winner)")
    print("  • Intelligent pattern matching with confidence scoring")
    print("  • Automatic quality optimization")
    print("  • Real-time performance tracking")
    print("  • Adaptive dataset management")
    print(f"Data storage: {DATA_PATH}")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
