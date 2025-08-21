"""Server-side predictor & dataset storage."""

import os
import csv
import json
import time
import logging
import math
from datetime import datetime
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional, Tuple
from statistics import mean, median

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RouletteV18Pro")

# ═══════════════════════ Configuration ═══════════════════════

# Dataset parameters
MIN_DATA_FOR_PREDICTION = 100   # Minimum records to start predictions
OPTIMAL_DATA_SIZE = 1500        # Target for good accuracy
MAX_DATASET_SIZE = 3000         # Maximum records to keep
CLEANUP_THRESHOLD = 2500        # Start cleanup when reaching this

# Speed gate: ignore spins that are too fast/unstable for T1→T2 (ms)
MIN_BALL_SPEED_MS = 450  # below this, ignore for prediction & dataset

# Matching parameters (RELAXED vs v17)
SPEED_TOLERANCE_MS = 80         # ±80ms for speed matching (was 50)
POSITION_TOLERANCE_PX = 30      # unchanged

# Quality parameters (RELAXED vs v17)
TARGET_ERROR = 3.0              # Target error in pockets
MAX_ACCEPTABLE_ERROR = 10.0     # Maximum acceptable error
MIN_PATTERN_CONFIDENCE = 0.40   # (was 0.60)

# Learning parameters
MAINTENANCE_INTERVAL = 100      # Check quality every N records
PATTERN_MIN_SAMPLES = 5         # Minimum samples to evaluate pattern
HISTORY_SIZE = 100              # Size of prediction history

# Runtime overrides via environment variables (optional)
MIN_DATA_FOR_PREDICTION = int(os.getenv("MIN_DATA_FOR_PREDICTION", MIN_DATA_FOR_PREDICTION))
OPTIMAL_DATA_SIZE = int(os.getenv("OPTIMAL_DATA_SIZE", OPTIMAL_DATA_SIZE))
MAX_DATASET_SIZE = int(os.getenv("MAX_DATASET_SIZE", MAX_DATASET_SIZE))
CLEANUP_THRESHOLD = int(os.getenv("CLEANUP_THRESHOLD", CLEANUP_THRESHOLD))
MIN_BALL_SPEED_MS = int(os.getenv("MIN_BALL_SPEED_MS", MIN_BALL_SPEED_MS))
SPEED_TOLERANCE_MS = int(os.getenv("SPEED_TOLERANCE_MS", SPEED_TOLERANCE_MS))
MIN_PATTERN_CONFIDENCE = float(os.getenv("MIN_PATTERN_CONFIDENCE", MIN_PATTERN_CONFIDENCE))

CSV_COLUMNS = [
    'timestamp', 'round_id', 'ball_speed_ms', 'traveled_pockets',
    'pockets_to_winner', 'ball_direction',
    'timestamp1_number', 'timestamp2_number', 'winning_number',
    'timestamp1_position_x', 'timestamp1_position_y',
    'timestamp2_position_x', 'timestamp2_position_y',
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
                    import csv as _csv
                    writer = _csv.writer(f)
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
    """Distance between two pockets in given direction (0..36)"""
    from_idx = POCKET_INDICES[from_number]
    to_idx = POCKET_INDICES[to_number]
    if direction.upper() == "CW":
        return (to_idx - from_idx) % 37
    return (from_idx - to_idx) % 37

def get_number_at_offset(from_number: int, offset: int, direction: str) -> int:
    """Get pocket number at given offset in given direction"""
    from_idx = POCKET_INDICES[from_number]
    if direction.upper() == "CW":
        target_idx = (from_idx + offset) % 37
    else:
        target_idx = (from_idx - offset) % 37
    return EUROPEAN_WHEEL[target_idx]

# ═══════════════════════ Models ═══════════════════════

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

# ═══════════════════════ Data Storage ═══════════════════════

class DataStorage:
    def __init__(self):
        self.active_dataset = []
        self.pending_round = None
        self.load_dataset()

    def load_dataset(self):
        if not os.path.exists(DATA_PATH):
            self.active_dataset = []
            return
        try:
            with open(DATA_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.active_dataset = []
                for row in reader:
                    # coerce ints
                    for field in ['ball_speed_ms', 'traveled_pockets', 'pockets_to_winner',
                                  'timestamp1_number', 'timestamp2_number', 'winning_number']:
                        if row.get(field, '') != '':
                            try:
                                row[field] = int(row[field])
                            except Exception:
                                pass
                    for field in ['predicted_number', 'prediction_error']:
                        if row.get(field, '') != '':
                            try:
                                row[field] = int(row[field])
                            except Exception:
                                pass
                    for field in ['confidence', 'timestamp1_position_x', 'timestamp1_position_y']:
                        if row.get(field, '') != '':
                            try:
                                row[field] = float(row[field])
                            except Exception:
                                pass
                    self.active_dataset.append(row)
            logger.info(f"Loaded {len(self.active_dataset)} records from CSV")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            self.active_dataset = []

    def save_record(self, record: Dict):
        try:
            with open(DATA_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(record)
            self.active_dataset.append(record)
        except Exception as e:
            logger.error(f"Error saving record: {e}")

    def rewrite_dataset(self):
        try:
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

# ═══════════════════════ Pattern Engine ═══════════════════════

def circular_std(offsets: List[int], modulo: int = 37) -> float:
    """
    Circular standard deviation for discrete offsets modulo `modulo`.
    Returns an equivalent spread in 'pockets' (same units as offsets).
    """
    n = len(offsets)
    if n == 0:
        return 0.0
    # Map offsets to angles
    angles = [2 * math.pi * (o % modulo) / modulo for o in offsets]
    sum_sin = sum(math.sin(a) for a in angles)
    sum_cos = sum(math.cos(a) for a in angles)
    R = (sum_sin ** 2 + sum_cos ** 2) ** 0.5 / n
    if R <= 0:
        # maximally dispersed
        return modulo / math.sqrt(12)  # ~uniform equivalent
    sigma = ( -2.0 * math.log(R) ) ** 0.5  # in radians
    std_eq = sigma * modulo / (2 * math.pi)  # convert back to 'pockets'
    return std_eq

class PatternEngine:
    def __init__(self):
        self.pattern_cache = defaultdict(list)

    def rebuild_cache(self, dataset: List[Dict]):
        self.pattern_cache.clear()
        for record in dataset:
            if all(k in record for k in ['ball_speed_ms', 'traveled_pockets', 'ball_direction', 'pockets_to_winner']):
                speed_bucket = record['ball_speed_ms'] // 50
                key = f"{speed_bucket}_{record['traveled_pockets']}_{record['ball_direction']}"
                self.pattern_cache[key].append(record)

    def find_matches(self, speed: int, pockets: int, direction: str, req_pos=None) -> Tuple[List[Dict], float]:
        """Find dataset records that match speed/pockets/direction and are
        spatially close to the request's T1 position (if provided).
        Args:
            speed: ball period in ms per revolution.
            pockets: traveled pockets between T1 and T2 (expected 7).
            direction: 'CW' or 'CCW'.
            req_pos: optional (x, y) in pixels captured at T1.
        Returns:
            (matches, confidence)
        """

        matches = []
        speed_bucket = speed // 50
        # Expanded bucket search [-2..+2]
        for bucket_offset in (-2, -1, 0, 1, 2):
            key = f"{speed_bucket + bucket_offset}_{pockets}_{direction}"
            if key not in self.pattern_cache:
                continue
            for record in self.pattern_cache[key]:
                if abs(record['ball_speed_ms'] - speed) <= SPEED_TOLERANCE_MS:
                    if req_pos is not None:
                        rx = record.get('timestamp1_position_x')
                        ry = record.get('timestamp1_position_y')
                        if rx in (None, '') or ry in (None, ''):
                            continue
                        try:
                            dx = float(rx) - float(req_pos[0])
                            dy = float(ry) - float(req_pos[1])
                        except Exception:
                            continue
                        if (dx*dx + dy*dy) ** 0.5 > POSITION_TOLERANCE_PX:
                            continue
                    matches.append(record)
        if not matches:
            return [], 0.0
        # Base confidence by sample size (faster ramp)
        base_conf = min(len(matches) / 8.0, 1.0)
        conf = base_conf
        # Consistency by circular spread
        if len(matches) >= 3:
            offsets = [m['pockets_to_winner'] for m in matches]
            spread = circular_std(offsets, 37)  # in pockets
            # Map spread to [0..1]: 0 spread -> 1.0, 10 pockets -> 0
            consistency = max(0.0, 1.0 - (spread / 10.0))
            conf *= consistency
        return matches, float(conf)

    def predict_offset(self, matches: List[Dict]) -> int:
        if not matches:
            return 0
        # Weighted by inverse error if available; else uniform
        offsets, weights = [], []
        for m in matches:
            o = int(m.get('pockets_to_winner', 0))
            offsets.append(o)
            err = m.get('prediction_error')
            w = 1.0 / max(1.0, float(err)) if err not in (None, '') else 1.0
            weights.append(w)
        total_w = sum(weights)
        if total_w <= 0:
            from statistics import mean as _mean
            return int(round(_mean(offsets)))
        weighted = sum(o*w for o, w in zip(offsets, weights)) / total_w
        return int(round(weighted))

# ═══════════════════════ Quality Management ═══════════════════════

class QualityManager:
    def __init__(self):
        self.last_maintenance = 0

    def optimize_dataset(self, dataset: List[Dict]) -> List[Dict]:
        if len(dataset) < MIN_DATA_FOR_PREDICTION:
            return dataset
        pattern_perf = defaultdict(list)
        for r in dataset:
            if all(k in r for k in ['ball_speed_ms','traveled_pockets','ball_direction']):
                key = f"{r['ball_speed_ms']//50}_{r['traveled_pockets']}_{r['ball_direction']}"
                if r.get('prediction_error') not in (None, ''):
                    try:
                        pattern_perf[key].append(float(r['prediction_error']))
                    except Exception:
                        pass
        cleaned = []
        removed = 0
        for r in dataset:
            key = f"{r.get('ball_speed_ms',0)//50}_{r.get('traveled_pockets',0)}_{r.get('ball_direction','')}"
            if key in pattern_perf and len(pattern_perf[key]) >= PATTERN_MIN_SAMPLES:
                avg_err = sum(pattern_perf[key])/len(pattern_perf[key])
                if avg_err > MAX_ACCEPTABLE_ERROR:
                    removed += 1
                    continue
            cleaned.append(r)
        if removed:
            logger.info(f"Removed {removed} poor quality records")
        if len(cleaned) > MAX_DATASET_SIZE:
            cleaned.sort(key=lambda x: float(x.get('prediction_error', MAX_ACCEPTABLE_ERROR)))
            cleaned = cleaned[:MAX_DATASET_SIZE]
        return cleaned

    def should_optimize(self, dataset: List[Dict]) -> bool:
        if len(dataset) < MIN_DATA_FOR_PREDICTION:
            return False
        if len(dataset) - self.last_maintenance >= MAINTENANCE_INTERVAL:
            self.last_maintenance = len(dataset)
            return True
        if len(dataset) >= CLEANUP_THRESHOLD:
            return True
        return False

# ═══════════════════════ Analytics ═══════════════════════

class Analytics:
    def __init__(self):
        self.prediction_history = deque(maxlen=HISTORY_SIZE)

    def track_prediction(self, predicted: int, actual: int, pattern_key: str, direction: str):
        err = calculate_pocket_distance(predicted, actual, direction)
        if err > 18:
            err = 37 - err
        self.prediction_history.append({'predicted': predicted, 'actual': actual, 'error': err, 'ts': time.time()})
        return err

    def get_statistics(self) -> Dict[str, Any]:
        if not self.prediction_history:
            return {'average_error': 'N/A', 'success_rate_3': 'N/A', 'success_rate_5': 'N/A', 'trend': 'collecting'}
        recent = list(self.prediction_history)
        errors = [p['error'] for p in recent]
        if len(errors) >= 20:
            first, second = errors[:len(errors)//2], errors[len(errors)//2:]
            trend = 'improving' if sum(second)/len(second) < sum(first)/len(first) else 'stable'
        else:
            trend = 'collecting'
        def sr(k): 
            return round(100.0 * sum(1 for e in errors if e <= k) / len(errors), 1)
        from statistics import mean, median
        return {
            'average_error': round(mean(errors), 1),
            'median_error': round(median(errors), 1),
            'success_rate_3': sr(3),
            'success_rate_5': sr(5),
            'best': min(errors),
            'worst': max(errors),
            'trend': trend,
            'total_predictions': len(errors)
        }

# ═══════════════════════ Main System ═══════════════════════

class PredictionSystem:
    def __init__(self):
        self.storage = DataStorage()
        self.pattern_engine = PatternEngine()
        self.quality_manager = QualityManager()
        self.analytics = Analytics()
        self.pattern_engine.rebuild_cache(self.storage.active_dataset)
        self.pending_predictions = {}

    def predict(self, request: 'PredictionRequest') -> Dict[str, Any]:
        if request.ball_speed_ms < MIN_BALL_SPEED_MS:
            return {
                'predicted_number': None,
                'dataset_rows': len(self.storage.active_dataset),
                'ignored': True,
                'reason': f'CRITICAL_SPEED_< {MIN_BALL_SPEED_MS}ms',
                'accuracy': self.analytics.get_statistics(),
                'data_quality': '0%'
            }
        # handle pending
        if self.storage.pending_round and self.storage.pending_round['round_id'] != request.round_id:
            self.storage.pending_round = None
        # store pending
        self.storage.pending_round = {
            'round_id': request.round_id,
            'ball_speed_ms': request.ball_speed_ms,
            'traveled_pockets': request.traveled_pockets,
            'ball_direction': request.ball_direction,
            'timestamp1_number': request.timestamp1_number,
            'timestamp2_number': request.timestamp2_number,
            'timestamp1_position_x': request.timestamp1_position.get('x') if request.timestamp1_position else None,
            'timestamp1_position_y': request.timestamp1_position.get('y') if request.timestamp1_position else None,
            'timestamp2_position_x': request.timestamp2_position.get('x') if request.timestamp2_position else None,
            'timestamp2_position_y': request.timestamp2_position.get('y') if request.timestamp2_position else None,
            'timestamp': time.time()
        }
        dataset_size = len(self.storage.active_dataset)
        if dataset_size < MIN_DATA_FOR_PREDICTION:
            return {
                'predicted_number': None,
                'dataset_rows': dataset_size,
                'error': f'Need {MIN_DATA_FOR_PREDICTION - dataset_size} more samples',
                'accuracy': {'error_margin': 'N/A', 'success_rate_3': 'N/A'},
                'data_quality': '0%'
            }
        req_pos = None
        if request.timestamp1_position:
            req_pos = (
                request.timestamp1_position.get('x'),
                request.timestamp1_position.get('y')
            )
        matches, confidence = self.pattern_engine.find_matches(
            request.ball_speed_ms,
            request.traveled_pockets,
            request.ball_direction,
            req_pos
        )
        if not matches or confidence < MIN_PATTERN_CONFIDENCE:
            return {
                'predicted_number': None,
                'dataset_rows': dataset_size,
                'error': 'No reliable pattern found',
                'accuracy': self.analytics.get_statistics(),
                'data_quality': f'{int(confidence * 100)}%'
            }
        offset = self.pattern_engine.predict_offset(matches)
        predicted_number = get_number_at_offset(request.timestamp2_number, offset, request.ball_direction)
        self.pending_predictions[request.round_id] = {
            'predicted': predicted_number,
            'pattern_matches': len(matches),
            'confidence': confidence
        }
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
        if not self.storage.pending_round or self.storage.pending_round['round_id'] != round_id:
            return {'ok': False, 'error': 'No pending data for this round'}
        pending = self.storage.pending_round
        pockets_to_winner = calculate_pocket_distance(
            pending['timestamp2_number'], winning_number, pending['ball_direction']
        )
        prediction_data = self.pending_predictions.pop(round_id, None)
        predicted_number = prediction_data['predicted'] if prediction_data else None
        prediction_error = None
        if predicted_number is not None:
            prediction_error = self.analytics.track_prediction(
                predicted_number, winning_number,
                f"{pending['ball_speed_ms']//50}_{pending['traveled_pockets']}_{pending['ball_direction']}",
                pending['ball_direction']
            )
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
            'timestamp2_position_x': pending.get('timestamp2_position_x', ''),
            'timestamp2_position_y': pending.get('timestamp2_position_y', ''),
            'predicted_number': predicted_number if predicted_number is not None else '',
            'prediction_error': prediction_error if prediction_error is not None else '',
            'confidence': prediction_data['confidence'] if prediction_data else ''
        }
        self.storage.save_record(record)
        self.pattern_engine.rebuild_cache(self.storage.active_dataset)
        self.storage.pending_round = None
        if self.quality_manager.should_optimize(self.storage.active_dataset):
            self.storage.active_dataset = self.quality_manager.optimize_dataset(self.storage.active_dataset)
            self.storage.rewrite_dataset()
            self.pattern_engine.rebuild_cache(self.storage.active_dataset)
        stats = self.analytics.get_statistics()
        return {
            'ok': True, 'stored': True, 'dataset_rows': len(self.storage.active_dataset),
            'winning_number': winning_number, 'predicted_number': predicted_number,
            'error': prediction_error if prediction_error is not None else 'N/A',
            'current_accuracy': {
                'average_error': stats['average_error'],
                'success_rate_3': f"{stats['success_rate_3']}%"
            }
        }

# ═══════════════════════ FastAPI App ═══════════════════════

app = FastAPI(
    title="Roulette Prediction Server v18 Professional",
    description="Advanced pattern matching with circular statistics",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

predictor = PredictionSystem()

@app.on_event("startup")
async def startup():
    stats = predictor.analytics.get_statistics()
    logger.info("="*70)
    logger.info("ROULETTE PREDICTION SERVER V18 PROFESSIONAL")
    logger.info("="*70)
    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Dataset size: {len(predictor.storage.active_dataset)}")
    logger.info(f"Minimum for predictions: {MIN_DATA_FOR_PREDICTION}")
    logger.info(f"Current accuracy: {stats['average_error']}")
    logger.info("="*70)

@app.get("/")
async def root():
    stats = predictor.analytics.get_statistics()
    dataset_size = len(predictor.storage.active_dataset)
    return {
        "server": "Roulette Prediction Server v18 Professional",
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
    try:
        result = predictor.predict(request)
        if result.get('predicted_number'):
            logger.info(f"Prediction: {result['predicted_number']} "
                        f"(confidence: {result.get('confidence','N/A')}, "
                        f"quality: {result.get('data_quality','N/A')})")
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"predicted_number": None, "error": str(e)}

@app.post("/log_winner")
async def log_winner(request: LogWinnerRequest):
    try:
        result = predictor.log_winner(request.round_id, request.winning_number)
        if result.get('stored'):
            logger.info(f"Winner logged: {request.winning_number}, "
                        f"Predicted: {result.get('predicted_number','N/A')}, "
                        f"Error: {result.get('error','N/A')}")
        return result
    except Exception as e:
        logger.error(f"Error logging winner: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/statistics")
async def get_statistics():
    stats = predictor.analytics.get_statistics()
    dataset = predictor.storage.active_dataset
    pattern_distribution = defaultdict(int)
    for r in dataset:
        if 'ball_speed_ms' in r and 'traveled_pockets' in r:
            key = f"{r['ball_speed_ms']//100}00ms_{r['traveled_pockets']}pockets"
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
    predictor.storage.pending_round = None
    predictor.pending_predictions.clear()
    return {"status": "cleared"}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("ROULETTE PREDICTION SERVER V18 PROFESSIONAL")
    print("="*70)
    print("Features:")
    print("  • Two-phase data collection (predict → log_winner)")
    print("  • Circular-std based confidence with expanded speed buckets")
    print("  • Automatic quality optimization")
    print("  • Real-time performance tracking")
    print("  • Adaptive dataset management")
    print(f"Data storage: {DATA_PATH}")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
