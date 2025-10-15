"""
Evolution Roulette Prediction Server
Dealer-based prediction system with exact matching
Version: 3.0.0
"""

import os
import csv
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import Counter

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

# Database configuration
DATABASE_FILE = "roulette_patterns.csv"

# CSV columns
CSV_COLUMNS = [
    'timestamp', 'round_id', 'dealer_name', 'rotor_speed_ms', 
    'ball_speed_ms', 'direction', 'winning_number'
]

# Search settings
USE_DEALER_FILTER = True        # Search by dealer name (can be changed to False)
ROTOR_TOLERANCE_MS = 0          # Exact match for rotor speed (can be changed)
BALL_TOLERANCE_MS = 0           # Exact match for ball speed (can be changed)

# ============================================================================
# DATA MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request for prediction based on measurements"""
    round_id: str = Field(..., min_length=5, max_length=50)
    dealer_name: str = Field(..., min_length=2, max_length=50)
    rotor_speed_ms: float = Field(..., ge=1000, le=10000, description="Rotor rotation time in ms")
    ball_speed_ms: int = Field(..., ge=200, le=10000, description="Ball rotation time in ms")
    direction: str = Field(..., pattern="^(CW|CCW)$", description="Ball direction")
    table_id: str = Field(default="evolution_roulette")

class WinnerRequest(BaseModel):
    """Log winning number for completed round"""
    round_id: str = Field(..., min_length=5, max_length=50)
    winning_number: int = Field(..., ge=0, le=36)

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class PatternDatabase:
    """Manages pattern storage and matching with dealer-based approach"""
    
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
    
    def find_matches(self, dealer_name: str, rotor_speed: float, ball_speed: int, 
                    direction: str) -> List[int]:
        """
        Find matching patterns in database
        Returns list of winning numbers that match the criteria
        """
        if not os.path.exists(DATABASE_FILE):
            return []
        
        winning_numbers = []
        
        try:
            with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Filter by dealer if enabled
                    if USE_DEALER_FILTER and row['dealer_name'] != dealer_name:
                        continue
                    
                    # Check exact match with tolerances
                    rotor_diff = abs(float(row['rotor_speed_ms']) - rotor_speed)
                    ball_diff = abs(int(row['ball_speed_ms']) - ball_speed)
                    
                    if (rotor_diff <= ROTOR_TOLERANCE_MS and
                        ball_diff <= BALL_TOLERANCE_MS and
                        row['direction'] == direction):
                        
                        # Match found - add winning number
                        if row.get('winning_number'):
                            winning_numbers.append(int(row['winning_number']))
                
                if winning_numbers:
                    dealer_info = f" for dealer {dealer_name}" if USE_DEALER_FILTER else ""
                    logger.info(f"Found {len(winning_numbers)} matches{dealer_info}: "
                              f"rotor={rotor_speed}ms, ball={ball_speed}ms")
                
                return winning_numbers
                
        except Exception as e:
            logger.error(f"Error searching database: {e}")
            return []
    
    def store_pending(self, request: PredictionRequest):
        """Store round data temporarily until winning number arrives"""
        
        # Check if already exists
        if request.round_id in self.pending_rounds:
            logger.warning(f"Round {request.round_id} already exists in pending - ignoring duplicate")
            return
        
        # Store with precise timestamp
        self.pending_rounds[request.round_id] = {
            'timestamp': datetime.now().isoformat(),
            'dealer_name': request.dealer_name,
            'rotor_speed_ms': request.rotor_speed_ms,
            'ball_speed_ms': request.ball_speed_ms,
            'direction': request.direction,
            'table_id': request.table_id
        }
        logger.info(f"Stored pending round: {request.round_id} (dealer: {request.dealer_name})")
    
    def complete_round(self, round_id: str, winning_number: int) -> Dict[str, Any]:
        """Complete round with winning number and save to database"""
        
        # Check if round exists
        if round_id not in self.pending_rounds:
            logger.warning(f"Round {round_id} not found in pending rounds")
            return {
                "success": False,
                "error": "Round not found in pending"
            }
        
        selected_data = self.pending_rounds[round_id]
        
        # Prepare complete record
        record = {
            'timestamp': selected_data['timestamp'],
            'round_id': round_id,
            'dealer_name': selected_data['dealer_name'],
            'rotor_speed_ms': selected_data['rotor_speed_ms'],
            'ball_speed_ms': selected_data['ball_speed_ms'],
            'direction': selected_data['direction'],
            'winning_number': winning_number
        }
        
        # Save to CSV
        try:
            with open(DATABASE_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(record)
            
            self.total_records += 1
            
            # Delete only this round from pending
            del self.pending_rounds[round_id]
            
            logger.info(f"Completed round {round_id}: dealer={record['dealer_name']}, "
                       f"winning={winning_number}, total_records={self.total_records}")
            
            return {
                "success": True,
                "total_records": self.total_records
            }
            
        except Exception as e:
            logger.error(f"Error saving record: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_prediction(self, request: PredictionRequest) -> Optional[Dict[str, Any]]:
        """Get prediction based on historical matches"""
        
        # First, store the pending round
        self.store_pending(request)
        
        # Find all matching winning numbers
        winning_numbers = self.find_matches(
            request.dealer_name,
            request.rotor_speed_ms,
            request.ball_speed_ms,
            request.direction
        )
        
        if not winning_numbers:
            return None
        
        # Count occurrences of each number
        number_counts = Counter(winning_numbers)
        
        # Get most common number and its count
        most_common_number, count = number_counts.most_common(1)[0]
        
        # Calculate confidence as percentage of total matches
        confidence = count / len(winning_numbers)
        
        logger.info(f"Prediction for round {request.round_id}: "
                   f"number={most_common_number} (confidence={confidence:.2f}, "
                   f"matches={len(winning_numbers)})")
        
        return {
            "predicted_number": most_common_number,
            "confidence": round(confidence, 2),
            "matches_found": len(winning_numbers)
        }

# ============================================================================
# API SERVER
# ============================================================================

app = FastAPI(
    title="Evolution Roulette Prediction Server",
    description="Dealer-based prediction system with exact matching",
    version="3.0.0"
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
        "server": "Evolution Roulette Prediction Server",
        "version": "3.0.0",
        "status": "operational",
        "method": "dealer_based_exact_matching",
        "statistics": {
            "total_records": db.total_records,
            "pending_rounds": len(db.pending_rounds)
        },
        "settings": {
            "use_dealer_filter": USE_DEALER_FILTER,
            "rotor_tolerance_ms": ROTOR_TOLERANCE_MS,
            "ball_tolerance_ms": BALL_TOLERANCE_MS
        }
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Get prediction based on dealer and speed measurements"""
    
    try:
        # Get prediction using dealer-based matching
        result = db.get_prediction(request)
        
        if result:
            # Found matches - return prediction
            return {
                "predicted_number": result["predicted_number"],
                "confidence": result["confidence"],
                "matches_found": result["matches_found"],
                "dataset_rows": db.total_records,
                "method": "dealer_based"
            }
        else:
            # No matches found
            dealer_info = f" for dealer {request.dealer_name}" if USE_DEALER_FILTER else ""
            message = f"No matches found{dealer_info}"
            
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
                "dataset_rows": result["total_records"]
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
    
    # Count patterns by dealer
    dealer_stats = {}
    
    try:
        with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                dealer = row['dealer_name']
                if dealer not in dealer_stats:
                    dealer_stats[dealer] = {
                        "total_spins": 0,
                        "unique_patterns": set()
                    }
                
                dealer_stats[dealer]["total_spins"] += 1
                
                # Create pattern key
                pattern = f"{row['rotor_speed_ms']}_{row['ball_speed_ms']}_{row['direction']}"
                dealer_stats[dealer]["unique_patterns"].add(pattern)
        
        # Convert to list format
        dealer_list = []
        for dealer, data in dealer_stats.items():
            dealer_list.append({
                "dealer": dealer,
                "total_spins": data["total_spins"],
                "unique_patterns": len(data["unique_patterns"])
            })
        
        # Sort by total spins
        dealer_list.sort(key=lambda x: x["total_spins"], reverse=True)
        
    except:
        dealer_list = []
    
    return {
        "total_records": db.total_records,
        "pending_rounds": len(db.pending_rounds),
        "dealers": dealer_list,
        "settings": {
            "use_dealer_filter": USE_DEALER_FILTER,
            "rotor_tolerance_ms": ROTOR_TOLERANCE_MS,
            "ball_tolerance_ms": BALL_TOLERANCE_MS
        }
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
    print("EVOLUTION ROULETTE PREDICTION SERVER v3.0.0")
    print("Dealer-Based Exact Matching System")
    print("="*60)
    print(f"Database: {DATABASE_FILE}")
    print(f"Records: {db.total_records}")
    print("="*60)
    print("\nSettings:")
    print(f"- Dealer Filter: {'ENABLED' if USE_DEALER_FILTER else 'DISABLED'}")
    print(f"- Rotor Tolerance: ±{ROTOR_TOLERANCE_MS}ms")
    print(f"- Ball Tolerance: ±{BALL_TOLERANCE_MS}ms")
    print("="*60)
    print("\nHow it works:")
    print("1. Collects: dealer_name, rotor_speed_ms, ball_speed_ms, direction")
    print("2. Searches: exact matches in database (0 tolerance)")
    print("3. Returns: most frequent winning number with confidence")
    print("4. Each dealer has isolated data (no cross-contamination)")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
