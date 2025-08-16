# Roulette Prediction Server v17

FastAPI server for roulette prediction based on ball speed and traveled pockets pattern matching.

## Method
- Tracks ball speed (ms) between timestamp1 and timestamp2
- Counts traveled pockets during one rotation
- Matches historical patterns to predict outcome
- Self-learning system with automatic optimization

## Features
- ✅ Pattern matching with ±50ms speed tolerance
- ✅ Intelligent learning control (stops at ≤4 pockets error)
- ✅ Poor quality data filtering
- ✅ Real-time performance tracking
- ✅ Dataset size management (max 5000 records)
