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

            ⚠️ Installation Script

bash <<'INSTALL_543VVT'
set -euo pipefail

# Configuration variables
REPO="https://github.com/Nikoo4/543vvt.git"
APP_DIR="/opt/543vvt"
PORT="8000"

# Stop any existing service
systemctl stop roulette.service 2>/dev/null || true

# Wait for apt locks to be released
while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
   sleep 2
done

# Install required system packages
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git python3-venv python3-pip

# Clone repository
rm -rf "$APP_DIR"
git clone --depth 1 "$REPO" "$APP_DIR"

# Create Python virtual environment
cd "$APP_DIR"
python3 -m venv venv

# Install Python packages
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# Create data directory
mkdir -p "$APP_DIR/roulette_v17"

# Set permissions for www-data user
chown -R www-data:www-data "$APP_DIR"

# Fix the systemd service file paths
sed -i "s|^WorkingDirectory=.*|WorkingDirectory=$APP_DIR|" "$APP_DIR/roulette.service"
sed -i "s|^ExecStart=.*|ExecStart=$APP_DIR/venv/bin/uvicorn server:app --host 0.0.0.0 --port $PORT|" "$APP_DIR/roulette.service"

# Copy service file to systemd
cp "$APP_DIR/roulette.service" /etc/systemd/system/

# Reload systemd and start service
systemctl daemon-reload
systemctl enable roulette.service
systemctl restart roulette.service

# Display status
sleep 2
systemctl status roulette.service --no-pager

# Show server URLs
IP=$(hostname -I | awk '{print $1}')
echo ""
echo "Server running at: http://$IP:$PORT/"

INSTALL_543VVT
