#!/bin/bash
set -euo pipefail

# Configuration
REPO="https://github.com/Nikoo4/543vvt.git"
APP_DIR="/opt/543vvt"
PORT="8000"

# Stop existing service
systemctl stop roulette.service 2>/dev/null || true

# Wait for apt locks
while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
   sleep 2
done

# Install dependencies
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git python3-venv python3-pip

# Clone repository
rm -rf "$APP_DIR"
git clone --depth 1 "$REPO" "$APP_DIR"

cd "$APP_DIR"

# Setup Python environment
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# Create data directory
mkdir -p "$APP_DIR/data"

# Set permissions
chown -R www-data:www-data "$APP_DIR"

# Create systemd service
cp roulette.service /etc/systemd/system/

# Start service
systemctl daemon-reload
systemctl enable roulette.service
systemctl restart roulette.service

# Display status
sleep 3
systemctl status roulette.service --no-pager

IP=$(hostname -I | awk '{print $1}')
echo "Server running at: http://$IP:$PORT/"
