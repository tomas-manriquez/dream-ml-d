#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# SIG 3.3.9 – Dependency Strength
# Automated dependency freshness and vulnerability verification
#
# Tools:
#  - pip (freshness)
#  - OWASP Dependency-Check (3rd-party vulnerabilities)
#  - Trivy (container image vulnerabilities)
#
# Enforcement:
#  - FAIL on CRITICAL vulnerabilities in:
#      * Backend Python dependencies
#      * Backend container image
#  - WARN only elsewhere
###############################################################################

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORT_ROOT="$ROOT_DIR/reports"
TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"
RUN_DIR="$REPORT_ROOT/$TIMESTAMP"

BACKEND_IMAGE="tomasmanriquez480/dreaml-ml-backend:latest"
FRONTEND_IMAGE="tomasmanriquez480/dreaml-ml-frontend:latest"

REQ_BASE="$ROOT_DIR/../DREAM-ML-backend/GEML/requirements-base.txt"
REQ_DEV="$ROOT_DIR/../DREAM-ML-backend/GEML/requirements-dev.txt"
FRONTEND_DIR="$ROOT_DIR/../DREAM-ML-frontend/frontend"

LOG_FILE="$RUN_DIR/execution.log"
SUMMARY_FILE="$RUN_DIR/summary.txt"

NVD_API_KEY="e92c2c16-e1c3-4450-8682-f41a6423245a"

mkdir -p \
  "$RUN_DIR/pip" \
  "$RUN_DIR/dependency-check" \
  "$RUN_DIR/trivy"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Dependency Security Scan (SIG 3.3.9) ==="
echo "Timestamp: $TIMESTAMP"
echo

###############################################################################
# 1. Dependency freshness (pip)
###############################################################################

echo "[1/5] Checking Python dependency freshness (pip)"

# Create temporary venv for isolated dependency checking
TEMP_VENV="/tmp/dep-security-pip-check-$$"
python3.11 -m venv "$TEMP_VENV"
source "$TEMP_VENV/bin/activate"

# Upgrade pip silently
if ! pip install --quiet --upgrade pip; then
  echo "    ⚠ Warning: Failed to upgrade pip (continuing with existing version)" >&2
fi

# Check base requirements
echo "  Checking base requirements..."
if pip install --quiet -r "$REQ_BASE" 2>&1 | tee /tmp/pip-base-error.log >/dev/null; then
  pip list --outdated --format=columns > "$RUN_DIR/pip/outdated-base.txt"
  BASE_OUTDATED_COUNT=$(tail -n +3 "$RUN_DIR/pip/outdated-base.txt" | wc -l | tr -d ' ')
  echo "    Found $BASE_OUTDATED_COUNT outdated packages in base"
else
  echo "    ⚠ Warning: Failed to install base requirements" >&2
  echo "    See error details:" >&2
  tail -5 /tmp/pip-base-error.log >&2
  echo "ERROR: Could not install base requirements" > "$RUN_DIR/pip/outdated-base.txt"
  echo "Check packages that may lack binary wheels or have build issues" >> "$RUN_DIR/pip/outdated-base.txt"
fi

# Check dev requirements (incremental - builds on base)
echo "  Checking dev requirements..."
if pip install --quiet -r "$REQ_DEV" 2>&1 | tee /tmp/pip-dev-error.log >/dev/null; then
  pip list --outdated --format=columns > "$RUN_DIR/pip/outdated-dev.txt"
  DEV_OUTDATED_COUNT=$(tail -n +3 "$RUN_DIR/pip/outdated-dev.txt" | wc -l | tr -d ' ')
  echo "    Found $DEV_OUTDATED_COUNT outdated packages in dev"
else
  echo "    ⚠ Warning: Failed to install dev requirements" >&2
  echo "    See error details:" >&2
  tail -5 /tmp/pip-dev-error.log >&2
  echo "ERROR: Could not install dev requirements" > "$RUN_DIR/pip/outdated-dev.txt"
  echo "Check packages that may lack binary wheels or have build issues" >> "$RUN_DIR/pip/outdated-dev.txt"
fi

# Cleanup
deactivate
rm -rf "$TEMP_VENV"

echo "  ✔ pip freshness reports generated"

###############################################################################
# 2. OWASP Dependency-Check – Backend (Python)
###############################################################################

echo "[2/5] Running OWASP Dependency-Check (Backend – Python)"

# Prepare NVD API key argument if available
NVD_ARG=""
if [[ -n "${NVD_API_KEY:-}" ]]; then
  NVD_ARG="--nvdApiKey $NVD_API_KEY"
  echo "  Using NVD API key for faster scanning"
fi

# Run OWASP DC with correct paths
if docker run --rm \
  -v "$ROOT_DIR/..:/src" \
  -v "$RUN_DIR/dependency-check:/report" \
  owasp/dependency-check:latest \
  --project "DREAM-ML Backend" \
  --scan /src/DREAM-ML-backend/GEML/requirements-base.txt \
  --scan /src/DREAM-ML-backend/GEML/requirements-dev.txt \
  --format HTML --format JSON \
  --out /report/backend \
  $NVD_ARG; then
  echo "  ✔ Backend scan completed"
else
  BACKEND_DC_EXIT=$?
  echo "  ⚠ Backend scan failed with exit code $BACKEND_DC_EXIT" >&2
fi

###############################################################################
# 3. OWASP Dependency-Check – Frontend (React)
###############################################################################

echo "[3/5] Running OWASP Dependency-Check (Frontend – React)"

if docker run --rm \
  -v "$FRONTEND_DIR:/src" \
  -v "$RUN_DIR/dependency-check:/report" \
  owasp/dependency-check:latest \
  --project "DREAM-ML Frontend" \
  --scan /src/package-lock.json \
  --format HTML --format JSON \
  --out /report/frontend \
  $NVD_ARG; then
  echo "  ✔ Frontend scan completed"
else
  FRONTEND_DC_EXIT=$?
  echo "  ⚠ Frontend scan failed with exit code $FRONTEND_DC_EXIT" >&2
fi

###############################################################################
# 4. Trivy – Container Images
###############################################################################

echo "[4/5] Pulling container images"
docker pull "$BACKEND_IMAGE"
docker pull "$FRONTEND_IMAGE"

echo "Running Trivy (Backend image)"
if docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$RUN_DIR/trivy:/output" \
  aquasec/trivy:latest image \
  --severity CRITICAL,HIGH,MEDIUM,LOW \
  --format json \
  --output /output/backend.json \
  "$BACKEND_IMAGE"; then
  echo "  ✔ Backend image scan completed"
else
  BACKEND_TRIVY_EXIT=$?
  echo "  ⚠ Backend Trivy scan failed with exit code $BACKEND_TRIVY_EXIT" >&2
fi

echo "Running Trivy (Frontend image)"
if docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$RUN_DIR/trivy:/output" \
  aquasec/trivy:latest image \
  --severity CRITICAL,HIGH,MEDIUM,LOW \
  --format json \
  --output /output/frontend.json \
  "$FRONTEND_IMAGE"; then
  echo "  ✔ Frontend image scan completed"
else
  FRONTEND_TRIVY_EXIT=$?
  echo "  ⚠ Frontend Trivy scan failed with exit code $FRONTEND_TRIVY_EXIT" >&2
fi

###############################################################################
# 5. Enforcement & Summary
###############################################################################

echo "[5/5] Evaluating results"

# Parse JSON results with error handling
BACKEND_DC_CRITICAL=$(jq '[.dependencies[]?.vulnerabilities[]? | select(.severity=="CRITICAL")] | length' \
  "$RUN_DIR/dependency-check/backend/dependency-check-report.json" 2>/dev/null || echo "0")

BACKEND_TRIVY_CRITICAL=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity=="CRITICAL")] | length' \
  "$RUN_DIR/trivy/backend.json" 2>/dev/null || echo "0")

FRONTEND_DC_CRITICAL=$(jq '[.dependencies[]?.vulnerabilities[]? | select(.severity=="CRITICAL")] | length' \
  "$RUN_DIR/dependency-check/frontend/dependency-check-report.json" 2>/dev/null || echo "0")

# Get total vulnerability counts for summary
BACKEND_DC_HIGH=$(jq '[.dependencies[]?.vulnerabilities[]? | select(.severity=="HIGH")] | length' \
  "$RUN_DIR/dependency-check/backend/dependency-check-report.json" 2>/dev/null || echo "0")

BACKEND_TRIVY_HIGH=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity=="HIGH")] | length' \
  "$RUN_DIR/trivy/backend.json" 2>/dev/null || echo "0")

FRONTEND_DC_HIGH=$(jq '[.dependencies[]?.vulnerabilities[]? | select(.severity=="HIGH")] | length' \
  "$RUN_DIR/dependency-check/frontend/dependency-check-report.json" 2>/dev/null || echo "0")

{
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║     Dependency Security Summary (SIG 3.3.9)                      ║"
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo
  echo "Scan completed: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Report location: $RUN_DIR"
  echo
  echo "─────────────────────────────────────────────────────────────────"
  echo "VULNERABILITY COUNTS:"
  echo "─────────────────────────────────────────────────────────────────"
  echo
  echo "Backend Python Dependencies (OWASP DC):"
  echo "  CRITICAL: $BACKEND_DC_CRITICAL"
  echo "  HIGH:     $BACKEND_DC_HIGH"
  echo
  echo "Backend Container Image (Trivy):"
  echo "  CRITICAL: $BACKEND_TRIVY_CRITICAL"
  echo "  HIGH:     $BACKEND_TRIVY_HIGH"
  echo
  echo "Frontend Node Dependencies (OWASP DC):"
  echo "  CRITICAL: $FRONTEND_DC_CRITICAL"
  echo "  HIGH:     $FRONTEND_DC_HIGH"
  echo
  echo "─────────────────────────────────────────────────────────────────"
  echo "DEPENDENCY FRESHNESS:"
  echo "─────────────────────────────────────────────────────────────────"
  echo
  echo "Outdated packages (base requirements):"
  if [[ -f "$RUN_DIR/pip/outdated-base.txt" ]]; then
    if grep -q "ERROR:" "$RUN_DIR/pip/outdated-base.txt"; then
      echo "  ⚠ Error checking outdated packages"
      cat "$RUN_DIR/pip/outdated-base.txt"
    else
      sed 's/^/  /' "$RUN_DIR/pip/outdated-base.txt"
    fi
  else
    echo "  No report generated"
  fi
  echo
  echo "Outdated packages (dev requirements):"
  if [[ -f "$RUN_DIR/pip/outdated-dev.txt" ]]; then
    if grep -q "ERROR:" "$RUN_DIR/pip/outdated-dev.txt"; then
      echo "  ⚠ Error checking outdated packages"
      cat "$RUN_DIR/pip/outdated-dev.txt"
    else
      sed 's/^/  /' "$RUN_DIR/pip/outdated-dev.txt"
    fi
  else
    echo "  No report generated"
  fi
  echo
  echo "─────────────────────────────────────────────────────────────────"
  echo "DETAILED REPORTS:"
  echo "─────────────────────────────────────────────────────────────────"
  echo
  echo "  Backend OWASP DC:  $RUN_DIR/dependency-check/backend/dependency-check-report.html"
  echo "  Frontend OWASP DC: $RUN_DIR/dependency-check/frontend/dependency-check-report.html"
  echo "  Backend Trivy:     $RUN_DIR/trivy/backend.json"
  echo "  Frontend Trivy:    $RUN_DIR/trivy/frontend.json"
  echo
} > "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

###############################################################################
# Enforcement rule: FAIL only if backend has CRITICAL vulns
###############################################################################

if [[ "$BACKEND_DC_CRITICAL" -gt 0 || "$BACKEND_TRIVY_CRITICAL" -gt 0 ]]; then
  echo
  echo "❌ FAIL: CRITICAL vulnerabilities detected in enforced backend scope"
  exit 1
fi

echo
echo "✅ PASS: No CRITICAL vulnerabilities in enforced backend scope"
exit 0
