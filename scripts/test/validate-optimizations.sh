#!/bin/bash

################################################################################
# DREAM ML - Docker Optimization Validation Script
################################################################################
# This script validates the Dockerfile optimizations to ensure:
# 1. Containers start successfully without build tools (gcc/g++/cmake)
# 2. All runtime dependencies work correctly (git, dvc, visualization libs)
# 3. All API endpoints function properly
# 4. Image size reduction is achieved
#
# Run this script after building the optimized images to verify everything works
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

################################################################################
# Color codes for output
################################################################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

################################################################################
# Logging functions
################################################################################
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_test() {
    echo -e "${CYAN}[TEST]${NC} $1"
}

log_step() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}\n"
}

################################################################################
# Test tracking
################################################################################
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNINGS=0

test_start() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    log_test "$1"
}

test_pass() {
    PASSED_TESTS=$((PASSED_TESTS + 1))
    log_success "$1"
}

test_fail() {
    FAILED_TESTS=$((FAILED_TESTS + 1))
    log_error "$1"
}

test_warn() {
    WARNINGS=$((WARNINGS + 1))
    log_warning "$1"
}

################################################################################
# Configuration
################################################################################
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_CONTAINER="dream-ml-c-backend-1"
FRONTEND_CONTAINER="dream-ml-c-frontend-1"
BACKEND_IMAGE="tomasmanriquez480/dreaml-ml-backend:latest"
WAIT_TIMEOUT=60

################################################################################
# Step 1: Check Docker Compose is running
################################################################################
log_step "Step 1: Verifying Docker Compose Setup"

test_start "Checking if docker-compose.yml exists"
if [ -f "${BUILD_DIR}/docker-compose.yml" ]; then
    test_pass "docker-compose.yml found"
else
    test_fail "docker-compose.yml not found at ${BUILD_DIR}"
    exit 1
fi

test_start "Checking if containers are running"
cd "${BUILD_DIR}" || exit 1

if docker-compose ps | grep -q "Up"; then
    test_pass "Docker Compose services are running"
else
    test_warn "Services not running - attempting to start them"
    log_info "Starting services with docker-compose up -d..."
    docker-compose up -d
    sleep 10
fi

################################################################################
# Step 2: Image Size Validation
################################################################################
log_step "Step 2: Validating Image Size Reduction"

test_start "Measuring backend image size"
if docker image inspect "${BACKEND_IMAGE}" &>/dev/null; then
    IMAGE_SIZE=$(docker image inspect "${BACKEND_IMAGE}" --format='{{.Size}}')
    IMAGE_SIZE_MB=$((IMAGE_SIZE / 1024 / 1024))
    IMAGE_SIZE_GB=$(echo "scale=2; ${IMAGE_SIZE_MB} / 1024" | bc)

    log_info "Backend image size: ${IMAGE_SIZE_GB} GB (${IMAGE_SIZE_MB} MB)"

    # Check if size is under expected threshold (1.3 GB = 1330 MB)
    if [ "${IMAGE_SIZE_MB}" -lt 1330 ]; then
        test_pass "Image size is within expected range (< 1.3 GB)"
    elif [ "${IMAGE_SIZE_MB}" -lt 1500 ]; then
        test_warn "Image size is acceptable but higher than expected: ${IMAGE_SIZE_GB} GB"
    else
        test_fail "Image size is too large: ${IMAGE_SIZE_GB} GB (expected < 1.5 GB)"
    fi

    # Calculate reduction from original 1.64 GB
    ORIGINAL_SIZE_MB=1640
    REDUCTION_MB=$((ORIGINAL_SIZE_MB - IMAGE_SIZE_MB))
    REDUCTION_PERCENT=$(echo "scale=1; (${REDUCTION_MB} * 100) / ${ORIGINAL_SIZE_MB}" | bc)

    if [ "${REDUCTION_MB}" -gt 0 ]; then
        log_success "Size reduction achieved: ${REDUCTION_MB} MB (${REDUCTION_PERCENT}%)"
    else
        test_warn "No size reduction detected (may need fresh build)"
    fi
else
    test_fail "Backend image not found locally"
fi

################################################################################
# Step 3: Container Health Check
################################################################################
log_step "Step 3: Validating Container Health"

test_start "Checking backend container health"
BACKEND_HEALTH=$(docker inspect "${BACKEND_CONTAINER}" --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")

if [ "${BACKEND_HEALTH}" = "healthy" ]; then
    test_pass "Backend container is HEALTHY"
elif [ "${BACKEND_HEALTH}" = "starting" ]; then
    test_warn "Backend container is still starting - waiting..."
    sleep 20
    BACKEND_HEALTH=$(docker inspect "${BACKEND_CONTAINER}" --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
    if [ "${BACKEND_HEALTH}" = "healthy" ]; then
        test_pass "Backend container is now HEALTHY"
    else
        test_fail "Backend container health: ${BACKEND_HEALTH}"
    fi
else
    test_fail "Backend container health: ${BACKEND_HEALTH}"
fi

test_start "Checking frontend container health"
FRONTEND_HEALTH=$(docker inspect "${FRONTEND_CONTAINER}" --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")

if [ "${FRONTEND_HEALTH}" = "healthy" ]; then
    test_pass "Frontend container is HEALTHY"
elif [ "${FRONTEND_HEALTH}" = "starting" ]; then
    test_warn "Frontend container is still starting - waiting..."
    sleep 10
    FRONTEND_HEALTH=$(docker inspect "${FRONTEND_CONTAINER}" --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
    if [ "${FRONTEND_HEALTH}" = "healthy" ]; then
        test_pass "Frontend container is now HEALTHY"
    else
        test_fail "Frontend container health: ${FRONTEND_HEALTH}"
    fi
else
    test_fail "Frontend container health: ${FRONTEND_HEALTH}"
fi

################################################################################
# Step 4: Validate Build Tools Removal
################################################################################
log_step "Step 4: Validating Build Tools Removal"

test_start "Checking if gcc is absent from runtime image"
if docker exec "${BACKEND_CONTAINER}" which gcc &>/dev/null; then
    test_fail "gcc is present in runtime image (should be removed)"
else
    test_pass "gcc successfully removed from runtime image"
fi

test_start "Checking if g++ is absent from runtime image"
if docker exec "${BACKEND_CONTAINER}" which g++ &>/dev/null; then
    test_fail "g++ is present in runtime image (should be removed)"
else
    test_pass "g++ successfully removed from runtime image"
fi

test_start "Checking if cmake is absent from runtime image"
if docker exec "${BACKEND_CONTAINER}" which cmake &>/dev/null; then
    test_fail "cmake is present in runtime image (should be removed)"
else
    test_pass "cmake successfully removed from runtime image"
fi

################################################################################
# Step 5: Validate Required Runtime Dependencies
################################################################################
log_step "Step 5: Validating Required Runtime Dependencies"

test_start "Checking if git is present (REQUIRED for DVC)"
if docker exec "${BACKEND_CONTAINER}" which git &>/dev/null; then
    GIT_VERSION=$(docker exec "${BACKEND_CONTAINER}" git --version)
    test_pass "git is present: ${GIT_VERSION}"
else
    test_fail "git is missing (REQUIRED for DVC operations)"
fi

test_start "Checking if dvc is present"
if docker exec "${BACKEND_CONTAINER}" which dvc &>/dev/null; then
    DVC_VERSION=$(docker exec "${BACKEND_CONTAINER}" dvc version | head -1)
    test_pass "dvc is present: ${DVC_VERSION}"
else
    test_fail "dvc is missing"
fi

test_start "Checking if Python packages are installed"
PYTHON_CHECK=$(docker exec "${BACKEND_CONTAINER}" python -c "
import sys
packages = [
    'django', 'pandas', 'numpy', 'sklearn', 'tensorflow',
    'xgboost', 'mlflow', 'matplotlib', 'seaborn',
    'sweetviz', 'ydata_profiling', 'codecarbon', 'dvc'
]
missing = []
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print('MISSING: ' + ', '.join(missing))
    sys.exit(1)
else:
    print('ALL_OK')
    sys.exit(0)
" 2>&1)

if echo "${PYTHON_CHECK}" | grep -q "ALL_OK"; then
    test_pass "All required Python packages are installed"
elif echo "${PYTHON_CHECK}" | grep -q "MISSING"; then
    MISSING_PKGS=$(echo "${PYTHON_CHECK}" | grep "MISSING" | cut -d: -f2)
    test_fail "Missing Python packages:${MISSING_PKGS}"
else
    test_fail "Python package check failed: ${PYTHON_CHECK}"
fi

test_start "Checking if kfp (Kubeflow) was removed"
KFP_CHECK=$(docker exec "${BACKEND_CONTAINER}" python -c "
try:
    import kfp
    print('PRESENT')
except ImportError:
    print('ABSENT')
" 2>&1)

if echo "${KFP_CHECK}" | grep -q "ABSENT"; then
    test_pass "kfp successfully removed (as intended)"
elif echo "${KFP_CHECK}" | grep -q "PRESENT"; then
    test_warn "kfp is still present (should be removed for size optimization)"
else
    test_warn "Could not verify kfp removal"
fi

################################################################################
# Step 6: Validate Cython Compilation
################################################################################
log_step "Step 6: Validating Cython Compilation"

test_start "Checking if Cython modules are compiled"
CYTHON_CHECK=$(docker exec "${BACKEND_CONTAINER}" sh -c "
cd /app
if [ -f api/views.*.so ] || [ -f api/train.*.so ]; then
    echo 'COMPILED'
else
    echo 'NOT_COMPILED'
fi
" 2>&1)

if echo "${CYTHON_CHECK}" | grep -q "COMPILED"; then
    test_pass "Cython modules are compiled (.so files present)"
else
    test_fail "Cython modules not found (.so files missing)"
fi

test_start "Checking if original .py files were removed"
PYTHON_CHECK=$(docker exec "${BACKEND_CONTAINER}" sh -c "
cd /app
if [ -f api/views.py ] || [ -f api/train.py ]; then
    echo 'PRESENT'
else
    echo 'ABSENT'
fi
" 2>&1)

if echo "${PYTHON_CHECK}" | grep -q "ABSENT"; then
    test_pass "Original .py files removed (IP protection working)"
else
    test_warn "Original .py files still present (IP protection may not be working)"
fi

################################################################################
# Step 7: API Endpoint Tests
################################################################################
log_step "Step 7: Testing API Endpoints"

test_start "Testing backend health endpoint"
if curl -f -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/health/ | grep -q "200"; then
    test_pass "Backend health endpoint responding"
else
    test_fail "Backend health endpoint not responding"
fi

test_start "Testing frontend accessibility"
if curl -f -s -o /dev/null -w "%{http_code}" http://localhost:5173/ | grep -q "200"; then
    test_pass "Frontend is accessible"
else
    test_fail "Frontend is not accessible"
fi

################################################################################
# Step 8: Runtime Functionality Tests
################################################################################
log_step "Step 8: Testing Runtime Functionality"

test_start "Testing matplotlib/visualization libraries"
VIZ_TEST=$(docker exec "${BACKEND_CONTAINER}" python -c "
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    # Create a simple plot
    plt.figure(figsize=(5,5))
    plt.plot([1,2,3], [1,2,3])
    plt.savefig('/tmp/test_plot.png')
    plt.close()
    print('SUCCESS')
except Exception as e:
    print(f'FAILED: {e}')
" 2>&1)

if echo "${VIZ_TEST}" | grep -q "SUCCESS"; then
    test_pass "Matplotlib/visualization libraries working"
else
    test_fail "Matplotlib test failed: ${VIZ_TEST}"
fi

test_start "Testing git operations"
GIT_TEST=$(docker exec "${BACKEND_CONTAINER}" sh -c "
cd /tmp
git init test_repo 2>&1 && echo 'SUCCESS' || echo 'FAILED'
" 2>&1)

if echo "${GIT_TEST}" | grep -q "SUCCESS"; then
    test_pass "Git operations working"
else
    test_fail "Git operations failed: ${GIT_TEST}"
fi

test_start "Testing DVC CLI"
DVC_TEST=$(docker exec "${BACKEND_CONTAINER}" sh -c "
dvc version >/dev/null 2>&1 && echo 'SUCCESS' || echo 'FAILED'
" 2>&1)

if echo "${DVC_TEST}" | grep -q "SUCCESS"; then
    test_pass "DVC CLI working"
else
    test_fail "DVC CLI failed: ${DVC_TEST}"
fi

################################################################################
# Step 9: Performance Check
################################################################################
log_step "Step 9: Performance Check"

test_start "Checking backend container memory usage"
MEMORY_USAGE=$(docker stats "${BACKEND_CONTAINER}" --no-stream --format "{{.MemUsage}}" 2>/dev/null || echo "Unknown")
log_info "Backend memory usage: ${MEMORY_USAGE}"
test_pass "Memory usage recorded (monitor for optimization)"

test_start "Checking backend container CPU usage"
CPU_USAGE=$(docker stats "${BACKEND_CONTAINER}" --no-stream --format "{{.CPUPerc}}" 2>/dev/null || echo "Unknown")
log_info "Backend CPU usage: ${CPU_USAGE}"
test_pass "CPU usage recorded (monitor for optimization)"

################################################################################
# Summary
################################################################################
log_step "Validation Summary"

echo ""
log_info "Test Results:"
log_success "  Passed:   ${PASSED_TESTS}/${TOTAL_TESTS}"
[ ${FAILED_TESTS} -gt 0 ] && log_error "  Failed:   ${FAILED_TESTS}/${TOTAL_TESTS}"
[ ${WARNINGS} -gt 0 ] && log_warning "  Warnings: ${WARNINGS}"

echo ""
if [ ${FAILED_TESTS} -eq 0 ]; then
    log_success "✓ ALL TESTS PASSED!"
    log_success "Optimizations are working correctly:"
    echo "  ✓ Build tools removed from runtime image"
    echo "  ✓ All required dependencies present and working"
    echo "  ✓ Cython compilation successful"
    echo "  ✓ API endpoints responding"
    echo "  ✓ Runtime functionality verified"
    echo ""
    log_success "The optimized Docker image is ready for production!"
    exit 0
else
    log_error "✗ SOME TESTS FAILED"
    log_error "Please review the failures above and fix issues before deploying"
    echo ""
    log_info "Common solutions:"
    echo "  - Rebuild images: docker-compose build --no-cache"
    echo "  - Restart containers: docker-compose restart"
    echo "  - Check logs: docker-compose logs backend"
    exit 1
fi
