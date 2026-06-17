#!/bin/bash

################################################################################
# DREAM ML - Multi-Platform Docker Build & Push Script
################################################################################
# This script builds Docker images for multiple platforms and pushes them to
# Docker Hub registry: tomasmanriquez480/dreaml-ml
#
# Supported platforms:
#   - linux/amd64 (Windows, Linux x86_64)
#   - linux/arm64 (macOS Apple Silicon, Linux ARM)
#
# Requirements:
#   - Docker with buildx support
#   - Authenticated to Docker Hub (docker login)
#   - Running on macOS arm64
#
# Usage:
#   ./build-and-push.sh
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

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

log_step() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}\n"
}

################################################################################
# Error handler
################################################################################
error_exit() {
    log_error "$1"
    log_error "Build process failed. Exiting..."
    exit 1
}

# Trap errors
trap 'error_exit "An unexpected error occurred at line $LINENO"' ERR

################################################################################
# Configuration
################################################################################
DOCKER_REGISTRY="tomasmanriquez480/dreaml-ml"
BACKEND_IMAGE="${DOCKER_REGISTRY}-backend"
FRONTEND_IMAGE="${DOCKER_REGISTRY}-frontend"
PLATFORMS="linux/amd64,linux/arm64"
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_DIR="${BUILD_DIR}/dist"
ZIP_NAME="dream-ml.zip"
NO_CACHE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            log_info "Build cache disabled"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Usage: $0 [--no-cache]"
            exit 1
            ;;
    esac
done

################################################################################
# Step 0: Check Docker Disk Space
################################################################################
log_step "Step 0: Checking Docker Disk Space"

log_info "Analyzing Docker disk usage..."
DOCKER_DISK_OUTPUT=$(docker system df --format "table {{.Type}}\t{{.Size}}\t{{.Reclaimable}}")
echo "$DOCKER_DISK_OUTPUT"

# Get total reclaimable space (in human-readable format)
RECLAIMABLE=$(docker system df --format "{{.Reclaimable}}" | grep -E "^[0-9.]+[KMGT]?B" | head -1 || echo "0B")

log_info "Reclaimable space: ${RECLAIMABLE}"

# Check Docker Desktop disk allocation on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    log_info "Checking Docker Desktop disk allocation..."
    # Get Docker disk image size if available
    DOCKER_DISK_SIZE=$(docker info 2>/dev/null | grep "Docker Root Dir" || echo "Unable to determine")
    log_info "$DOCKER_DISK_SIZE"
fi

# Warn if low on space
log_warning "Multi-platform builds require significant disk space (~20-40GB during build)"
log_info "If build fails due to disk space, run: docker system prune -a"
read -p "Continue with build? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    error_exit "Build cancelled by user"
fi

log_success "Disk space check completed"

################################################################################
# Step 1: Verify Docker Hub Authentication
################################################################################
log_step "Step 1: Verifying Docker Hub Authentication"

log_info "Checking if logged in to Docker Hub..."
if ! docker info 2>/dev/null | grep "Username"; then
    error_exit "Not logged in to Docker Hub. Please run 'docker login' first."
fi

DOCKER_USERNAME=$(docker info 2>/dev/null | grep "Username:" | awk '{print $2}')
log_success "Logged in as: ${DOCKER_USERNAME}"

# Verify username matches expected registry
if [[ "${DOCKER_USERNAME}" != "tomasmanriquez480" ]]; then
    log_warning "Logged in as '${DOCKER_USERNAME}' but registry is 'tomasmanriquez480'"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        error_exit "Build cancelled by user"
    fi
fi

################################################################################
# Step 2: Get Version Number
################################################################################
log_step "Step 2: Version Configuration"

log_info "Please enter the semantic version for this build (e.g., 1.0.0):"
read -p "Version: " VERSION

# Validate version format
if [[ ! $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    error_exit "Invalid version format. Please use semantic versioning (e.g., 1.0.0)"
fi

VERSION_TAG="v${VERSION}"
log_success "Building version: ${VERSION_TAG}"

################################################################################
# Step 3: Verify Docker Buildx
################################################################################
log_step "Step 3: Verifying Docker Buildx Support"

log_info "Checking if docker buildx is available..."
if ! docker buildx version &>/dev/null; then
    error_exit "Docker buildx is not available. Please update Docker to a version with buildx support."
fi
log_success "Docker buildx is available"

log_info "Checking for existing builder instance..."
BUILDER_NAME="dream-ml-builder"

if docker buildx inspect "${BUILDER_NAME}" &>/dev/null; then
    log_info "Using existing builder: ${BUILDER_NAME}"
    docker buildx use "${BUILDER_NAME}"
else
    log_info "Creating new builder instance: ${BUILDER_NAME}"
    docker buildx create --name "${BUILDER_NAME}" --driver docker-container --bootstrap --use
    log_success "Builder created: ${BUILDER_NAME}"
fi

log_info "Inspecting builder..."
docker buildx inspect --bootstrap

################################################################################
# Step 4: Build Backend Image (Sequential Multi-Platform)
################################################################################
log_step "Step 4: Building Backend Image (Sequential Multi-Platform)"

log_info "Building backend sequentially for platforms: ${PLATFORMS}"
log_info "This approach uses less disk space and is more reliable"
log_info "Tagged images:"
log_info "  - ${BACKEND_IMAGE}:${VERSION_TAG}-amd64"
log_info "  - ${BACKEND_IMAGE}:${VERSION_TAG}-arm64"
log_info "  - ${BACKEND_IMAGE}:${VERSION_TAG} (manifest)"
log_info "  - ${BACKEND_IMAGE}:latest (manifest)"

cd "${BUILD_DIR}/DREAM-ML-backend/GEML" || error_exit "Backend directory not found"

# Build linux/amd64
log_step "Step 4a: Building Backend for linux/amd64 (Platform 1/2)"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "   PLATFORM: linux/amd64"
log_info "   TARGET:   ${BACKEND_IMAGE}:${VERSION_TAG}-amd64"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

docker buildx build \
    --platform "linux/amd64" \
    --tag "${BACKEND_IMAGE}:${VERSION_TAG}-amd64" \
    ${NO_CACHE} \
    --push \
    --progress=plain \
    . 2>&1 | tee "${BUILD_DIR}/build-backend-amd64.log"

log_success "✓ Backend linux/amd64 built and pushed successfully"

# Build linux/arm64
log_step "Step 4b: Building Backend for linux/arm64 (Platform 2/2)"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "   PLATFORM: linux/arm64"
log_info "   TARGET:   ${BACKEND_IMAGE}:${VERSION_TAG}-arm64"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

docker buildx build \
    --platform "linux/arm64" \
    --tag "${BACKEND_IMAGE}:${VERSION_TAG}-arm64" \
    ${NO_CACHE} \
    --push \
    --progress=plain \
    . 2>&1 | tee "${BUILD_DIR}/build-backend-arm64.log"

log_success "✓ Backend linux/arm64 built and pushed successfully"

# Create multi-platform manifest using buildx imagetools
log_step "Step 4c: Creating Multi-Platform Manifest for Backend"
log_info "Creating manifest that combines both platforms..."
log_info "Using docker buildx imagetools to merge platform-specific images..."

# Create and push multi-platform manifest for VERSION_TAG
docker buildx imagetools create \
    --tag "${BACKEND_IMAGE}:${VERSION_TAG}" \
    "${BACKEND_IMAGE}:${VERSION_TAG}-amd64" \
    "${BACKEND_IMAGE}:${VERSION_TAG}-arm64"

log_info "Created manifest: ${BACKEND_IMAGE}:${VERSION_TAG}"

# Create and push multi-platform manifest for latest
docker buildx imagetools create \
    --tag "${BACKEND_IMAGE}:latest" \
    "${BACKEND_IMAGE}:${VERSION_TAG}-amd64" \
    "${BACKEND_IMAGE}:${VERSION_TAG}-arm64"

log_info "Created manifest: ${BACKEND_IMAGE}:latest"

log_success "✓ Backend multi-platform manifests created and pushed successfully"

################################################################################
# Step 5: Build Frontend Image (Sequential Multi-Platform)
################################################################################
log_step "Step 5: Building Frontend Image (Sequential Multi-Platform)"

log_info "Building frontend sequentially for platforms: ${PLATFORMS}"
log_info "This approach uses less disk space and is more reliable"
log_info "Tagged images:"
log_info "  - ${FRONTEND_IMAGE}:${VERSION_TAG}-amd64"
log_info "  - ${FRONTEND_IMAGE}:${VERSION_TAG}-arm64"
log_info "  - ${FRONTEND_IMAGE}:${VERSION_TAG} (manifest)"
log_info "  - ${FRONTEND_IMAGE}:latest (manifest)"

cd "${BUILD_DIR}/DREAM-ML-frontend/frontend" || error_exit "Frontend directory not found"

# Build linux/amd64
log_step "Step 5a: Building Frontend for linux/amd64 (Platform 1/2)"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "   PLATFORM: linux/amd64"
log_info "   TARGET:   ${FRONTEND_IMAGE}:${VERSION_TAG}-amd64"
log_info "   BUILD ARGS:"
log_info "     VITE_API_URL=http://localhost:8000"
log_info "     VITE_WS_URL=ws://localhost:8000"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

docker buildx build \
    --platform "linux/amd64" \
    --tag "${FRONTEND_IMAGE}:${VERSION_TAG}-amd64" \
    --build-arg VITE_API_URL=http://localhost:8000 \
    --build-arg VITE_WS_URL=ws://localhost:8000 \
    --build-arg VITE_EXPERIMENTS_DIR=/app/experimentos \
    ${NO_CACHE} \
    --push \
    --progress=plain \
    . 2>&1 | tee "${BUILD_DIR}/build-frontend-amd64.log"

log_success "✓ Frontend linux/amd64 built and pushed successfully"

# Build linux/arm64
log_step "Step 5b: Building Frontend for linux/arm64 (Platform 2/2)"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "   PLATFORM: linux/arm64"
log_info "   TARGET:   ${FRONTEND_IMAGE}:${VERSION_TAG}-arm64"
log_info "   BUILD ARGS:"
log_info "     VITE_API_URL=http://localhost:8000"
log_info "     VITE_WS_URL=ws://localhost:8000"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

docker buildx build \
    --platform "linux/arm64" \
    --tag "${FRONTEND_IMAGE}:${VERSION_TAG}-arm64" \
    --build-arg VITE_API_URL=http://localhost:8000 \
    --build-arg VITE_WS_URL=ws://localhost:8000 \
    --build-arg VITE_EXPERIMENTS_DIR=/app/experimentos \
    ${NO_CACHE} \
    --push \
    --progress=plain \
    . 2>&1 | tee "${BUILD_DIR}/build-frontend-arm64.log"

log_success "✓ Frontend linux/arm64 built and pushed successfully"

# Create multi-platform manifest using buildx imagetools
log_step "Step 5c: Creating Multi-Platform Manifest for Frontend"
log_info "Creating manifest that combines both platforms..."
log_info "Using docker buildx imagetools to merge platform-specific images..."

# Create and push multi-platform manifest for VERSION_TAG
docker buildx imagetools create \
    --tag "${FRONTEND_IMAGE}:${VERSION_TAG}" \
    "${FRONTEND_IMAGE}:${VERSION_TAG}-amd64" \
    "${FRONTEND_IMAGE}:${VERSION_TAG}-arm64"

log_info "Created manifest: ${FRONTEND_IMAGE}:${VERSION_TAG}"

# Create and push multi-platform manifest for latest
docker buildx imagetools create \
    --tag "${FRONTEND_IMAGE}:latest" \
    "${FRONTEND_IMAGE}:${VERSION_TAG}-amd64" \
    "${FRONTEND_IMAGE}:${VERSION_TAG}-arm64"

log_info "Created manifest: ${FRONTEND_IMAGE}:latest"

log_success "✓ Frontend multi-platform manifests created and pushed successfully"

################################################################################
# Step 6: Verify Images on Docker Hub
################################################################################
log_step "Step 6: Verifying Multi-Platform Manifests on Docker Hub"

log_info "Verifying backend manifest..."
if docker manifest inspect "${BACKEND_IMAGE}:${VERSION_TAG}" &>/dev/null; then
    log_success "✓ Backend manifest verified on Docker Hub"
    log_info "Platform details:"
    docker manifest inspect "${BACKEND_IMAGE}:${VERSION_TAG}" | grep -A 3 "platform" | head -20
    echo ""
else
    error_exit "Failed to verify backend manifest on Docker Hub"
fi

log_info "Verifying backend individual platform images..."
if docker manifest inspect "${BACKEND_IMAGE}:${VERSION_TAG}-amd64" &>/dev/null; then
    log_success "✓ Backend linux/amd64 image verified"
else
    log_warning "Backend linux/amd64 image not found (this is expected for manifest-based builds)"
fi

if docker manifest inspect "${BACKEND_IMAGE}:${VERSION_TAG}-arm64" &>/dev/null; then
    log_success "✓ Backend linux/arm64 image verified"
else
    log_warning "Backend linux/arm64 image not found (this is expected for manifest-based builds)"
fi

log_info "Verifying frontend manifest..."
if docker manifest inspect "${FRONTEND_IMAGE}:${VERSION_TAG}" &>/dev/null; then
    log_success "✓ Frontend manifest verified on Docker Hub"
    log_info "Platform details:"
    docker manifest inspect "${FRONTEND_IMAGE}:${VERSION_TAG}" | grep -A 3 "platform" | head -20
    echo ""
else
    error_exit "Failed to verify frontend manifest on Docker Hub"
fi

log_info "Verifying frontend individual platform images..."
if docker manifest inspect "${FRONTEND_IMAGE}:${VERSION_TAG}-amd64" &>/dev/null; then
    log_success "✓ Frontend linux/amd64 image verified"
else
    log_warning "Frontend linux/amd64 image not found (this is expected for manifest-based builds)"
fi

if docker manifest inspect "${FRONTEND_IMAGE}:${VERSION_TAG}-arm64" &>/dev/null; then
    log_success "✓ Frontend linux/arm64 image verified"
else
    log_warning "Frontend linux/arm64 image not found (this is expected for manifest-based builds)"
fi

log_success "All multi-platform manifests verified successfully!"

################################################################################
# Step 7: Create Production docker-compose.yml
################################################################################
log_step "Step 7: Creating Production docker-compose.yml"

cd "${BUILD_DIR}" || error_exit "Could not return to build directory"

# Create dist directory
mkdir -p "${DIST_DIR}"
log_info "Created distribution directory: ${DIST_DIR}"

# Create production docker-compose.yml
log_info "Generating production docker-compose.yml..."
cat > "${DIST_DIR}/docker-compose.yml" <<EOF
version: '3.8'

################################################################################
# DREAM ML - Production Deployment Configuration
################################################################################
# This docker-compose.yml uses pre-built images from Docker Hub
# Version: ${VERSION_TAG}
#
# Usage:
#   docker-compose up -d
#
# Access:
#   Frontend: http://localhost:5173
#   Backend:  http://localhost:8000
#   MLflow:   http://localhost:5000
################################################################################

services:
  backend:
    image: ${BACKEND_IMAGE}:${VERSION_TAG}
    # Platform auto-detected by Docker based on host architecture
    # Multi-platform manifest supports both linux/amd64 and linux/arm64
    ports:
      - "8000:8000"
      - "5000:5000"
    env_file:
      - .env
    environment:
      - DEBUG=0
    volumes:
      # Bind mount for experiments - allows direct file access from host
      - ./experimentos:/app/experimentos
      # Named volumes for persistent data (portable across platforms)
      - mlruns:/app/mlruns
      - media:/app/media
      - staticfiles:/app/staticfiles
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "sh", "-c", "curl -f http://localhost:8000/api/health/ || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  frontend:
    image: ${FRONTEND_IMAGE}:${VERSION_TAG}
    # Platform auto-detected by Docker based on host architecture
    # Multi-platform manifest supports both linux/amd64 and linux/arm64
    ports:
      - "5173:80"
    env_file:
      - .env
    depends_on:
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "sh", "-c", "wget --quiet --tries=1 --spider http://localhost/ || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

# Named volumes for data persistence
# Note: experiments uses bind mount (./experimentos) for direct host access
volumes:
  mlruns:
  media:
  staticfiles:
EOF

log_success "Production docker-compose.yml created"

################################################################################
# Step 8: Copy .env File
################################################################################
log_step "Step 8: Copying Environment Configuration"

if [ -f "${BUILD_DIR}/.env.example" ]; then
    cp "${BUILD_DIR}/.env.example" "${DIST_DIR}/.env"
    log_success "Copied .env.example to distribution"
elif [ -f "${BUILD_DIR}/.env" ]; then
    cp "${BUILD_DIR}/.env" "${DIST_DIR}/.env"
    log_success "Copied .env to distribution"
else
    error_exit ".env or .env.example file not found"
fi

# Add comment to .env about bind mount
cat >> "${DIST_DIR}/.env" <<'ENV_COMMENT'

# ============================================================================
# Experiments Directory
# ============================================================================
# The EXPERIMENTS_DIR variable points to /app/experimentos inside the container.
# This directory is bind-mounted to ./experimentos on your host machine,
# allowing you to directly access experiment files, datasets, and results.
#
# You can access files at: <installation-directory>/experimentos/
ENV_COMMENT

log_success "Added experimentos documentation to .env"

################################################################################
# Step 8a: Create Experimentos Directory
################################################################################
log_step "Step 8a: Creating Experimentos Directory Structure"

log_info "Creating experimentos directory in distribution..."
mkdir -p "${DIST_DIR}/experimentos"

log_info "Copying README to experimentos directory..."
if [ -f "${BUILD_DIR}/scripts/build/experimentos-README.md" ]; then
    cp "${BUILD_DIR}/scripts/build/experimentos-README.md" "${DIST_DIR}/experimentos/README.md"
    log_success "Created experimentos directory with README"
else
    log_warning "experimentos-README.md not found, creating basic README..."
    cat > "${DIST_DIR}/experimentos/README.md" <<'EOF'
# Experimentos Directory

This directory is used by DREAM ML to store experiment data, datasets, and results.
Files created by the application will appear here automatically.

You can directly access, modify, or backup files in this directory from your file manager.
EOF
    log_success "Created experimentos directory with basic README"
fi

################################################################################
# Step 9: Create README.txt
################################################################################
log_step "Step 9: Creating README.txt"

cat > "${DIST_DIR}/README.txt" <<'EOF'
################################################################################
#                           DREAM ML - Installation Guide                     #
################################################################################

Version: ${VERSION_TAG}
Last Updated: $(date +%Y-%m-%d)

================================================================================
TABLE OF CONTENTS
================================================================================
1. System Requirements
2. Prerequisites
3. Installation Instructions
   3.1. macOS (Apple Silicon)
   3.2. Windows 10/11
   3.3. Linux (x86_64)
4. Post-Installation Verification
5. Accessing the Application
6. Troubleshooting
7. Cleanup/Uninstall

================================================================================
1. SYSTEM REQUIREMENTS
================================================================================

Minimum Hardware Requirements:
  - CPU: 4 cores (8 cores recommended)
  - RAM: 8 GB (16 GB recommended for large datasets)
  - Disk: 20 GB free space (more for datasets and experiments)
  - Internet: Required for initial image download (~2-5 GB)

Supported Operating Systems:
  - macOS 11+ (Apple Silicon M1/M2/M3)
  - Windows 10/11 (64-bit)
  - Linux (x86_64) - Ubuntu 20.04+, Debian 11+, RHEL 8+, etc.

================================================================================
2. PREREQUISITES
================================================================================

REQUIRED SOFTWARE:
  - Docker Desktop (macOS/Windows) or Docker Engine (Linux)
  - Docker Compose v2.0+

INSTALLATION INSTRUCTIONS:

macOS:
  1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop
  2. Install Docker Desktop
  3. Start Docker Desktop from Applications
  4. Verify installation:
     docker --version
     docker-compose --version

Windows:
  1. Enable WSL 2 (Windows Subsystem for Linux):
     - Open PowerShell as Administrator
     - Run: wsl --install
     - Restart your computer
  2. Download Docker Desktop from: https://www.docker.com/products/docker-desktop
  3. Install Docker Desktop
  4. Start Docker Desktop
  5. Verify installation:
     docker --version
     docker-compose --version

Linux:
  1. Install Docker Engine:
     curl -fsSL https://get.docker.com -o get-docker.sh
     sudo sh get-docker.sh
     sudo usermod -aG docker $USER
     newgrp docker

  2. Install Docker Compose:
     sudo apt-get update
     sudo apt-get install docker-compose-plugin

  3. Verify installation:
     docker --version
     docker compose version

================================================================================
3. INSTALLATION INSTRUCTIONS
================================================================================

3.1. macOS (Apple Silicon)
---------------------------
1. Extract the dream-ml.zip file
2. Open Terminal and navigate to the extracted folder
3. Make the install script executable:
   chmod +x install-macos.sh
4. Run the installation script:
   ./install-macos.sh
5. Follow the on-screen instructions

3.2. Windows 10/11
------------------
1. Extract the dream-ml.zip file
2. Right-click on "install-windows.bat" and select "Run as Administrator"
   OR
   Open PowerShell as Administrator, navigate to the folder, and run:
   .\install-windows.ps1
3. Follow the on-screen instructions

Note: If you see a security warning, you may need to allow script execution:
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

3.3. Linux (x86_64)
-------------------
1. Extract the dream-ml.zip file:
   unzip dream-ml.zip
   cd dream-ml
2. Make the install script executable:
   chmod +x install-linux.sh
3. Run the installation script:
   ./install-linux.sh
4. Follow the on-screen instructions

================================================================================
4. POST-INSTALLATION VERIFICATION
================================================================================

After installation, the scripts will automatically verify that all services
are running. You should see:

✓ Backend container: HEALTHY
✓ Frontend container: HEALTHY
✓ All services: UP

If any service shows as UNHEALTHY, check the troubleshooting section.

================================================================================
5. ACCESSING THE APPLICATION
================================================================================

Once installation is complete, access the application at:

Frontend (Web UI):    http://localhost:5173
Backend API:          http://localhost:8000
MLflow UI:            http://localhost:5000
API Documentation:    http://localhost:8000/api/docs/

Default Credentials: (if applicable)
  - Refer to project documentation for authentication details

================================================================================
5.1. EXPERIMENT DATA DIRECTORY
================================================================================

The installation includes an "experimentos" directory for storing:
  - Uploaded datasets (CSV files)
  - Experiment configurations
  - Generated reports and visualizations
  - ML model artifacts

ACCESSING EXPERIMENT FILES:

macOS:
  - Use Finder to navigate to the installation folder
  - Open the "experimentos" folder
  - Files appear automatically as you run experiments

Windows:
  - Use File Explorer to navigate to the installation folder
  - Open the "experimentos" folder
  - Files appear automatically as you run experiments

Linux:
  - Navigate to the installation folder
  - Open the "experimentos" folder
  - Files appear automatically as you run experiments

BACKING UP YOUR DATA:

macOS/Linux:
  cp -r experimentos/ experimentos-backup/
  tar -czf experiments-backup.tar.gz experimentos/

Windows (PowerShell):
  Copy-Item -Recurse experimentos\ experimentos-backup\

IMPORTANT: The experimentos directory uses a bind mount, which means files
are stored directly on your computer (not inside Docker). This allows you to:
  - Access files without Docker commands
  - Backup data easily
  - Move files in/out freely
  - Keep data even if containers are removed

================================================================================
6. TROUBLESHOOTING
================================================================================

Issue: "Docker daemon is not running"
Solution:
  - macOS/Windows: Start Docker Desktop application
  - Linux: sudo systemctl start docker

Issue: "Port already in use" (e.g., 5173, 8000, 5000)
Solution:
  - Stop the application using that port
  - Or modify the ports in docker-compose.yml:
    ports:
      - "NEW_PORT:CONTAINER_PORT"

Issue: "Permission denied" errors
Solution:
  - macOS/Linux: Ensure scripts are executable (chmod +x script-name.sh)
  - Linux: Ensure user is in docker group (sudo usermod -aG docker \$USER)
  - Windows: Run PowerShell as Administrator

Issue: Cannot read/write files in experimentos directory (Linux only)
Solution:
  1. Fix ownership:
     sudo chown -R \$USER:\$USER experimentos/
  2. Fix permissions:
     sudo chmod -R 775 experimentos/
  3. If issues persist, match container UID:
     sudo chown -R 1000:1000 experimentos/

  The container runs as UID 1000. Most Linux systems use UID 1000 for the
  first user, so this usually isn't an issue.

Issue: Containers start but show UNHEALTHY status
Solution:
  1. Check container logs:
     docker-compose logs backend
     docker-compose logs frontend
  2. Verify .env file configuration
  3. Ensure sufficient system resources (RAM, disk space)
  4. Restart containers:
     docker-compose restart

Issue: "Cannot pull image" or network errors
Solution:
  - Check internet connection
  - Verify Docker Hub is accessible: https://hub.docker.com
  - Try pulling manually:
    docker pull tomasmanriquez480/dreaml-ml-backend:${VERSION_TAG}
    docker pull tomasmanriquez480/dreaml-ml-frontend:${VERSION_TAG}

Issue: Slow performance
Solution:
  - Increase Docker Desktop memory allocation (macOS/Windows):
    Docker Desktop → Settings → Resources → Memory
  - Recommended: At least 8 GB
  - Close other resource-intensive applications

================================================================================
7. CLEANUP/UNINSTALL
================================================================================

To stop the application:
  docker-compose down

To stop and remove all data (WARNING: This deletes all experiments/data):
  docker-compose down -v

To completely uninstall:
  1. Stop and remove containers and volumes:
     docker-compose down -v
  2. Remove downloaded images:
     docker rmi tomasmanriquez480/dreaml-ml-backend:${VERSION_TAG}
     docker rmi tomasmanriquez480/dreaml-ml-frontend:${VERSION_TAG}
  3. Delete the dream-ml folder

Alternative: Use the cleanup option in the installation scripts:
  - macOS/Linux: ./install-macos.sh (or install-linux.sh) and select cleanup
  - Windows: .\install-windows.ps1 and select cleanup

================================================================================
SUPPORT & DOCUMENTATION
================================================================================

For more information, visit:
  - Project Repository: [Add your GitHub URL]
  - Documentation: [Add docs URL]
  - Issue Tracker: [Add issues URL]

For questions or issues, please file an issue on GitHub or contact the
development team.

================================================================================
EOF

# Replace template variables in README.txt
sed -i '' "s/\${VERSION_TAG}/${VERSION_TAG}/g" "${DIST_DIR}/README.txt"
sed -i '' "s/\$(date +%Y-%m-%d)/$(date +%Y-%m-%d)/g" "${DIST_DIR}/README.txt"

log_success "README.txt created"

################################################################################
# Step 10: Copy Installation Scripts
################################################################################
log_step "Step 10: Copying Installation Scripts"

log_info "Copying installation scripts to distribution directory..."

cp "${BUILD_DIR}/scripts/install/macos/install-macos.sh" "${DIST_DIR}/"
cp "${BUILD_DIR}/scripts/install/linux/install-linux.sh" "${DIST_DIR}/"
cp "${BUILD_DIR}/scripts/install/windows/install-windows.ps1" "${DIST_DIR}/"
cp "${BUILD_DIR}/scripts/install/windows/install-windows.bat" "${DIST_DIR}/"

# Make scripts executable
chmod +x "${DIST_DIR}/install-macos.sh"
chmod +x "${DIST_DIR}/install-linux.sh"

log_success "Installation scripts copied"

################################################################################
# Step 11: Create ZIP Archive
################################################################################
log_step "Step 11: Creating Distribution Archive"

cd "${BUILD_DIR}" || error_exit "Could not return to build directory"

log_info "Creating ${ZIP_NAME}..."

# Remove old zip if exists
[ -f "${ZIP_NAME}" ] && rm "${ZIP_NAME}"

# Create zip
cd "${DIST_DIR}" || error_exit "Could not enter dist directory"
zip -r "../${ZIP_NAME}" . -x "*.DS_Store"
cd "${BUILD_DIR}" || error_exit "Could not return to build directory"

log_success "Distribution archive created: ${ZIP_NAME}"

################################################################################
# Step 12: Summary
################################################################################
log_step "Build Complete!"

echo ""
log_success "Images built and pushed successfully:"
echo "  Backend:  ${BACKEND_IMAGE}:${VERSION_TAG}"
echo "  Frontend: ${FRONTEND_IMAGE}:${VERSION_TAG}"
echo ""
log_success "Distribution package created:"
echo "  Location: ${BUILD_DIR}/${ZIP_NAME}"
echo "  Contents:"
echo "    - docker-compose.yml"
echo "    - .env"
echo "    - README.txt"
echo "    - experimentos/ (directory for experiment data)"
echo "    - install-macos.sh"
echo "    - install-linux.sh"
echo "    - install-windows.ps1"
echo "    - install-windows.bat"
echo ""
log_info "Build logs saved:"
echo "  Backend (amd64):  ${BUILD_DIR}/build-backend-amd64.log"
echo "  Backend (arm64):  ${BUILD_DIR}/build-backend-arm64.log"
echo "  Frontend (amd64): ${BUILD_DIR}/build-frontend-amd64.log"
echo "  Frontend (arm64): ${BUILD_DIR}/build-frontend-arm64.log"
echo ""
log_info "Next steps:"
echo "  1. Distribute ${ZIP_NAME} to users"
echo "  2. Users extract the zip and run the appropriate install script"
echo "  3. Tag this release in git: git tag ${VERSION_TAG}"
echo ""
log_success "All done! 🚀"
