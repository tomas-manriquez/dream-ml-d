#!/bin/bash

################################################################################
# DREAM ML - Linux Installation Script (x86_64/amd64)
################################################################################
# This script installs and runs DREAM ML on Linux (x86_64/amd64)
#
# Requirements:
#   - Linux (x86_64/amd64) - Ubuntu 20.04+, Debian 11+, RHEL 8+, etc.
#   - Docker Engine
#   - Docker Compose v2.0+
#
# Usage:
#   ./install-linux.sh
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
MAGENTA='\033[0;35m'
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

log_header() {
    echo -e "${MAGENTA}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║                   DREAM ML - Linux Installer                   ║"
    echo "║                      Linux x86_64/amd64                        ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"
}

################################################################################
# Error handler
################################################################################
error_exit() {
    log_error "$1"
    log_error "Installation failed. Please check the errors above."
    log_info "For troubleshooting, see README.txt"
    exit 1
}

# Trap errors
trap 'error_exit "An unexpected error occurred at line $LINENO"' ERR

################################################################################
# Cleanup function
################################################################################
cleanup() {
    log_step "Cleanup Mode"

    log_warning "This will stop and remove all DREAM ML containers and data."
    log_warning "All experiments, MLflow runs, and uploaded files will be deleted."
    read -p "Are you sure you want to continue? (yes/no): " -r

    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Cleanup cancelled"
        exit 0
    fi

    log_info "Stopping containers..."
    if docker compose down -v 2>/dev/null || docker-compose down -v 2>/dev/null; then
        log_success "Containers stopped and volumes removed"
    else
        log_warning "Failed to stop containers (they may not be running)"
    fi

    log_info "Removing images..."
    docker images | grep "tomasmanriquez480/dreaml-ml" | awk '{print $1":"$2}' | xargs -r docker rmi || true

    log_success "Cleanup complete"
    exit 0
}

################################################################################
# Main installation
################################################################################
main() {
    log_header

    # Check for cleanup flag
    if [[ "${1:-}" == "--cleanup" ]] || [[ "${1:-}" == "-c" ]]; then
        cleanup
    fi

    ############################################################################
    # Step 1: System Check
    ############################################################################
    log_step "Step 1: System Verification"

    log_info "Checking operating system..."
    if [[ "$(uname -s)" != "Linux" ]]; then
        error_exit "This script is for Linux only. Use install-macos.sh or install-windows.ps1"
    fi
    log_success "Operating system: Linux"

    log_info "Checking architecture..."
    ARCH=$(uname -m)
    if [[ "$ARCH" != "x86_64" ]] && [[ "$ARCH" != "amd64" ]]; then
        log_warning "This script is optimized for x86_64/amd64"
        log_warning "Detected architecture: $ARCH"
        log_info "The application may not work correctly on this architecture"
    else
        log_success "Architecture: $ARCH"
    fi

    log_info "Detecting Linux distribution..."
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        log_info "Distribution: $NAME $VERSION"
    else
        log_warning "Could not detect Linux distribution"
    fi

    ############################################################################
    # Step 2: Check Prerequisites
    ############################################################################
    log_step "Step 2: Checking Prerequisites"

    log_info "Checking if Docker is installed..."
    if ! command -v docker &>/dev/null; then
        log_error "Docker is not installed"
        echo ""
        echo "Please install Docker Engine:"
        echo "  Quick install:"
        echo "    curl -fsSL https://get.docker.com -o get-docker.sh"
        echo "    sudo sh get-docker.sh"
        echo "    sudo usermod -aG docker \$USER"
        echo "    newgrp docker"
        echo ""
        echo "  Or visit: https://docs.docker.com/engine/install/"
        echo ""
        echo "After installation, re-run this script"
        echo ""
        exit 1
    fi
    log_success "Docker is installed: $(docker --version)"

    log_info "Checking if Docker daemon is running..."
    if ! docker info &>/dev/null; then
        log_error "Docker daemon is not running"
        echo ""
        echo "To start Docker, try:"
        echo "  sudo systemctl start docker"
        echo "  sudo systemctl enable docker"
        echo ""
        exit 1
    fi
    log_success "Docker daemon is running"

    log_info "Checking Docker permissions..."
    if ! docker ps &>/dev/null; then
        log_warning "Current user may not have Docker permissions"
        log_info "You may need to:"
        echo "  1. Add user to docker group: sudo usermod -aG docker \$USER"
        echo "  2. Log out and log back in, or run: newgrp docker"
        echo "  3. Re-run this script"
        echo ""
        error_exit "Docker permission error"
    fi
    log_success "Docker permissions OK"

    log_info "Checking Docker Compose..."
    # Try docker compose (v2) first, then docker-compose (v1)
    if docker compose version &>/dev/null; then
        DOCKER_COMPOSE_CMD="docker compose"
        log_success "Docker Compose v2 is available: $(docker compose version)"
    elif command -v docker-compose &>/dev/null; then
        DOCKER_COMPOSE_CMD="docker-compose"
        log_success "Docker Compose v1 is available: $(docker-compose --version)"
        log_warning "Consider upgrading to Docker Compose v2"
    else
        log_error "Docker Compose is not available"
        echo ""
        echo "Please install Docker Compose:"
        echo "  For Docker Compose v2 (recommended):"
        echo "    sudo apt-get update"
        echo "    sudo apt-get install docker-compose-plugin"
        echo ""
        echo "  Or visit: https://docs.docker.com/compose/install/"
        echo ""
        exit 1
    fi

    ############################################################################
    # Step 3: Verify Required Files
    ############################################################################
    log_step "Step 3: Verifying Installation Files"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    log_info "Checking for docker-compose.yml..."
    if [ ! -f "${SCRIPT_DIR}/docker-compose.yml" ]; then
        error_exit "docker-compose.yml not found in ${SCRIPT_DIR}"
    fi
    log_success "Found docker-compose.yml"

    log_info "Checking for .env file..."
    if [ ! -f "${SCRIPT_DIR}/.env" ]; then
        error_exit ".env file not found in ${SCRIPT_DIR}"
    fi
    log_success "Found .env file"

    ############################################################################
    # Step 3a: Setup Experimentos Directory
    ############################################################################
    log_step "Step 3a: Setting Up Experimentos Directory"

    log_info "Creating experimentos directory for experiment data..."
    if [ ! -d "${SCRIPT_DIR}/experimentos" ]; then
        mkdir -p "${SCRIPT_DIR}/experimentos"
        log_success "Created experimentos directory"
    else
        log_success "Experimentos directory already exists"
    fi

    # Detect current user's UID and GID
    USER_UID=$(id -u)
    USER_GID=$(id -g)
    log_info "Detected user UID: ${USER_UID}, GID: ${USER_GID}"

    # Set ownership to current user (container runs as UID 1000)
    # If user is UID 1000, this will match perfectly
    # Otherwise, we'll set it and add a note
    chown -R "${USER_UID}:${USER_GID}" "${SCRIPT_DIR}/experimentos" 2>/dev/null || {
        log_warning "Could not change ownership. You may need to run:"
        echo "  sudo chown -R ${USER_UID}:${USER_GID} ${SCRIPT_DIR}/experimentos"
    }

    # Set proper permissions (775 = rwxrwxr-x)
    chmod -R 775 "${SCRIPT_DIR}/experimentos" 2>/dev/null || {
        log_warning "Could not change permissions. You may need to run:"
        echo "  sudo chmod -R 775 ${SCRIPT_DIR}/experimentos"
    }

    log_success "Set permissions to 775 (read/write for user and group)"

    if [ "${USER_UID}" != "1000" ]; then
        log_info "Note: Container runs as UID 1000. Your UID is ${USER_UID}"
        log_info "If you encounter permission issues, run:"
        echo "  sudo chown -R 1000:1000 ${SCRIPT_DIR}/experimentos"
    fi

    log_info "Experiment files will be accessible at: ${SCRIPT_DIR}/experimentos"

    ############################################################################
    # Step 4: Pull Docker Images
    ############################################################################
    log_step "Step 4: Pulling Docker Images"

    cd "${SCRIPT_DIR}" || error_exit "Could not change to script directory"

    log_info "Pulling images from Docker Hub..."
    log_info "This may take several minutes depending on your internet connection..."
    echo ""

    if ${DOCKER_COMPOSE_CMD} pull; then
        log_success "Images pulled successfully"
    else
        error_exit "Failed to pull images. Check your internet connection and Docker Hub access."
    fi

    ############################################################################
    # Step 5: Start Services
    ############################################################################
    log_step "Step 5: Starting Services"

    log_info "Starting containers in detached mode..."
    if ${DOCKER_COMPOSE_CMD} up -d; then
        log_success "Containers started"
    else
        error_exit "Failed to start containers"
    fi

    ############################################################################
    # Step 6: Wait for Services to be Healthy
    ############################################################################
    log_step "Step 6: Waiting for Services to be Ready"

    log_info "Waiting for backend to be healthy (this may take 30-60 seconds)..."
    TIMEOUT=120
    ELAPSED=0
    BACKEND_HEALTHY=false

    while [ $ELAPSED -lt $TIMEOUT ]; do
        if ${DOCKER_COMPOSE_CMD} ps | grep -q "backend.*healthy"; then
            BACKEND_HEALTHY=true
            break
        fi
        echo -n "."
        sleep 2
        ELAPSED=$((ELAPSED + 2))
    done
    echo ""

    if [ "$BACKEND_HEALTHY" = true ]; then
        log_success "Backend is healthy"
    else
        log_warning "Backend health check timeout - checking status..."
        ${DOCKER_COMPOSE_CMD} ps
        log_warning "Backend may still be starting. Check logs below."
    fi

    log_info "Waiting for frontend to be healthy..."
    ELAPSED=0
    FRONTEND_HEALTHY=false

    while [ $ELAPSED -lt $TIMEOUT ]; do
        if ${DOCKER_COMPOSE_CMD} ps | grep -q "frontend.*healthy"; then
            FRONTEND_HEALTHY=true
            break
        fi
        echo -n "."
        sleep 2
        ELAPSED=$((ELAPSED + 2))
    done
    echo ""

    if [ "$FRONTEND_HEALTHY" = true ]; then
        log_success "Frontend is healthy"
    else
        log_warning "Frontend health check timeout - checking status..."
        ${DOCKER_COMPOSE_CMD} ps
        log_warning "Frontend may still be starting. Check logs below."
    fi

    ############################################################################
    # Step 7: Display Container Status
    ############################################################################
    log_step "Step 7: Container Status"

    log_info "Current container status:"
    ${DOCKER_COMPOSE_CMD} ps

    ############################################################################
    # Step 8: Display Logs
    ############################################################################
    log_step "Step 8: Recent Container Logs"

    log_info "Backend logs (last 20 lines):"
    echo "----------------------------------------"
    ${DOCKER_COMPOSE_CMD} logs --tail=20 backend
    echo "----------------------------------------"
    echo ""

    log_info "Frontend logs (last 20 lines):"
    echo "----------------------------------------"
    ${DOCKER_COMPOSE_CMD} logs --tail=20 frontend
    echo "----------------------------------------"
    echo ""

    ############################################################################
    # Step 9: Check for Errors
    ############################################################################
    log_step "Step 9: Error Detection"

    log_info "Checking for errors in logs..."
    ERROR_COUNT=0

    if ${DOCKER_COMPOSE_CMD} logs backend | grep -i "error\|exception\|failed" | grep -v "Failed to find" | tail -5; then
        ERROR_COUNT=$((ERROR_COUNT + 1))
        log_warning "Found errors in backend logs (see above)"
    else
        log_success "No critical errors in backend logs"
    fi

    if ${DOCKER_COMPOSE_CMD} logs frontend | grep -i "error\|exception\|failed" | tail -5; then
        ERROR_COUNT=$((ERROR_COUNT + 1))
        log_warning "Found errors in frontend logs (see above)"
    else
        log_success "No critical errors in frontend logs"
    fi

    if [ $ERROR_COUNT -gt 0 ]; then
        log_warning "Some errors were detected. The application may still work."
        log_info "Check the full logs with: ${DOCKER_COMPOSE_CMD} logs"
    fi

    ############################################################################
    # Step 10: Success Message & Access Instructions
    ############################################################################
    log_step "Installation Complete!"

    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║                    ✓ INSTALLATION SUCCESSFUL                   ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"

    log_success "DREAM ML is now running!"
    echo ""
    echo "Access the application at:"
    echo ""
    echo -e "  ${CYAN}Frontend (Web UI):${NC}     http://localhost:5173"
    echo -e "  ${CYAN}Backend API:${NC}           http://localhost:8000"
    echo -e "  ${CYAN}MLflow UI:${NC}             http://localhost:5000"
    echo -e "  ${CYAN}API Documentation:${NC}     http://localhost:8000/api/docs/"
    echo ""

    log_info "Useful commands:"
    echo ""
    echo "  View logs:           ${DOCKER_COMPOSE_CMD} logs -f"
    echo "  Stop services:       ${DOCKER_COMPOSE_CMD} stop"
    echo "  Start services:      ${DOCKER_COMPOSE_CMD} start"
    echo "  Restart services:    ${DOCKER_COMPOSE_CMD} restart"
    echo "  Remove everything:   ./install-linux.sh --cleanup"
    echo ""
    echo "  Access experiments:  cd ${SCRIPT_DIR}/experimentos && ls"
    echo "  Backup experiments:  cp -r experimentos/ experimentos-backup/"
    echo "  Fix permissions:     sudo chown -R \$USER:\$USER experimentos/"
    echo ""

    log_info "For troubleshooting, see README.txt"
    echo ""

    # Offer to open browser if available
    if command -v xdg-open &>/dev/null; then
        read -p "Would you like to open the application in your browser? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Opening browser..."
            xdg-open http://localhost:5173 &>/dev/null &
        fi
    fi

    log_success "Installation script completed successfully! 🚀"
}

################################################################################
# Entry point
################################################################################
main "$@"
