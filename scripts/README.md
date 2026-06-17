# DREAM ML - Build & Installation Scripts

This directory contains scripts for building multi-platform Docker images and installing DREAM ML across different operating systems.

## Directory Structure

```
scripts/
├── build/
│   └── build-and-push.sh          # Multi-platform build and push script
├── install/
│   ├── macos/
│   │   └── install-macos.sh       # macOS (Apple Silicon) installer
│   ├── windows/
│   │   ├── install-windows.ps1    # Windows PowerShell installer
│   │   └── install-windows.bat    # Windows batch wrapper
│   └── linux/
│       └── install-linux.sh       # Linux (x86_64) installer
└── README.md                       # This file
```

## Overview

### Build Script (Developer)

**Script:** `build/build-and-push.sh`

This script is for **developers** who need to build and publish new versions of DREAM ML.

**What it does:**
- Builds Docker images for multiple platforms (linux/amd64, linux/arm64)
- Pushes images to Docker Hub registry: `tomasmanriquez480/dreaml-ml`
- Creates a distribution package (`dream-ml.zip`) containing:
  - Production-ready `docker-compose.yml`
  - `.env` configuration file
  - `README.txt` with installation instructions
  - Platform-specific installation scripts

**Requirements:**
- macOS with Apple Silicon (M1/M2/M3)
- Docker Desktop with buildx support
- Authenticated to Docker Hub (`docker login`)

**Usage:**
```bash
cd scripts/build
./build-and-push.sh
```

You'll be prompted for:
- Semantic version number (e.g., 1.0.0)

**Output:**
- Images pushed to Docker Hub with tags: `vX.X.X` and `latest`
- Distribution package: `dream-ml.zip` in project root
- Build logs: `build-backend.log` and `build-frontend.log`

### Installation Scripts (End Users)

These scripts are for **end users** who want to install and run DREAM ML on their systems.

#### macOS Installation

**Script:** `install/macos/install-macos.sh`

**Platform:** macOS 11+ with Apple Silicon (M1/M2/M3)

**Requirements:**
- Docker Desktop for Mac
- macOS 11+ (Big Sur or later)

**Usage:**
```bash
chmod +x install-macos.sh
./install-macos.sh
```

**Cleanup:**
```bash
./install-macos.sh --cleanup
```

#### Windows Installation

**Scripts:**
- `install/windows/install-windows.ps1` (PowerShell)
- `install/windows/install-windows.bat` (Batch wrapper)

**Platform:** Windows 10/11 (64-bit)

**Requirements:**
- Docker Desktop for Windows
- WSL 2 (Windows Subsystem for Linux)

**Usage (Easy - Double-click):**
1. Right-click `install-windows.bat`
2. Select "Run as Administrator"

**Usage (PowerShell):**
```powershell
# Run PowerShell as Administrator
.\install-windows.ps1
```

**Cleanup:**
```powershell
.\install-windows.ps1 -cleanup
```

**Note:** If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Linux Installation

**Script:** `install/linux/install-linux.sh`

**Platform:** Linux x86_64/amd64 (Ubuntu 20.04+, Debian 11+, RHEL 8+, etc.)

**Requirements:**
- Docker Engine
- Docker Compose v2.0+

**Usage:**
```bash
chmod +x install-linux.sh
./install-linux.sh
```

**Cleanup:**
```bash
./install-linux.sh --cleanup
```

## Features

All scripts include:

### ✓ Verbose Logging
- Color-coded output (INFO, SUCCESS, WARNING, ERROR)
- Step-by-step progress tracking
- Detailed error messages

### ✓ Prerequisite Verification
- Checks for Docker installation
- Verifies Docker daemon is running
- Validates required files exist

### ✓ Error Handling
- Safe error exits
- Clear error messages
- Troubleshooting guidance

### ✓ Health Monitoring
- Waits for containers to be healthy
- Displays container status
- Shows recent logs

### ✓ User Guidance
- Access URLs displayed after installation
- Useful commands provided
- Optional browser auto-open

### ✓ Cleanup Functionality
- Safe uninstallation
- Volume removal (with confirmation)
- Image cleanup

## Build Process Details

### Multi-Platform Build

The build script uses Docker Buildx to create multi-architecture images:

**Supported Platforms:**
- `linux/amd64` - For Windows, Linux x86_64 servers
- `linux/arm64` - For Apple Silicon Macs, ARM Linux

**How it Works:**
1. Creates a buildx builder instance
2. Builds both backend and frontend images
3. Pushes images with multi-arch manifest
4. Single image tag works on all platforms

**Benefits:**
- Users pull the correct architecture automatically
- No need for platform-specific tags
- Efficient image distribution

### Version Management

**Semantic Versioning:**
- Format: `vMAJOR.MINOR.PATCH` (e.g., v1.2.3)
- Two tags created: `vX.X.X` and `latest`

**Example:**
```bash
# Building version 1.2.3 creates:
tomasmanriquez480/dreaml-ml-backend:v1.2.3
tomasmanriquez480/dreaml-ml-backend:latest
tomasmanriquez480/dreaml-ml-frontend:v1.2.3
tomasmanriquez480/dreaml-ml-frontend:latest
```

## Distribution Package

The build script creates `dream-ml.zip` containing:

```
dream-ml.zip
├── docker-compose.yml      # Production config (uses remote images)
├── .env                    # Environment configuration
├── README.txt              # Detailed installation guide
├── install-macos.sh        # macOS installer
├── install-linux.sh        # Linux installer
├── install-windows.ps1     # Windows PowerShell installer
└── install-windows.bat     # Windows batch wrapper
```

**Distribution Workflow:**
1. Developer runs `build-and-push.sh`
2. Images are pushed to Docker Hub
3. `dream-ml.zip` is created
4. Distribute zip to end users
5. Users extract and run appropriate installer

## Troubleshooting

### Build Script Issues

**Error: "Not logged in to Docker Hub"**
- Run: `docker login`
- Enter Docker Hub credentials

**Error: "Docker buildx is not available"**
- Update Docker Desktop to latest version
- Buildx is included in Docker Desktop 19.03+

**Error: "Failed to push images"**
- Check internet connection
- Verify Docker Hub credentials
- Ensure repository access permissions

### Installation Script Issues

**Error: "Docker is not installed"**
- See installation instructions in README.txt
- Visit: https://www.docker.com/products/docker-desktop

**Error: "Docker daemon is not running"**
- macOS/Windows: Start Docker Desktop application
- Linux: `sudo systemctl start docker`

**Error: "Permission denied"**
- macOS/Linux: Ensure scripts are executable (`chmod +x`)
- Linux: Add user to docker group: `sudo usermod -aG docker $USER`
- Windows: Run as Administrator

**Error: "Port already in use"**
- Stop conflicting services using ports 5173, 8000, 5000
- Or modify ports in `docker-compose.yml`

**Containers unhealthy:**
- Check logs: `docker compose logs`
- Verify system resources (RAM, disk space)
- Restart: `docker compose restart`

## Development Workflow

### For Developers (Building New Versions)

1. Make code changes to DREAM ML
2. Test changes locally
3. Commit and push to repository
4. Run build script:
   ```bash
   cd scripts/build
   ./build-and-push.sh
   ```
5. Enter version number when prompted
6. Distribute `dream-ml.zip` to users
7. Tag release in git:
   ```bash
   git tag vX.X.X
   git push origin vX.X.X
   ```

### For End Users (Installing)

1. Download `dream-ml.zip`
2. Extract the zip file
3. Run the appropriate installer:
   - **macOS:** `./install-macos.sh`
   - **Windows:** Double-click `install-windows.bat`
   - **Linux:** `./install-linux.sh`
4. Follow on-screen instructions
5. Access application at http://localhost:5173

## Technical Details

### Docker Images

**Backend Image:**
- Repository: `tomasmanriquez480/dreaml-ml-backend`
- Base: `python:3.11-slim`
- Multi-stage build with Cython compilation
- Exposes ports: 8000 (Django), 5000 (MLflow)

**Frontend Image:**
- Repository: `tomasmanriquez480/dreaml-ml-frontend`
- Base: `node:18-alpine` (build), `nginx:alpine` (runtime)
- Multi-stage build with Vite
- Exposes port: 80 (mapped to 5173)

### Named Volumes

All installations use Docker named volumes for data persistence:
- `experiments` - Experiment data
- `mlruns` - MLflow tracking data
- `media` - User-uploaded files
- `staticfiles` - Django static files

**Benefits:**
- Platform-independent paths
- Automatic management by Docker
- Survives container restarts
- Easy backup/restore

### Health Checks

**Backend:**
```dockerfile
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/api/health/"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

**Frontend:**
```dockerfile
healthcheck:
  test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 10s
```

## Support

For issues or questions:
- Check `README.txt` in distribution package
- Review container logs: `docker compose logs`
- File issue on GitHub repository
- Contact development team

## License

Same as main DREAM ML project.

## Version History

Track versions using git tags:
```bash
git tag                    # List all versions
git show vX.X.X           # Show version details
```

---

**Last Updated:** 2025-10-15
**Maintained By:** DREAM ML Development Team
