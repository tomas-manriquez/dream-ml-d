# GEML Multi-Architecture Docker Build Instructions

This guide provides comprehensive instructions for building the GEML Docker image for multiple architectures using Docker Buildx.

## Prerequisites

1. **Docker Desktop** (Mac/Windows) or **Docker Engine 19.03+** (Linux)
2. **Docker Buildx** plugin (included in Docker Desktop, may need installation on Linux)
3. **QEMU** for cross-platform builds (handled automatically by Docker Desktop)

### Verify Docker Buildx Installation

```bash
docker buildx version
```

If not installed, follow: https://docs.docker.com/buildx/working-with-buildx/

## Quick Start

### Local Build (Your Platform Only)

Build for your current platform:

```bash
cd /path/to/DREAM-ML-backend/GEML
docker build -t geml:latest .
```

### Single-Platform Builds with Buildx

**For Windows 10/11 (linux/amd64):**
```bash
docker buildx build --platform linux/amd64 -t geml:latest-amd64 --load .
```

**For macOS Apple Silicon M1/M2/M3 (linux/arm64):**
```bash
docker buildx build --platform linux/arm64 -t geml:latest-arm64 --load .
```

**For Linux ARM64:**
```bash
docker buildx build --platform linux/arm64 -t geml:latest-arm64 --load .
```

> **Note**: The `--load` flag loads the image into your local Docker daemon for immediate use.

## Multi-Platform Build Strategy

Since you're building **separately for each architecture**, here are the recommended approaches:

### Option 1: Build and Load Locally (Recommended for Testing)

Build for specific platform and load into Docker:

```bash
# AMD64 (Windows/Linux x86_64)
docker buildx build \
  --platform linux/amd64 \
  -t geml:latest-amd64 \
  --load \
  .

# ARM64 (Mac M1/M2, Linux ARM)
docker buildx build \
  --platform linux/arm64 \
  -t geml:latest-arm64 \
  --load \
  .
```

### Option 2: Build and Push to Registry (Recommended for Production)

If using a container registry (Docker Hub, AWS ECR, GCP GCR, etc.):

```bash
# AMD64 build
docker buildx build \
  --platform linux/amd64 \
  -t your-registry/geml:latest-amd64 \
  --push \
  .

# ARM64 build
docker buildx build \
  --platform linux/arm64 \
  -t your-registry/geml:latest-arm64 \
  --push \
  .
```

### Option 3: Multi-Platform Manifest (Advanced)

Build both architectures in one command and create a multi-platform manifest:

```bash
# Create a new builder instance with multi-platform support
docker buildx create --name geml-builder --use
docker buildx inspect --bootstrap

# Build and push for both platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t your-registry/geml:latest \
  --push \
  .
```

This creates a single tag (`geml:latest`) that automatically serves the correct image for each platform.

## Running the Container

### Basic Run

```bash
docker run -d \
  --name geml-app \
  -p 8000:8000 \
  -p 5000:5000 \
  geml:latest-arm64
```

### Production Run with Volumes

Mount volumes for persistent data:

```bash
docker run -d \
  --name geml-app \
  -p 8000:8000 \
  -p 5000:5000 \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/media:/app/media \
  -v $(pwd)/staticfiles:/app/staticfiles \
  --env-file .env \
  geml:latest-arm64
```

### With Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  geml:
    image: geml:latest-arm64  # Change based on your platform
    container_name: geml-app
    ports:
      - "8000:8000"
      - "5000:5000"
    volumes:
      - ./mlruns:/app/mlruns
      - ./media:/app/media
      - ./staticfiles:/app/staticfiles
    env_file:
      - .env
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## Build Optimization Tips

### 1. Use BuildKit Cache

Enable BuildKit for faster builds:

```bash
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain  # For detailed output
```

### 2. Layer Caching

The Dockerfile is optimized for layer caching. Rebuild only when necessary:
- Requirements changes → Full rebuild
- Code changes → Partial rebuild (faster)

### 3. Build Arguments

Pass custom build arguments if needed:

```bash
docker buildx build \
  --platform linux/arm64 \
  --build-arg PYTHON_VERSION=3.11 \
  -t geml:latest \
  .
```

## Platform-Specific Notes

### macOS Apple Silicon (M1/M2/M3)

- Your native platform is `linux/arm64`
- Use `--load` to load images directly
- Cross-compilation to AMD64 works but is slower
- QEMU emulation is automatic via Docker Desktop

### Windows 10/11

- Your target platform is `linux/amd64`
- Enable WSL2 backend in Docker Desktop
- Use `--load` to load images into Docker Desktop
- Cross-compilation to ARM64 works but is slower

### Linux (x86_64 or ARM64)

Install QEMU for cross-platform builds:

```bash
# Ubuntu/Debian
sudo apt-get install qemu-user-static

# Enable binfmt
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

## Troubleshooting

### Build Fails with "executor failed running..."

**Solution**: Ensure buildx builder is running:
```bash
docker buildx inspect --bootstrap
```

### Cross-Platform Build is Very Slow

**Expected behavior**: Cross-platform builds use QEMU emulation and are 10-50x slower.
**Solution**: Build on native platform when possible, or use CI/CD with native runners.

### "failed to solve with frontend dockerfile.v0"

**Solution**: Update Docker and Buildx to latest versions:
```bash
docker buildx version
# Should be v0.10.0 or higher
```

### Permission Denied on Volumes

The container runs as non-root user (UID 1000). Ensure host directories have correct permissions:

```bash
chown -R 1000:1000 mlruns/ media/ staticfiles/
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build Multi-Arch Docker Image

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v2

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Build AMD64
        run: |
          docker buildx build \
            --platform linux/amd64 \
            -t geml:latest-amd64 \
            --load \
            .

      - name: Build ARM64
        run: |
          docker buildx build \
            --platform linux/arm64 \
            -t geml:latest-arm64 \
            --load \
            .
```

## Image Size Optimization

The optimized Dockerfile includes:
- ✅ Multi-stage builds (removes build dependencies)
- ✅ `--no-install-recommends` for apt packages
- ✅ `--no-cache-dir` for pip installs
- ✅ `.dockerignore` to exclude unnecessary files
- ✅ `tensorflow-cpu` instead of `tensorflow` (~500MB saved)
- ✅ Minimal base image (`python:3.11-slim`)

Expected final image size: **2.5-3.5 GB** (down from 4-5 GB with tensorflow-gpu)

## Development vs Production

### Development Build

Include test dependencies:

```bash
# Temporarily modify dockerfile to use both requirements files
COPY requirements-base.txt requirements-dev.txt ./
RUN pip install -r requirements-base.txt -r requirements-dev.txt
```

### Production Build

Uses only `requirements-base.txt` (excludes pytest, coverage, etc.)

## Security Considerations

1. **Non-root user**: Container runs as `appuser` (UID 1000)
2. **No secrets in image**: Use `--env-file` or secret management
3. **Regular updates**: Rebuild images monthly for security patches
4. **Scan images**: Use `docker scan` or Trivy for vulnerability scanning

```bash
docker scan geml:latest-arm64
```

## Support

For issues with:
- **Docker/Buildx**: https://docs.docker.com/buildx/
- **GEML Application**: Contact project maintainers
- **Multi-arch builds**: https://www.docker.com/blog/multi-arch-build-and-images-the-simple-way/

## Version History

- **v1.0** (2025-10-14): Initial production-optimized multi-arch Dockerfile
  - Python 3.11
  - tensorflow-cpu for multi-platform support
  - Non-root user security
  - Volume support for persistence
