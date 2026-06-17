# DREAM ML Frontend - Multi-Architecture Docker Build Instructions

This guide provides comprehensive instructions for building the DREAM ML Frontend Docker image for multiple architectures using Docker Buildx.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Build Strategies](#build-strategies)
- [Running the Container](#running-the-container)
- [Development vs Production](#development-vs-production)
- [Multi-Platform Build Details](#multi-platform-build-details)
- [Optimization Tips](#optimization-tips)
- [Troubleshooting](#troubleshooting)
- [Environment Variables](#environment-variables)

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

### Production Build (Recommended)

Build the optimized production image for your current platform:

```bash
cd /path/to/DREAM-ML-frontend/frontend
docker build -t dream-ml-frontend:latest .
```

This creates a **production-optimized** image with:
- ✅ Multi-stage build (build deps removed)
- ✅ nginx serving static files
- ✅ ~40-50MB final image size (vs ~400MB+ dev image)
- ✅ Gzip compression enabled
- ✅ SPA routing support
- ✅ Security headers configured

### Run with Docker Compose

From the project root:

```bash
docker-compose up -d
```

Access the frontend at: http://localhost:5173

## Build Strategies

Based on your requirements, we support **on-demand builds per platform** (recommended), with fallback options if needed.

### Option 1: On-Demand Build (Recommended - Simplest)

Build for your current platform automatically:

```bash
docker build -t dream-ml-frontend:latest .
```

Docker automatically detects your platform:
- **macOS Apple Silicon (M1/M2/M3)**: Builds `linux/arm64`
- **Windows 10/11**: Builds `linux/amd64`
- **Linux x86_64**: Builds `linux/amd64`
- **Linux ARM64**: Builds `linux/arm64`

### Option 2: Explicit Platform Build

Specify the target platform explicitly:

**For Windows 10/11 (linux/amd64):**
```bash
docker buildx build --platform linux/amd64 -t dream-ml-frontend:latest-amd64 --load .
```

**For macOS Apple Silicon M1/M2/M3 (linux/arm64):**
```bash
docker buildx build --platform linux/arm64 -t dream-ml-frontend:latest-arm64 --load .
```

**For Linux ARM64:**
```bash
docker buildx build --platform linux/arm64 -t dream-ml-frontend:latest-arm64 --load .
```

> **Note**: The `--load` flag loads the image into your local Docker daemon for immediate use.

### Option 3: Multi-Platform Manifest (Advanced)

If you need a single image tag that works across all platforms:

```bash
# Create a new builder instance with multi-platform support
docker buildx create --name frontend-builder --use
docker buildx inspect --bootstrap

# Build and push for both platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t your-registry/dream-ml-frontend:latest \
  --push \
  .
```

This creates a single tag (`dream-ml-frontend:latest`) that automatically serves the correct image for each platform.

### Option 4: Separate Platform Images

Build images separately per platform with distinct tags:

```bash
# AMD64 (Windows/Linux x86_64)
docker buildx build \
  --platform linux/amd64 \
  -t dream-ml-frontend:latest-amd64 \
  --load \
  .

# ARM64 (Mac M1/M2, Linux ARM)
docker buildx build \
  --platform linux/arm64 \
  -t dream-ml-frontend:latest-arm64 \
  --load \
  .
```

## Running the Container

### Standalone Run

```bash
docker run -d \
  --name dream-ml-frontend \
  -p 5173:80 \
  dream-ml-frontend:latest
```

Access at: http://localhost:5173

### With Docker Compose (Recommended)

The included [docker-compose.yml](../../docker-compose.yml) is pre-configured for production deployment:

```bash
# From project root
docker-compose up -d
```

This automatically:
- Builds the frontend with production optimizations
- Passes environment variables via build args
- Connects to the backend service
- Exposes port 5173

### Health Check

Verify the container is healthy:

```bash
docker ps
# Look for "healthy" status

# Or test the health endpoint
curl http://localhost:5173/health
```

## Development vs Production

### Production Mode (Current Configuration)

**Dockerfile**: Multi-stage build with nginx
- ✅ Optimized for size (~40-50MB)
- ✅ Fast serving with nginx
- ✅ No hot-reloading (rebuild required for changes)
- ✅ Production-ready with security headers

**To rebuild after code changes:**
```bash
docker-compose up -d --build frontend
```

### Development Mode (Optional)

If you need hot-reloading for development, you can temporarily modify `docker-compose.yml`:

```yaml
frontend:
  build:
    context: ./DREAM-ML-frontend/frontend
    dockerfile: dockerfile
  ports:
    - "5173:5173"
  volumes:
    - ./DREAM-ML-frontend/frontend:/app  # Enable hot-reloading
    - /app/node_modules  # Preserve node_modules
  environment:
    - CHOKIDAR_USEPOLLING=true
  command: npm run dev -- --host
```

And use a simpler Dockerfile (dev mode):

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host"]
```

**Note**: The current production setup is optimized for deployment. Use development mode only when actively developing.

## Multi-Platform Build Details

### Supported Platforms

- ✅ **linux/amd64**: Windows 10/11, Linux x86_64
- ✅ **linux/arm64**: macOS Apple Silicon (M1/M2/M3), Linux ARM

### Build Time Comparison

| Build Type | Approximate Time |
|------------|------------------|
| Native build (same platform) | 2-5 minutes |
| Cross-platform build (QEMU emulation) | 10-30 minutes |

**Recommendation**: Always build on your native platform when possible to minimize build time.

### Cross-Platform Build Notes

Docker uses **QEMU emulation** for cross-platform builds:
- **macOS → AMD64**: Slower (10-50x), but works
- **Windows → ARM64**: Slower (10-50x), but works
- **Native builds**: Full speed

### Platform-Specific Instructions

#### macOS Apple Silicon (M1/M2/M3)

Your native platform is `linux/arm64`:

```bash
# Fast native build
docker build -t dream-ml-frontend:latest .

# Or explicitly
docker buildx build --platform linux/arm64 -t dream-ml-frontend:latest --load .
```

**Cross-compile to AMD64** (slower):
```bash
docker buildx build --platform linux/amd64 -t dream-ml-frontend:latest-amd64 --load .
```

#### Windows 10/11

Your target platform is `linux/amd64`:

```bash
# Native build (recommended)
docker build -t dream-ml-frontend:latest .

# Or explicitly
docker buildx build --platform linux/amd64 -t dream-ml-frontend:latest --load .
```

**Requirements**:
- ✅ Enable WSL2 backend in Docker Desktop
- ✅ Use PowerShell or Command Prompt

**Cross-compile to ARM64** (slower):
```bash
docker buildx build --platform linux/arm64 -t dream-ml-frontend:latest-arm64 --load .
```

#### Linux (x86_64 or ARM64)

Install QEMU for cross-platform builds:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install qemu-user-static

# Enable binfmt
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

Then build as usual:
```bash
docker build -t dream-ml-frontend:latest .
```

## Optimization Tips

### 1. Use BuildKit Cache

Enable BuildKit for faster builds:

```bash
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain  # For detailed output
```

### 2. Layer Caching

The Dockerfile is optimized for layer caching:
- `package*.json` copied first → npm dependencies cached
- Source code copied later → only rebuilds when code changes

**To maximize caching:**
- Don't modify `package.json` unnecessarily
- Use `.dockerignore` to exclude unnecessary files (already configured)

### 3. Build Arguments

Override environment variables at build time:

```bash
docker build \
  --build-arg VITE_API_URL=http://custom-backend:8000 \
  --build-arg VITE_WS_URL=custom-backend:8000 \
  -t dream-ml-frontend:latest \
  .
```

### 4. Reduce Build Context

The included `.dockerignore` excludes:
- `node_modules/` (rebuilt in container)
- `dist/` (generated during build)
- `.git/`, IDE files, logs, etc.

This reduces build context and speeds up the build.

## Environment Variables

### Build-Time Variables (VITE_*)

These variables are **baked into the JavaScript bundle** at build time:

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `VITE_API_URL` | `http://backend:8000` | Backend API base URL |
| `VITE_WS_URL` | `backend:8000` | WebSocket server URL (without protocol) |
| `VITE_EXPERIMENTS_DIR` | `/app/experimentos` | Experiments directory path |

**Important Notes:**
- ⚠️ These values are **hardcoded** into the built JavaScript files
- ⚠️ Cannot be changed at runtime without rebuilding
- ✅ Perfect for docker-compose where service names are consistent
- ❌ If you need runtime configuration, a different approach is required (see [Runtime Configuration](#runtime-configuration) below)

### Setting Custom Values

**Via docker-compose.yml** (recommended):
```yaml
frontend:
  build:
    args:
      - VITE_API_URL=http://backend:8000
      - VITE_WS_URL=backend:8000
      - VITE_EXPERIMENTS_DIR=/app/experimentos
```

**Via command line**:
```bash
docker build \
  --build-arg VITE_API_URL=http://backend:8000 \
  --build-arg VITE_WS_URL=backend:8000 \
  --build-arg VITE_EXPERIMENTS_DIR=/app/experimentos \
  -t dream-ml-frontend:latest \
  .
```

### Runtime Configuration (Advanced)

If you need to change backend URLs at **runtime** without rebuilding:

1. Create a `public/config.js` file:
```javascript
window.ENV = {
  VITE_API_URL: 'http://backend:8000',
  VITE_WS_URL: 'backend:8000',
  VITE_EXPERIMENTS_DIR: '/app/experimentos'
};
```

2. Load it in `index.html`:
```html
<script src="/config.js"></script>
```

3. Use in your React code:
```javascript
const apiUrl = window.ENV?.VITE_API_URL || import.meta.env.VITE_API_URL;
```

4. Add a startup script to generate `config.js` from environment variables at container startup.

**Note**: This is more complex and not included in the current setup. The current configuration assumes docker-compose deployment with consistent service names.

## Troubleshooting

### Build Fails with "executor failed running..."

**Cause**: Buildx builder not running

**Solution**:
```bash
docker buildx inspect --bootstrap
```

### Cross-Platform Build is Very Slow

**Cause**: QEMU emulation is 10-50x slower than native builds

**Expected behavior**: Cross-platform builds can take 10-30 minutes

**Solution**:
- Build on native platform when possible
- Use CI/CD with platform-specific runners
- Use Option 3 (multi-platform manifest) with a registry

### "failed to solve with frontend dockerfile.v0"

**Cause**: Outdated Docker or Buildx version

**Solution**: Update Docker and Buildx:
```bash
docker buildx version
# Should be v0.10.0 or higher
```

Update Docker Desktop or install latest Docker Engine.

### npm ci fails during build

**Cause**: `package-lock.json` mismatch or missing

**Solution**:
```bash
# On your host, regenerate package-lock.json
cd DREAM-ML-frontend/frontend
npm install
git add package-lock.json
git commit -m "Update package-lock.json"

# Rebuild
docker-compose up -d --build frontend
```

### nginx: [emerg] host not found in upstream "backend"

**Cause**: Frontend container trying to connect to backend before it's ready

**Solution**: The current nginx config serves **static files only**. API calls are made from the browser using JavaScript.

If you see this error, ensure you're not proxying in nginx.conf. The current configuration does not proxy to backend.

### Build succeeds but frontend shows blank page

**Cause**: Missing environment variables during build

**Solution**: Ensure build args are passed:
```bash
docker-compose up -d --build frontend
```

Check build logs:
```bash
docker-compose build --no-cache frontend
```

Verify environment variables are set during build:
```bash
# Look for: "VITE_API_URL=http://backend:8000" in build output
```

### WebSocket connection fails

**Cause**: Incorrect `VITE_WS_URL` format

**Solution**: Ensure `VITE_WS_URL` is set to `backend:8000` (without `ws://` or `wss://` protocol prefix).

The code in [ExecutePipelineCard.jsx:96](src/components/ExecutePipelineCard.jsx#L96) and [ProgressBar.jsx:29](src/components/ProgressBar.jsx#L29) constructs the full WebSocket URL:

```javascript
const wsUrl = `${wsBaseUrl}/ws/progreso/`;
const ws = new WebSocket(wsUrl);
```

Or:
```javascript
const socket = new WebSocket(`ws://${import.meta.env.VITE_WS_URL}/ws/progreso/`);
```

So `VITE_WS_URL` should be just `backend:8000`.

### Frontend can't connect to backend API

**Possible causes:**
1. Backend not running
2. Incorrect `VITE_API_URL` during build
3. Network isolation in Docker

**Solution**:
```bash
# Verify backend is running
docker-compose ps

# Check backend logs
docker-compose logs backend

# Verify frontend build args
docker-compose config

# Rebuild with correct args
docker-compose up -d --build frontend
```

Test API connectivity from inside the frontend container:
```bash
docker exec -it dream-ml-frontend sh
wget -O- http://backend:8000/api/some-endpoint
```

### Image size larger than expected

**Expected size**: ~40-50MB

**Cause**: Build cache or incorrect Dockerfile

**Solution**:
```bash
# Rebuild without cache
docker build --no-cache -t dream-ml-frontend:latest .

# Check image size
docker images dream-ml-frontend:latest
```

If still large, ensure:
- Multi-stage build is working (check Dockerfile has two `FROM` statements)
- Final stage is `FROM nginx:alpine` (not `node:18-alpine`)

## Image Size Optimization

The optimized Dockerfile includes:
- ✅ Multi-stage builds (build deps removed from final image)
- ✅ nginx:alpine base image (~40MB)
- ✅ npm cache cleaned (`npm cache clean --force`)
- ✅ `.dockerignore` excludes `node_modules/`, `dist/`, `.git/`
- ✅ Gzip compression enabled in nginx
- ✅ Static assets served with caching headers

**Expected final image size**: ~40-50MB (down from ~400MB+ dev image)

Breakdown:
- nginx:alpine base: ~40MB
- Built React app (dist/): ~5-10MB
- nginx config: <1KB
- **Total**: ~40-50MB

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build Multi-Arch Frontend Image

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
          cd DREAM-ML-frontend/frontend
          docker buildx build \
            --platform linux/amd64 \
            --build-arg VITE_API_URL=http://backend:8000 \
            --build-arg VITE_WS_URL=backend:8000 \
            --build-arg VITE_EXPERIMENTS_DIR=/app/experimentos \
            -t dream-ml-frontend:latest-amd64 \
            --load \
            .

      - name: Build ARM64
        run: |
          cd DREAM-ML-frontend/frontend
          docker buildx build \
            --platform linux/arm64 \
            --build-arg VITE_API_URL=http://backend:8000 \
            --build-arg VITE_WS_URL=backend:8000 \
            --build-arg VITE_EXPERIMENTS_DIR=/app/experimentos \
            -t dream-ml-frontend:latest-arm64 \
            --load \
            .
```

## Security Considerations

1. **nginx security headers**: X-Frame-Options, X-Content-Type-Options, X-XSS-Protection, Referrer-Policy
2. **No secrets in image**: Use build args, not hardcoded secrets
3. **Regular updates**: Rebuild images monthly for security patches
4. **Health checks**: Automatic health monitoring via Docker
5. **Non-root nginx**: nginx:alpine runs as nginx user (not root)

### Scan for Vulnerabilities

```bash
docker scan dream-ml-frontend:latest
```

Or use Trivy:
```bash
trivy image dream-ml-frontend:latest
```

## Performance Benchmarks

### Build Performance

| Scenario | Time (Native) | Time (Cross-Platform) |
|----------|---------------|------------------------|
| First build (no cache) | 2-3 min | 10-30 min |
| Rebuild (cache hit) | 10-30 sec | 2-5 min |
| Code change only | 30-60 sec | 3-8 min |

### Runtime Performance

| Metric | nginx (Production) | npm dev (Development) |
|--------|-------------------|----------------------|
| Cold start | <1s | 3-5s |
| Request latency | <10ms | 50-200ms |
| Static asset serving | 10-100x faster | Baseline |
| Memory usage | ~10MB | ~100-200MB |
| Image size | ~40-50MB | ~400MB+ |

## Support and Resources

### Official Documentation
- **Docker Buildx**: https://docs.docker.com/buildx/
- **Multi-arch builds**: https://www.docker.com/blog/multi-arch-build-and-images-the-simple-way/
- **Vite Build**: https://vitejs.dev/guide/build.html
- **nginx**: https://nginx.org/en/docs/

### DREAM ML Resources
- **Backend Build Instructions**: [DREAM-ML-backend/GEML/BUILD_INSTRUCTIONS.md](../../DREAM-ML-backend/GEML/BUILD_INSTRUCTIONS.md)
- **docker-compose.yml**: [docker-compose.yml](../../docker-compose.yml)
- **.env.example**: [.env.example](../../.env.example)

### Common Issues
- For Docker/Buildx issues: https://docs.docker.com/buildx/
- For DREAM ML application issues: Contact project maintainers
- For nginx configuration: https://nginx.org/en/docs/

## Version History

- **v1.0** (2025-10-15): Initial production-optimized multi-arch Dockerfile
  - Multi-stage build (Node 18 + nginx:alpine)
  - Production build with Vite
  - nginx serving with SPA routing
  - Multi-platform support (linux/amd64, linux/arm64)
  - ~40-50MB final image size
  - Security headers and gzip compression
  - Health check endpoint
  - Comprehensive build instructions

## Quick Reference

### Essential Commands

```bash
# Build production image
docker build -t dream-ml-frontend:latest .

# Run standalone
docker run -d -p 5173:80 dream-ml-frontend:latest

# Build and run with docker-compose
docker-compose up -d

# Rebuild frontend only
docker-compose up -d --build frontend

# View logs
docker-compose logs -f frontend

# Stop and remove
docker-compose down

# Check image size
docker images dream-ml-frontend:latest

# Health check
curl http://localhost:5173/health
```

### Build Args Reference

```bash
--build-arg VITE_API_URL=http://backend:8000
--build-arg VITE_WS_URL=backend:8000
--build-arg VITE_EXPERIMENTS_DIR=/app/experimentos
```

---

**Happy building! 🚀**

For questions or issues, please contact the DREAM ML development team.
