################################################################################
#                           DREAM ML - Installation Guide                     #
################################################################################

Version: v2.2.8
Last Updated: 2026-03-03

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
    docker pull tomasmanriquez480/dreaml-ml-backend:v2.2.8
    docker pull tomasmanriquez480/dreaml-ml-frontend:v2.2.8

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
     docker rmi tomasmanriquez480/dreaml-ml-backend:v2.2.8
     docker rmi tomasmanriquez480/dreaml-ml-frontend:v2.2.8
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
