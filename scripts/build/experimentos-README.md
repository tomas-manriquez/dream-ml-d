# Experimentos Directory

This directory is used by DREAM ML to store all experiment-related data, including:

- **Datasets**: CSV files and data uploaded for analysis
- **Experiment Configurations**: Pipeline configurations and parameters
- **Results**: Generated reports, visualizations, and model outputs
- **Artifacts**: Intermediate files created during ML workflows

## How It Works

This directory is mounted directly from your host machine into the Docker container at `/app/experimentos`. This means:

- ✓ You can **directly access** all experiment files from your file manager
- ✓ Files created by DREAM ML appear **immediately** in this folder
- ✓ You can **copy files in/out** without using Docker commands
- ✓ Data **persists** even if containers are stopped or removed

## Directory Structure

DREAM ML will automatically create subdirectories as needed:

```
experimentos/
├── <experiment_name>/
│   ├── data/              # Raw and processed datasets
│   ├── artifacts/         # Pipeline configs, logs
│   └── eda_reports/       # Exploratory data analysis reports
└── ...
```

## Accessing Files

### From File Manager
- **macOS**: Open Finder → Navigate to the dream-ml folder → Open `experimentos/`
- **Windows**: Open File Explorer → Navigate to the dream-ml folder → Open `experimentos\`
- **Linux**: Open your file manager → Navigate to the dream-ml folder → Open `experimentos/`

### From Command Line
```bash
# List all experiments
ls experimentos/

# View a specific experiment
ls experimentos/<experiment_name>/

# Copy files out
cp experimentos/<experiment_name>/data/results.csv ~/Desktop/
```

## Troubleshooting

### Permission Issues (Linux Only)
If you encounter permission errors on Linux:

```bash
# Fix ownership (run from dream-ml directory)
sudo chown -R $USER:$USER ./experimentos
chmod -R 775 ./experimentos
```

### Can't See Files
If files aren't appearing:
1. Refresh your file manager
2. Check that the container is running: `docker compose ps`
3. Verify the mount with: `docker compose exec backend ls /app/experimentos`

### Backup Your Data
To backup your experiments:
```bash
# Create a backup
tar -czf experiments-backup-$(date +%Y%m%d).tar.gz experimentos/

# Or simply copy the folder
cp -r experimentos/ experimentos-backup/
```

## Support

For more information, see the main README.txt or visit the project documentation.
