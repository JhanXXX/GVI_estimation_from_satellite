# GeoAI-GVI Project

Automated AI pipeline for Green View Index (GVI) estimation using satellite data.
Check notebooks.00 for pipeline
Remember to restart the kernel if you have modified anything outside the board.... 

## Quick Start Guide

### Prerequisites

1. **Install VS Code**
   - Download from [code.visualstudio.com](https://code.visualstudio.com/)

2. **Install Docker Desktop**
   - Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
   - Make sure Docker is running before proceeding

3. **Install VS Code Dev Containers Extension**
   - Open VS Code
   - Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on Mac)
   - Search for "Dev Containers"
   - Install the **"Dev Containers"** extension by Microsoft

### Environment Setup

#### Step 1: Clone and Setup Project Structure
```bash
# Clone the repository
git clone <repository-url>
cd geoai-gvi-project

# Run the project setup script (creates all necessary files and directories)
chmod +x quick_setup.sh
./quick_setup.sh
```

#### Step 2: Build and Start Dev Container
```bash
# Open project in VS Code
code .

# Build and start the development container
# Press Ctrl+Shift+P (or Cmd+Shift+P on Mac)
# Type: "Dev Containers: Reopen in Container"
# Select the option and wait for container to build (~2-5 minutes)
```

#### Step 3: Install Dependencies (Inside Container)
Once the container is running, open the VS Code terminal and run:

```bash
# Install system dependencies for geospatial packages
chmod +x install_system_deps.sh
./install_system_deps.sh

# Install core Python packages
pip install -r requirements-core.txt

# Verify installation
python -c "import numpy, pandas, rasterio; print('Core packages installed!')"
```

#### Step 4: Install ML Dependencies (Optional)
```bash
# When ready for machine learning development
pip install -r requirements-ml.txt

# Or use the convenience script
./setup_scripts.sh install-ml
```

### Quick Verification

Test that everything is working:

```bash
# Test environment
./setup_scripts.sh test-env

# Test core modules
python src/sentinel_retrieval.py
python src/gvi_calculator.py

# Start Jupyter Lab
jupyter lab
# Access at http://localhost:8888
```

## Project Structure

```
workspace/
├── src/                           # Core Python modules
│   ├── sentinel_retrieval.py      # Direct point-based satellite data acquisition
│   ├── enhanced_gsv_retriever.py  # GSV directional image download
│   ├── gvi_calculator.py          # Ground truth GVI calculation
│   └── utils/                     # Utility functions
│       ├── config_loader.py       # Configuration management
│       ├── logger.py              # Logging utilities
│       └── spatial_utils.py       # Geospatial operations
├── notebooks/                     # Jupyter development and analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_sentinel_testing.ipynb
│   ├── 03_gsv_testing.ipynb      # GSV image download testing
│   ├── 04_gvi_calculation.ipynb  # GVI calculation experiments
│   └── experiments/
├── data/                          # All data files (Docker mounted)
│   ├── panorama/                  # Street view images
│   │   ├── metadata/              # GeoJSON metadata files
│   │   │   └── {country}/
│   │   │       └── {city}.json    # Panorama metadata
│   │   ├── base_maps/             # Tiff clips
│   │   │   └── {country}/{city}/raw_{buffer} # Referes to corresponding size of data;
│   │   │       │                             # The buffer is half of the geographical
│   │   │       │                             #     length of the clipped cell.
│   │   │       └── {PanoID}.tiff      # Bands: ["B02", "B03", "B04", "B05",
│   │   │                              #         "B06", "B07", "B08", "B11", "B12"]
│   │   │                              #          Resolution: 20m
│   │   └── previews/              # Downloaded images
│   │       └── {country}/{city}/
│   │           ├── CAoSLEner9OouLLEwhHGIA/          # panoid folder
│   │           │   ├── 0.jpg             # 0° direction (90° FOV)
│   │           │   ├── 90.jpg  
│   │           │   ├── 180.jpg  
│   │           │   └── 270.jpg          
│   │           └── 9W7iEfP4WP8YNSL2tKXijw/          # Another panoid
│   │               ├── 0.jpg
│   │               ├── 90.jpg  
│   │               ├── 180.jpg  
│   │               └── 270.jpg   
│   └── resource/                  # Input datasets
│       └── {country}/
│           └── {city}.shp         # Sampling location points
├── models/                        # Trained models and results
│   ├── checkpoints/               # Training checkpoints
│   └── final_models/              # Trained model files
├── logs/                          # Application logs
├── keys/                          # API keys (add your own)
│   └── gsv_keys.txt              # Google Street View API key
├── .devcontainer/                 # Docker configuration
└── docs/                          # Documentation
```
## Database Structure
used database: SQLite3
    """
    CREATE TABLE IF NOT EXISTS directional_gvi (
        panoid TEXT,
        city TEXT,
        country TEXT,
        method TEXT,
        total_directions INTEGER,
        gvi_average REAL,
        gvi_max REAL,
        gvi_min REAL,
        gvi_std REAL,
        directions_processed TEXT,
        config_headings TEXT,
        processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (panoid, city, country, method)
    )
    """
db_path = "/workspace/data/processed/GVI_training_labels.db
## Development Workflow

### 0. Generate point shp file
# Use shp_producer.py to prepare shp file

This module hasn't be integrated in the class files. Run and configure it independently, including for the rootpath setting.
Run src.shp_producer.py
Input can be downloaded from Geofabrik.de (.shp.zip). Use the layer named as something with "road".
Be cautious about directly using the original shp file without selecting a certain area in GIS (such as QGIS) softwares first.
Be careful with area size and point generating space, otherwise can result in a super huge point set.
Be sure about your {country} and {city} as names, which determines all the file in the lateral processes.

### 1. Panorama preview retrieving

```python
from src.gsv_panorama_retriever import download_gsv_panos
cities = [
    ("Denmark","Copenhagen"),
    ("Estonia","Tallinn"),
]
download_gsv_panos(cities)

# Existed panoID and (lat,lon) pair (tolerance 1e6) in the corresponding .json file will be escaped.
# Only those were successfully downloaded data are loaded into the .json file
# Heading set, year range, etc., can be configured from config.json
# retrieving metadata is free, but each time calling a heading image, costs one api response
# NB: change "batch_settings"["test_batch_size"] in config.json. This determins how many points are loaded from produced shp file of each city for checking metadata, e.g., actually downloaded panoramas might be smaller than this due to out of year range
# Dont put too large test_batch_size if you dont want to burn up your budget
```


### 2. Ground Truth GVI Computition Methods, Data Comparison and Quality Control
```python
# Use "pixel" instead of "semantic" if you dont want to waste your time
cities = [
    ("Denmark","Copenhagen"),
    ("Estonia","Tallinn"),
]
method = "pixel" # or "semantic" for DeepLab (much slower)
process_multiple_cities(cities, method)
# The results are inserted into a SQLite3 database file. Check and configure the path in config.json
```


### 3. Data Collection
```python
# Use direct point-based Sentinel-2A data retrieval. 
# For base downloading, get a relatively larger buffer, then the smaller-ranged data can be clipped from it

# Downloading reprocessed band dataa: config in config.json -> "rs_data"
# Configure buffer sizes in config.json
from src.sentinel_retrieval import SentinelDataRetriever
cities = [
    ("Denmark","Copenhagen"),
    ("Estonia","Tallinn")
    ]
download_ground_feature_rs_data(cities)

# Downloading raw band data: specify bands in config.json -> "raw_band_names"
# Configure buffer size in input
from src.sentinel_retrieval import download_raw_rs_data
cities = [
    ("Denmark","Copenhagen"),
    ("Estonia","Tallinn")
    ]
download_raw_rs_data(cities, buffer_size=600)

# Clip smaller buffers from current data:
# produce smaller sized buffers
from src.sentinel_retrieval import clip_buffer_data
target_buffer = 40
cities = [
    ("Denmark","Copenhagen"),
    ("Estonia","Tallinn"),
]
source_buffer = 200
method = "buffer" # for ground features, put "buffer"; for raw bands, put "raw"

print(f"Clipping buffer for: {target_buffer}")
clip_buffer_data(cities, method, source_buffer, target_buffer)
```

### 4. CNN Model Training

#### Quick Start
Configure the model settings in src.cnn_models_config.py
```python
cities = ["Stockholm",
          "Gothenburg",
          "Helsinki"
]
# test_city = ["Berlin", "Helsinki"] # if strategy is specified to be generalizability, test city must be included in cities
# if strategy is specified to be generalizability, the test dataset will select samples from those specified cities, while excluding them from training set and validation set.

test_model = "originla" # or: "simple", "micro", "resnet"
ground_features = ["NDVI", "EVI", "MSAVI", "GNDVI", "NDRE", "MNDWI", "UI", "BSI"] # excluding any of them for ablation study
raw_band_features = ["B02", "B03", "B04", "B05","B06", "B07", "B08", "B11", "B12"] # excluding any of them for ablation study
strategy = "train" # or "generalizability"
buffer_size = 40

results = run_cnn_training(
    cities=cities,
    buffer_size=buffer_size,
    input_features=ground_features, # choose whatever you like, but don't mix them up
    strategy=strategy,
    output_dir=f"/workspace/models/final_models/all_ground_features/{buffer_size}/{test_model}/{strategy}",
    training_params={"model_type": test_model}
)
# critical information will be loaded into result json file
```

### 5. GCN Model Training
Configure the graph structure and model settings in src.gcn_models_config.py

```python

cities = ["Stockholm",
          "Gothenburg",
          "Helsinki"
]
# test_city = ["Berlin", "Helsinki"] # if strategy is specified to be generalizability, test city must be included in cities
# if strategy is specified to be generalizability, the test dataset will select samples from those specified cities, while excluding them from training set and validation set.

test_model = "hierarchical_gcn" # "hierarchical_gat", "attention_pooling"
ground_features = ["NDVI", "EVI", "MSAVI", "GNDVI", "NDRE", "MNDWI", "UI", "BSI"] # excluding any of them for ablation study
raw_band_features = ["B02", "B03", "B04", "B05","B06", "B07", "B08", "B11", "B12"] # excluding any of them for ablation study
strategy = "train" # or "generalizability"
buffer_size = 40 # only use this one for ensuring 4*4 grid. Otherwise do more data engineering

results = run_gcn_training(
    cities=cities,
    buffer_size=buffer_size,
    input_features=input_features,
    strategy=strategy,
    output_dir=f"/workspace/models/final_models/{model}/{strategy}",
    training_params={# "test_city": test_city, if specified the strategy of generalizability
                    "model_type": test_model}
)
# run gcn_models_config.py to draw graph structure illustration
```

### 6. Model Application

TBD

## Configuration

### Environment Variables
Copy `.env.template` to `.env` and configure:
```bash
# API Keys
Google Street View API Key (stored in ./keys/gsv_keys.txt)
```

## Troubleshooting

### Container Build Issues
If container build fails:
```bash
# Clean Docker cache
docker system prune -f

# Rebuild container
# Ctrl+Shift+P → "Dev Containers: Rebuild Container"
```

### Network Issues During Package Installation
If apt-get fails in the container:
```bash
# Try switching networks (e.g., mobile hotspot)
# Or wait and retry later
sudo apt-get update && sudo apt-get install -y gdal-bin libgdal-dev
```

### GDAL Installation Problems
```bash
# Set environment variables manually
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
export GDAL_DATA=/usr/share/gdal

# Reinstall rasterio
pip install rasterio --no-cache-dir
```

### Permission Issues
```bash
# Fix cache permissions
sudo chown -R vscode:vscode /home/vscode/.cache
```

## Research Context

This project is part of the **Urban Green Fairness in Transportation** research initiative, focusing on:

- Direct point-based satellite GVI estimation methodology
- Comparison of mean-aggregation vs CNN approaches  
- Multi-scale analysis (200m-1000m buffer sizes)
- Open-source implementation for reproducible research

For detailed methodology and research contributions, see the project documentation.

---

**Note**: This setup uses VS Code Dev Containers for consistent development environments across team members. All dependencies and configurations are containerized for easy collaboration.