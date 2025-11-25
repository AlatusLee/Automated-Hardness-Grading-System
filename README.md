Automated Hardness Grading Evaluation System
An automated hardness grading evaluation system based on unsupervised learning, utilizing pressure experiment data including torque, direction, and Paxini sensor data to automatically assess hardness and generate 9×11 hardness grids.

![Alt Text](https://gitee.com/alatus-lee/Automated-Hardness-Grading-Evaluation-System/raw/master/Figure_1.png)
![Alt Text](https://gitee.com/alatus-lee/Automated-Hardness-Grading-Evaluation-System/raw/master/Figure_2.png)

System Overview
This system analyzes pressure experiment data (including 6-axis force sensor data, position data, and 239 Paxini contact point data) using unsupervised learning algorithms to automatically classify hardness into 4 grades and generate visual 9×11 hardness score grids for each sample.

Key Features
Unsupervised Learning: No pre-labeled data required, automatically discovers hardness patterns

Multi-Feature Fusion: Combines mechanical features, Paxini statistical features, and torque features

Intelligent Clustering: Automatically maps meaningful hardness grades based on stiffness features

Grid Visualization: Generates intuitive 9×11 hardness distribution grids

Real-time Prediction: Supports real-time data monitoring and prediction

Batch Processing: Processes multiple sample files at once

System Architecture

Automated-Hardness-Grading-System/
├── core_processor.py     # Core data processing and model training
├── main.py              # Main control program
├── realtime_predictor.py # Real-time prediction module
├── config.py            # Configuration file
├── data92/
│   └── data926/         # Data file directory (12 CSV samples)
├── models/              # Trained model storage directory
├── results/             # Output results directory
└── 指腹L5325 PX6AX-GEN3-CP-L5325-Omega PXSR-STDCP03A.xlsx  # Coordinate file

Data Format
Each CSV file contains:

First 6 columns: 6-axis force sensor data (XYZ forces + XYZ torques)

Next 3 columns: Position data

Next 4 columns: Quaternion data

Remaining 239 columns: Paxini sensor contact point data

Installation Requirements

pip install pandas numpy scipy scikit-learn matplotlib openpyxl

Usage
1. Environment Check

python main.py
Select option 4: Check data environment

2. Offline Training

Select option 1: Offline training model


The system will:

Process all CSV sample files

Extract features and train clustering model

Generate hardness grades and 9×11 grids

Save models and visualization results

3. Prediction Modes
Option 2: Real-time prediction (monitor new files)

Option 3: Batch prediction for all files

Output Results
After training, generated in results directory:

hardness_assessment_results.csv - Sample hardness grades

{sample_name}_hardness_grid_9x11.csv - 9×11 grid for each sample

all_samples_hardness_grids.png - Visualization of all samples

feature_importance.png - Feature importance ranking

clustering_info.txt - Detailed clustering information

Configuration Parameters
Adjust in config.py:

NUM_CLUSTERS: Number of hardness grades (default 4)

Data paths and file locations

Sensor data column indices

Visualization parameters

Technical Details
Algorithm: KMeans clustering + feature standardization

Feature Engineering: Statistical feature extraction (avoiding dimensionality curse)

Hardness Mapping: Intelligent remapping based on stiffness features

Grid Generation: 9×11 regular grid based on coordinate interpolation

License
MIT License

Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

Contact
For questions and support, please open an issue in the GitHub repository.
