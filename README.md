# MLManthan - Anomaly Detection Dashboard

A Streamlit-based anomaly detection application using **Isolation Forest** and **One-Class SVM** machine learning models to identify outliers in multivariate datasets.

## Overview

MLManthan processes `.dat` files (numeric data) and detects anomalies using two complementary algorithms:

- **Isolation Forest**: Fast, scalable tree-based anomaly detector
- **One-Class SVM (OC-SVM)**: Kernel-based outlier detection

Results include:
- Anomaly scores and classifications from both models
- Interactive visualizations (scatter plots, histograms)
- Downloadable results as CSV
- Alert notifications when anomalies are detected

## Features

✨ **Dual Detection**: Combine predictions from two algorithms for robust anomaly detection  
📊 **Interactive Dashboard**: Built with Streamlit for easy exploration  
📈 **Visualizations**: PCA-reduced scatter plots and anomaly distribution charts  
🔔 **Audio Alerts**: Alert sound when anomalies are detected  
📥 **CSV Export**: Download results for further analysis  
⚡ **Pre-trained Models**: Uses pre-trained `.pkl` models (no retraining needed)  

## Project Structure

```
MLManthan/
├── app.py                          # Main Streamlit dashboard application
├── train.py                        # Training script for models (optional)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── iso_forest_10batch.pkl          # Pre-trained Isolation Forest model
├── ocsvm_model_10batch.pkl         # Pre-trained One-Class SVM model
├── scaler_10batch.pkl              # Feature scaler (StandardScaler)
├── Dataset/                        # Sample data directory
└── myvenv/                         # Python virtual environment
```

## Installation

### 1. Clone or download this project

```bash
cd C:\MLManthan
```

### 2. Create and activate virtual environment

**PowerShell:**
```powershell
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1
```

**Bash/Linux:**
```bash
python -m venv myvenv
source myvenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the Dashboard

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Dashboard

1. **Upload Data**: Click "Browse files" and select a `.dat` file
   - File must contain only numeric values
   - Supported delimiters: spaces, commas, tabs
   - No header row required

2. **View Results**:
   - **Metrics**: Anomaly counts by detector type
   - **Anomaly Scores**: Numerical scores (0.5–1.0 scale) from each model
   - **Predictions**: Boolean classifications (anomaly = True/False)
   - **Charts**: Interactive visualizations of detected anomalies

3. **Download Results**: 
   - Click "Download Results" to export predictions as CSV

## Data Format

Input `.dat` files should contain:
- **Only numeric values** (floats or integers)
- **No headers**
- **Space, comma, or tab-delimited**

Example `data.dat`:
```
1.2 3.4 5.6 7.8
2.1 3.2 4.3 5.4
10.1 20.2 30.3 40.4
```

## Pre-trained Models

The project includes three pre-trained pickle (`.pkl`) files:

| File | Description |
|------|-------------|
| `iso_forest_10batch.pkl` | Isolation Forest model trained on 10 batches |
| `ocsvm_model_10batch.pkl` | One-Class SVM model trained on 10 batches |
| `scaler_10batch.pkl` | StandardScaler fit during training |

### Note
All input data is automatically scaled using `scaler_10batch.pkl` before predictions to ensure compatibility with model training parameters.

## Training New Models

To retrain models on your own data:

1. Place `.dat` files in `Dataset/` directory (e.g., `batch1.dat`, `batch2.dat`, etc.)

2. Run the training script:
   ```bash
   python train.py
   ```

3. New model files will be saved as `.pkl` files in the project root

**Configuration** (in `train.py`):
- `DATA_DIR`: Path to `.dat` files
- `BATCH_RANGE`: Range of batch files to load (default: batches 1–7)

## Output

### Results CSV
Contains the following columns:
- `index`: Row number from input file
- `iso_value`: Isolation Forest anomaly score (0.5–1.0)
- `iso_pred_outlier`: Isolation Forest prediction (True=anomaly)
- `oc_value`: OC-SVM anomaly score (0.5–1.0)
- `oc_pred_outlier`: OC-SVM prediction (True=anomaly)
- `combined_outlier`: Combined prediction (True if either model flags as anomaly)

### Visualizations
- **Anomaly Distribution**: Bar chart of counts by detection method
- **Scatter Plots**: PCA-projected data with anomalies highlighted
- **Histograms**: Score distributions for each model

## Dependencies

- **streamlit** ≥1.39 – Web framework for dashboard
- **pandas** ≥2.2 – Data manipulation
- **numpy** ≥1.26 – Numerical computing
- **scikit-learn** ≥1.3 – Machine learning models
- **plotly** ≥5.18 – Interactive visualizations

See `requirements.txt` for exact versions.

## Troubleshooting

### Import Error: `No module named 'streamlit'`
```bash
pip install -r requirements.txt
```

### File Not Found: Missing `.pkl` models
Ensure `iso_forest_10batch.pkl`, `ocsvm_model_10batch.pkl`, and `scaler_10batch.pkl` exist in the project root directory.

### Data Parse Error
- Verify file contains only numeric values
- Check delimiter (space, comma, or tab)
- Remove any header rows

### Streamlit Not Opening
Try:
```bash
streamlit run app.py --logger.level=debug
```

## Performance Notes

- **Fastest for**: Small to medium datasets (< 100k rows)
- **Scaling**: For large datasets, consider batch processing or distributed frameworks
- **Memory**: All data is loaded into memory; check RAM before processing large files

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions, verify:
1. All `.pkl` files are present
2. Dependencies are installed (`pip install -r requirements.txt`)
3. Python version is 3.8 or newer
4. Input data is in correct format (numeric, delimited)

---

**Last Updated**: November 21, 2025