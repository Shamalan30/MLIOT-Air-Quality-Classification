# 🌍 ML-IoT Air Quality Classification System

An end-to-end **Machine Learning and IoT air-quality classification system** that processes environmental data, trains an XGBoost classification model, evaluates its performance against a threshold-based baseline, and exposes the trained model through a **Flask REST API with ngrok** for IoT and Node-RED integration.

The system classifies environmental conditions into four air-quality categories:

> **Good · Moderate · Poor · Hazardous**

---

## 📌 Project Overview

This project demonstrates the integration of:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Data normalization
* Machine learning classification
* Model evaluation
* Cross-validation
* Feature importance analysis
* Model serialization
* Flask REST API deployment
* ngrok public tunneling
* Node-RED integration
* Simulated IoT sensor data replay

The complete workflow is implemented in a single Jupyter/Google Colab notebook:

```text
MLIOT_Project_1.ipynb
```

---

# 🎯 Objectives

The main objectives of this project are to:

1. Process and clean environmental air-quality data.
2. Explore relationships between environmental variables.
3. Engineer a combined `Pollution_Index` feature.
4. Train a machine learning model to classify air quality.
5. Compare machine learning performance against a simple threshold-based baseline.
6. Evaluate the trained model using accuracy, F1 score, classification reports, and confusion matrices.
7. Examine model stability using stratified 5-fold cross-validation.
8. Identify important input features using XGBoost feature importance.
9. Export the trained model and preprocessing objects.
10. Deploy the model through a Flask REST API.
11. Expose the API using ngrok.
12. Enable the API to receive environmental readings from an IoT/Node-RED workflow.

---

# 🏗️ System Architecture

```text
                Environmental Dataset
                         │
                         ▼
                Dataset Download
                         │
                         ▼
                  Data Loading
                         │
                         ▼
              Data Inspection & EDA
                         │
                         ▼
                 Data Cleaning
                         │
                         ▼
             Pollution Index Creation
                         │
                         ▼
                Feature Selection
                         │
                         ▼
              Label Encoding
                         │
                         ▼
            Stratified Train/Test Split
                    80% / 20%
                         │
                         ▼
                 Min-Max Scaling
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
    Threshold Baseline          XGBoost Model
             │                       │
             └───────────┬───────────┘
                         ▼
                 Model Evaluation
                         │
                         ▼
              Feature Importance
                         │
                         ▼
                  Model Export
                         │
                         ▼
                   Flask API
                         │
                         ▼
                      ngrok
                         │
                         ▼
                    Node-RED
                         │
                         ▼
             IoT Air Quality Output
```

---

# 📊 Dataset

The notebook downloads the dataset automatically from Google Drive using `gdown`.

The dataset contains:

* **5,000 rows**
* **10 original columns**

The original dataset contains the following variables:

| Feature                         | Description                              |
| ------------------------------- | ---------------------------------------- |
| `Temperature`                   | Environmental temperature                |
| `Humidity`                      | Environmental humidity                   |
| `PM2.5`                         | Fine particulate matter concentration    |
| `PM10`                          | Particulate matter concentration         |
| `NO2`                           | Nitrogen dioxide measurement             |
| `SO2`                           | Sulfur dioxide measurement               |
| `CO`                            | Carbon monoxide measurement              |
| `Proximity_to_Industrial_Areas` | Proximity-related environmental variable |
| `Population_Density`            | Population density variable              |
| `Air Quality`                   | Target air-quality class                 |

The target variable is:

```text
Air Quality
```

The four target classes are:

```text
Good
Moderate
Poor
Hazardous
```

---

# 📈 Dataset Class Distribution

The notebook reports the following class distribution:

| Air Quality Class |   Samples | Percentage |
| ----------------- | --------: | ---------: |
| Good              |     2,000 |        40% |
| Moderate          |     1,500 |        30% |
| Poor              |     1,000 |        20% |
| Hazardous         |       500 |        10% |
| **Total**         | **5,000** |   **100%** |

The dataset is therefore not evenly distributed across the four classes.

---

# 🔎 Exploratory Data Analysis

The notebook performs several exploratory analysis steps before model training.

## Data Inspection

The project examines:

* Dataset shape
* Column names
* Data types
* Descriptive statistics
* Missing values
* Duplicate rows
* Target class distribution

The dataset contains:

```text
5,000 samples
10 original columns
0 missing values
0 duplicate rows
```

---

## 📊 Feature Distribution Analysis

The notebook visualizes the distributions of the environmental features:

```text
Temperature
Humidity
PM2.5
PM10
NO2
SO2
CO
```

These plots are used to understand the distribution and range of the environmental variables before preprocessing.

---

## 🔗 Correlation Analysis

A correlation heatmap is generated to examine relationships between the numerical variables.

This provides a visual overview of relationships between:

* Temperature
* Humidity
* PM2.5
* PM10
* NO2
* SO2
* CO
* Other environmental variables
* Air-quality-related measurements

---

## 📦 Boxplot Analysis

The notebook also creates boxplots for selected key pollution-related features:

```text
PM2.5
PM10
NO2
CO
```

These visualizations help inspect differences and potential outliers across air-quality classes.

---

# 🧹 Data Cleaning

The notebook checks for physically questionable values before model training.

The following checks are performed:

| Variable | Cleaning Rule                       |
| -------- | ----------------------------------- |
| Humidity | Values above 100 are clipped to 100 |
| PM2.5    | Values below 0 are clipped to 0     |
| PM10     | Values below 0 are clipped to 0     |
| NO2      | Values below 0 are clipped to 0     |
| SO2      | Values below 0 are clipped to 0     |
| CO       | Values below 0 are clipped to 0     |

The rows are retained while invalid values are clipped to the defined limits.

After cleaning, the notebook retains:

```text
5,000 / 5,000 rows
```

---

# 🧮 Feature Engineering

A new feature called:

```text
Pollution_Index
```

is created from five pollution-related measurements.

The notebook calculates it using:

```text
Pollution_Index =
    0.35 × PM2.5
  + 0.25 × PM10
  + 0.20 × NO2
  + 0.10 × SO2
  + 0.10 × CO
```

The resulting value is rounded to four decimal places.

After feature engineering, the dataset contains:

```text
5,000 rows × 11 columns
```

The additional column is:

```text
Pollution_Index
```

---

# 🤖 Machine Learning

## Input Features

The machine learning model uses eight features:

```text
Temperature
Humidity
PM2.5
PM10
NO2
SO2
CO
Pollution_Index
```

The following original dataset variables are therefore not used directly by the classifier:

```text
Proximity_to_Industrial_Areas
Population_Density
```

---

# 🎯 Target Encoding

The categorical target variable:

```text
Air Quality
```

is encoded using `LabelEncoder`.

The encoding used by the notebook is:

| Class     | Encoded Value |
| --------- | ------------: |
| Good      |             0 |
| Hazardous |             1 |
| Moderate  |             2 |
| Poor      |             3 |

The encoder is saved as:

```text
label_encoder.pkl
```

---

# ✂️ Train/Test Split

The dataset is divided using a stratified train/test split.

```text
Training set: 80%
Testing set: 20%
```

This results in:

```text
Training samples: 4,000
Testing samples: 1,000
```

The split uses:

```python
random_state = 42
```

and preserves the class proportions through stratification.

---

# 📏 Feature Normalization

The project uses:

```python
MinMaxScaler
```

The scaler is fitted **only on the training data** and then applied to the test data.

This produces:

```text
X_train_scaled → 4,000 × 8
X_test_scaled  → 1,000 × 8
```

The fitted scaler is exported as:

```text
scaler.pkl
```

---

# 📐 Baseline Model

Before evaluating the machine learning model, the notebook creates a simple threshold-based baseline.

The baseline predicts air quality using only:

```text
Pollution_Index
```

The thresholds are:

| Pollution Index | Prediction |
| --------------: | ---------- |
|          ≤ 0.25 | Good       |
| > 0.25 – ≤ 0.50 | Moderate   |
| > 0.50 – ≤ 0.75 | Poor       |
|          > 0.75 | Hazardous  |

The baseline does not use:

```text
Temperature
Humidity
NO2
SO2
```

and relies on fixed threshold values.

### Baseline Performance

The notebook reports:

```text
Accuracy : 41.20%
F1 Score : 0.2602
```

---

# 🌲 XGBoost Model

The main machine learning model is:

```text
XGBClassifier
```

The configuration used in the notebook is:

```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)
```

The model is trained using:

```text
4,000 training samples
8 input features
```

---

# 📊 Model Evaluation

The trained XGBoost model is evaluated using:

* Accuracy
* Weighted F1 Score
* Classification Report
* Confusion Matrix

## Test Set Results

The notebook reports:

| Metric            |    XGBoost |
| ----------------- | ---------: |
| Accuracy          | **91.30%** |
| Weighted F1 Score | **0.9126** |

The test set contains:

```text
1,000 samples
```

---

# 📋 Classification Report

The notebook reports the following class-level results:

| Class     | Precision | Recall | F1 Score | Support |
| --------- | --------: | -----: | -------: | ------: |
| Good      |      0.99 |   0.99 |     0.99 |     400 |
| Hazardous |      0.82 |   0.76 |     0.79 |     100 |
| Moderate  |      0.92 |   0.93 |     0.92 |     300 |
| Poor      |      0.80 |   0.81 |     0.80 |     200 |

Overall:

```text
Accuracy = 0.9130
Weighted F1 = 0.9126
```

A confusion matrix is also generated by the notebook to visualize correct and incorrect classifications across the four classes.

---

# 🔄 Stratified 5-Fold Cross Validation

The notebook also performs stratified 5-fold cross-validation.

Configuration:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The reported fold results are:

### Accuracy

```text
Fold 1: 0.920
Fold 2: 0.922
Fold 3: 0.924
Fold 4: 0.935
Fold 5: 0.911
```

Mean:

```text
0.9224 ± 0.0077
```

### Weighted F1 Score

```text
Fold 1: 0.9199
Fold 2: 0.9216
Fold 3: 0.9233
Fold 4: 0.9350
Fold 5: 0.9097
```

Mean:

```text
0.9219 ± 0.0081
```

> **Note:** The notebook performs this cross-validation using the combined scaled training and testing feature data. Therefore, these results should be interpreted as an additional cross-validation analysis of the notebook's workflow rather than as an independent final test-set estimate.

---

# ⚖️ ML vs Baseline

The notebook compares the threshold-based baseline with XGBoost.

| Metric      | Baseline |    XGBoost |
| ----------- | -------: | ---------: |
| Accuracy    |   41.20% | **91.30%** |
| Weighted F1 |   0.2602 | **0.9126** |

The difference reported by the notebook is:

```text
Accuracy improvement: +50.10 percentage points
F1 improvement:       +0.6524
```

The comparison demonstrates how the trained model performs differently from the simple Pollution Index threshold approach used as the baseline.

---

# ⭐ Feature Importance

The notebook extracts feature importance values from the trained XGBoost model.

The reported ranking is:

| Feature         | Importance |
| --------------- | ---------: |
| CO              |     66.24% |
| NO2             |      8.49% |
| SO2             |      6.76% |
| Temperature     |      5.53% |
| Humidity        |      4.54% |
| PM10            |      3.15% |
| Pollution_Index |      2.97% |
| PM2.5           |      2.31% |

The notebook also generates a feature-importance visualization.

According to the trained model, `CO` has the largest feature-importance value in this particular training run.

Feature importance values are model-specific and should not automatically be interpreted as causal relationships.

---

# 💾 Model Export

The trained machine learning components are exported using `joblib`.

Three files are generated:

```text
model.pkl
label_encoder.pkl
scaler.pkl
```

### `model.pkl`

Contains the trained:

```text
XGBoost classifier
```

### `label_encoder.pkl`

Contains the encoder used to convert between:

```text
Good
Hazardous
Moderate
Poor
```

and their numerical representations.

### `scaler.pkl`

Contains the:

```text
MinMaxScaler
```

fitted on the training features.

These three files are required by the Flask API for inference.

---

# 🌐 Flask REST API

The trained model is exposed through a Flask API.

The API runs locally on:

```text
Port 5001
```

The notebook then creates a public ngrok tunnel.

---

# ❤️ Health Endpoint

The API provides:

```http
GET /health
```

The endpoint returns information including:

```json
{
  "status": "running",
  "model": "XGBoost - Air Quality Classifier",
  "classes": [...]
}
```

This endpoint can be used to check whether the API is running.

---

# 🔮 Prediction Endpoint

The main prediction endpoint is:

```http
POST /predict
```

It expects a JSON payload containing all eight model features.

Example:

```json
{
  "Temperature": 35.2,
  "Humidity": 78.5,
  "PM2.5": 85.3,
  "PM10": 120.4,
  "NO2": 45.6,
  "SO2": 18.2,
  "CO": 9.1,
  "Pollution_Index": 68.7
}
```

---

# 📤 API Response

The API returns:

* Prediction
* Confidence
* Class probabilities
* Alert level
* Alert color
* Recommendation

Example response structure:

```json
{
  "prediction": "Hazardous",
  "confidence": 0.996,
  "probabilities": {
    "Good": 0.0,
    "Hazardous": 0.996,
    "Moderate": 0.0,
    "Poor": 0.004
  },
  "alert": "DANGER",
  "color": "red",
  "recommendation": "Stay indoors. Activate air purifiers."
}
```

---

# 🚨 Alert Mapping

The Flask API maps the predicted class to an alert response.

| Prediction | Alert   | Color  | Recommendation                        |
| ---------- | ------- | ------ | ------------------------------------- |
| Good       | NORMAL  | Green  | No action needed.                     |
| Moderate   | CAUTION | Yellow | Reduce outdoor activities.            |
| Poor       | WARNING | Orange | Wear mask. Limit outdoor exposure.    |
| Hazardous  | DANGER  | Red    | Stay indoors. Activate air purifiers. |

These responses are the rules implemented directly in the Flask API.

---

# 🧪 API Test

The notebook tests the API using a simulated environmental reading:

```json
{
  "Temperature": 35.2,
  "Humidity": 78.5,
  "PM2.5": 85.3,
  "PM10": 120.4,
  "NO2": 45.6,
  "SO2": 18.2,
  "CO": 9.1,
  "Pollution_Index": 68.7
}
```

The recorded notebook test returned:

```text
Prediction     : Hazardous
Confidence     : 99.6%
Alert          : DANGER
Color          : red
Recommendation : Stay indoors. Activate air purifiers.
```

The probabilities reported for this particular test were:

```text
Good        : 0.0%
Hazardous   : 99.6%
Moderate    : 0.0%
Poor        : 0.4%
```

This is a single simulated API test and should not be interpreted as general model performance.

---

# 🔌 Node-RED Integration

The Flask API is designed to be integrated with Node-RED.

The intended workflow is:

```text
Environmental Sensor / Simulated Data
                │
                ▼
             Node-RED
                │
                ▼
       HTTP Request Node
                │
                ▼
         Flask /predict
                │
                ▼
        XGBoost Prediction
                │
                ▼
       JSON Response
                │
                ▼
         Node-RED Output
```

The notebook generates replay files that can be used to simulate an incoming stream of environmental readings.

---

# 📁 Node-RED Replay Data

The notebook creates:

```text
nodered_replay.csv
```

This contains the first 500 rows of the cleaned/preprocessed data.

It also creates:

```text
nodered_replay_v2.csv
```

The second replay file contains:

* Raw cleaned values
* `Pollution_Index`
* Normalized versions of the ML features

The normalized columns are created for:

```text
Temperature
Humidity
PM2.5
PM10
NO2
SO2
CO
Pollution_Index
```

This allows the replay data to contain both raw and normalized values for the IoT workflow.

---

# 📂 Project Structure

The repository can be organized as:

```text
MLIOT-Air-Quality-Classification/
│
├── MLIOT_Project_1.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

When the notebook is executed, additional generated files can include:

```text
dataset.csv
pollution_preprocessed.csv
nodered_replay.csv
nodered_replay_v2.csv
model.pkl
label_encoder.pkl
scaler.pkl
```

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Data Processing

* Pandas
* NumPy

## Data Visualization

* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn
* XGBoost

## Model Persistence

* Joblib

## Dataset Download

* gdown

## API

* Flask
* Requests

## Public API Tunnel

* pyngrok / ngrok

## IoT Integration

* Node-RED

## Development Environment

The notebook is configured for:

* Python 3
* Google Colab / Jupyter
* GPU-enabled Colab environment metadata

---

# 📦 Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
gdown
joblib
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
flask
pyngrok
requests
```

---

# ▶️ Running the Project

## Option 1 — Google Colab

The notebook is designed to work with Google Colab.

### Step 1

Upload:

```text
MLIOT_Project_1.ipynb
```

to Google Colab.

### Step 2

Run the notebook from the beginning.

### Step 3

The notebook downloads the dataset automatically.

### Step 4

Run the preprocessing and EDA sections.

### Step 5

Run the machine learning training and evaluation sections.

### Step 6

The notebook exports:

```text
model.pkl
label_encoder.pkl
scaler.pkl
```

### Step 7

Run the Flask deployment section.

### Step 8

The notebook creates an ngrok public URL.

### Step 9

Use the generated `/predict` URL in the Node-RED HTTP Request node.

---

# 🧪 Complete Workflow

The notebook is divided into three major stages.

## Stage 1 — Data Preprocessing

```text
Download Dataset
       ↓
Load Dataset
       ↓
Inspect Data
       ↓
Check Missing Values
       ↓
Check Duplicates
       ↓
Analyze Target Classes
       ↓
Feature Distributions
       ↓
Correlation Analysis
       ↓
Boxplot Analysis
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Export Preprocessed Data
```

---

## Stage 2 — Machine Learning

```text
Load Preprocessed Data
       ↓
Select Features
       ↓
Encode Target
       ↓
Train/Test Split
       ↓
Min-Max Normalization
       ↓
Baseline Prediction
       ↓
XGBoost Training
       ↓
Test Evaluation
       ↓
Confusion Matrix
       ↓
5-Fold Cross Validation
       ↓
Baseline Comparison
       ↓
Feature Importance
       ↓
Export Model
```

---

## Stage 3 — Deployment

```text
Load Trained Model
       ↓
Start Flask
       ↓
Create /health
       ↓
Create /predict
       ↓
Start ngrok Tunnel
       ↓
Test API
       ↓
Generate Node-RED Replay Data
```

---

# ⚠️ Model Limitations

The notebook explicitly identifies several limitations.

### 1. Static Training Dataset

The model is trained using a static dataset and may not generalize to all cities or regions.

### 2. No Automatic Retraining

There is currently no automatic retraining mechanism to handle sensor drift over time.

### 3. Model Interpretability

XGBoost does not provide simple row-by-row explanations of individual predictions.

### 4. Input Data Quality

Prediction quality depends on the quality of the incoming sensor data.

### 5. Dataset-Based Results

The reported accuracy and F1 scores are based on this particular dataset and experimental setup.

They should not be interpreted as proof that the system will achieve the same performance with real-world sensor data.

---

# 🔐 Security Notice

The notebook currently contains an **ngrok authentication token inside the deployment code**.

If this repository is going to be made public on GitHub, the exposed token should be **revoked and replaced before publication**.

Do not publish active authentication credentials in a public repository.

For a production implementation, credentials should instead be stored securely using environment variables or a dedicated secrets-management system.

The notebook itself has been kept unchanged to preserve the original project implementation.

---

# 🚧 Production Considerations

This project is primarily an ML-IoT demonstration and prototype workflow.

A production implementation would require additional considerations such as:

* Secure API authentication
* HTTPS configuration
* Production WSGI server
* Secure secret management
* Sensor authentication
* Input validation
* Rate limiting
* Monitoring
* Logging
* Model versioning
* Automatic model retraining
* Sensor calibration
* Data drift monitoring
* Persistent storage
* Production IoT infrastructure

The Flask development server used in the notebook should not be treated as a production deployment server.

---

# 📊 Project Results Summary

The notebook reports the following results:

| Measurement               |         Result |
| ------------------------- | -------------: |
| Dataset samples           |          5,000 |
| Original features/columns |             10 |
| ML features               |              8 |
| Training samples          |          4,000 |
| Testing samples           |          1,000 |
| Baseline Accuracy         |         41.20% |
| XGBoost Accuracy          |     **91.30%** |
| Baseline Weighted F1      |         0.2602 |
| XGBoost Weighted F1       |     **0.9126** |
| 5-Fold Mean Accuracy      | 92.24% ± 0.77% |
| 5-Fold Mean Weighted F1   | 92.19% ± 0.81% |
| XGBoost Trees             |            200 |

---

# 💡 Key Takeaway

This project demonstrates a complete pipeline from **environmental data to machine-learning prediction and IoT integration**.

The workflow combines:

```text
Data
+
Preprocessing
+
Feature Engineering
+
Machine Learning
+
Evaluation
+
Model Export
+
REST API
+
ngrok
+
Node-RED
```

The resulting system can receive environmental measurements, process them using the same preprocessing pipeline used during training, classify the air-quality condition, and return the prediction together with confidence, probabilities, alert information, and a recommendation.

---

# 👨‍💻 Author

**SHAMALAN A/L VELLU THAVAR**

**Matric Number:** A24AI0087

---

# 📄 Project File

Main notebook:

```text
MLIOT_Project_1.ipynb
```

This notebook contains the complete implementation of the project's:

* Data preprocessing
* Exploratory analysis
* Feature engineering
* Machine learning training
* Model evaluation
* Model export
* Flask API
* ngrok deployment
* API testing
* Node-RED replay data generation

---

# 📜 License

No open-source license is currently specified for this project.

Unless a license is added to the repository, the project's source code remains under the author's default copyright rights.
