# Sepsis Prediction Pipeline

This repository contains a machine learning pipeline for **sepsis prediction**, designed to handle **missing data**, **class imbalance**, and **model training** in a structured way.  
The project follows a modular organization to make experimentation and scaling easier.

## 📁 Project Structure

### Folders Overview

- **data/**: Contains the dataset for sepsis prediction (`sepsis_dataset.csv`).
- **models/**: Stores the machine learning models (Random Forest, XGBoost, LightGBM).
- **pipelines/**: Includes scripts for training, validation, and data processing.
- **utils/**: Utility functions for preprocessing, feature engineering, and evaluation.

---

## 🧪 Parametrizable Pipeline

The pipeline is **fully parametrizable**, allowing you to configure the experiment, algorithm, and imputation strategy from a single YAML or dictionary configuration.  
Below is an example of a configuration file:

```yaml
experiment:
  name: "First GBDT"
run:
  name: "GBDT mean imputation strategy"
algorithm:
  training_algorithm: "GBDT"
imputation:
  strategy: "mean"
  fill_value: null
```
## How to Run
1. **Create python environment**:
   ```bash
   python -m venv venv
3. **Activate environment (Windows)**:
    ```bash
    .\venv\Scripts\activate
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
4. **Run the script (Make sure you have your .csv file at /data/raw folder):**
    ```bash
    python main.py
