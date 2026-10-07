# Depression Indicator

End-to-end machine learning application: raw survey data → eight trained classifiers → serialized artifacts → Flask REST API → Tkinter desktop GUI.

Most academic ML projects stop at the notebook. This one covers the full workflow from raw dataset to a user-facing application.

## What It Demonstrates

* Data preprocessing and feature engineering (imputation, one-hot encoding, ordinal mapping, MinMax scaling)
* Training and comparing 8 classification models plus a soft-voting ensemble
* Model serialization and reuse with `joblib`
* REST API deployment with Flask
* Desktop GUI integration for interactive predictions
* Reproducible transformation documentation via a generated audit file

## My Role

I independently developed the entire codebase. Classmates contributed ideas and conceptual feedback; I was solely responsible for the preprocessing pipeline, model selection and training, evaluation, serialization, the Flask API, the Tkinter GUI, and final integration.

The project was initially scaffolded with AI-assisted code generation. I modified, debugged, integrated, and structured the final implementation to meet the project's functional goals.

## Dataset

The **Student Depression Dataset** is a publicly available survey dataset on Kaggle. Each row is a self-reported survey response from a student, with a binary `Depression` target (`1`/`0`).

> **Dataset link:** TBD

The dataset contains **14 survey features**, including:

| Feature                          | Type / Range                                |
| -------------------------------- | ------------------------------------------- |
| Age                              | 18–59                                       |
| Gender                           | Male / Female                               |
| Degree                           | 28 categories (B.Tech, BSc, MBA, PhD, …)    |
| Academic Pressure                | 0–5 scale                                   |
| Study Satisfaction               | 0–5 scale                                   |
| Work Pressure                    | 0–5 scale                                   |
| Job Satisfaction                 | 0–4 scale                                   |
| CGPA                             | 0–10                                        |
| Work/Study Hours                 | 0–12 per day                                |
| Financial Stress                 | 1–5                                         |
| Sleep Duration                   | 4 bands (Less than 5 hrs → More than 8 hrs) |
| Dietary Habits                   | 3 bands (Unhealthy / Moderate / Healthy)    |
| History of Suicidal Thoughts     | Yes / No                                    |
| Family History of Mental Illness | Yes / No                                    |

**Rows:** TBD
Run `len(df)` on the raw CSV to determine the dataset size.

**Class balance:** TBD
Run `df["Depression"].value_counts(normalize=True)` to determine the class distribution.

## Architecture

```text
Student Depression Dataset.csv
            ↓
Preprocessing (model_prediction.py)
    ├── Median imputation
    ├── One-hot encoding
    ├── Ordinal mapping
    └── MinMax scaling
            ↓
Processed dataset + saved artifacts
    ├── ohe_general.pkl
    └── minmax_scaler.pkl
            ↓
Model Training (TrainModel.py)
    ├── 8 classifiers
    └── Soft-voting ensemble
            ↓
Serialized models (.pkl)
            ↓
Flask REST API (app.py)
    └── POST /predict
            ↓
Tkinter GUI (gui.py)
```

## The Pipeline

### 1. Preprocessing

**Script:** `model_prediction.py`

* Drops `id`, `City`, and `Profession` as identifiers/non-informative fields
* Performs median imputation for missing numeric values
* One-hot encodes:

  * Gender
  * History of Suicidal Thoughts
  * Family History of Mental Illness
  * Degree (28 categories)
* Produces **44 model features** after encoding
* Ordinally maps:

  * Sleep Duration → `0 / 0.33 / 0.66 / 1`
  * Dietary Habits → `0 / 0.5 / 1`
* MinMax-scales the 8 numeric features to `[0, 1]`
* Writes every transformation to `Conversion_Descriptions.csv`

  * Each encoding
  * Each scaler's minimum and maximum
  * The exact formula applied
* Serializes the encoders and scaler alongside the processed data, guaranteeing that training and inference apply identical transformations

### 2. Training

**Script:** `TrainModel.py`

* Uses an 80/20 train/test split with `random_state=42`
* Trains and evaluates 8 classification models:

  * Random Forest
  * Support Vector Machine (SVM)
  * Naive Bayes
  * Logistic Regression
  * K-Nearest Neighbors (KNN)
  * XGBoost
  * LightGBM
  * MLP Neural Network
* Trains an additional soft-voting ensemble using `VotingClassifier`
* Prints per-model accuracy and full classification reports
* Ranks all models based on performance
* Serializes every trained model to `.pkl` for reuse

### 3. Deployment

**Script:** `app.py`

* Provides a Flask REST API with a `POST /predict` endpoint
* Currently loads the **Logistic Regression** model
* The saved `VotingClassifier` can be substituted with a one-line change
* Replicates the training-time transformations on incoming JSON data
* Aligns feature columns against `model.feature_names_in_`
* Returns a JSON response in the following format:

```json
{
  "prediction": 0,
  "label": "..."
}
```

### 4. GUI

**Script:** `gui.py`

* Provides a Tkinter desktop interface mirroring the 14 survey features
* Sends user input to the local Flask API
* Displays the returned prediction

## Results

| Metric    | Logistic Regression (Deployed) |
| --------- | -----------------------------: |
| Accuracy  |                            TBD |
| Precision |                            TBD |
| Recall    |                            TBD |
| F1 Score  |                            TBD |

> **TBD:** Run `python TrainModel.py`. The script prints the accuracy and full classification report for every model. Copy the deployed model's metrics into the table above and note the best-performing model overall from the printed comparison.

The goal of this project was not clinical-grade diagnosis, but rather demonstrating applied machine learning pipeline construction and deployment readiness.

## Disclaimer

> This project is an academic machine learning exercise and is not a clinical diagnostic tool. It should not be used for medical decision-making.

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess the Dataset

Creates the processed CSV, encoder, scaler, and transformation audit file.

```bash
python model_prediction.py
```

### 3. Train the Models

Trains all 8 models plus the soft-voting ensemble, prints evaluation metrics, and saves the trained models as `.pkl` files.

```bash
python TrainModel.py
```

### 4. Start the API

```bash
python app.py
```

### 5. Launch the GUI

Open a second terminal and run:

```bash
python gui.py
```

> **Note:** Scripts use relative paths matching the repository's folder structure. Run each script from its appropriate directory.

## Project Layout

```text
Depression-Indicator/
├── Preprocessing/
│   ├── model_prediction.py
│   ├── raw CSV
│   ├── processed CSV
│   ├── ohe_general.pkl
│   ├── minmax_scaler.pkl
│   └── Conversion_Descriptions.csv
│
├── Trained_Model/
│   ├── TrainModel.py
│   └── saved .pkl models
│
├── <API folder>/
│   └── app.py
│
└── <GUI folder>/
    └── gui.py
```

> **TODO:** Confirm the exact folder names for the API and GUI directories before finalizing this section.

## Challenges & Lessons Learned

### Preprocessing Consistency Between Training and Inference

Training and inference must apply the exact same transformations to incoming data.

This was addressed by:

* Serializing the encoder and scaler alongside the model
* Reusing those artifacts during inference
* Generating `Conversion_Descriptions.csv` as an auditable record of the transformations

### Column Alignment at Inference Time

One-hot encoding can produce different columns depending on the input data.

The API addresses this by rebuilding the feature matrix against `model.feature_names_in_`, adding missing one-hot columns and dropping unexpected extras.

### Modular Application Structure

The project separates:

* Data preprocessing
* Model training
* Model inference
* REST API functionality
* User interface functionality

This made the project easier to test, debug, and extend.

### Moving Beyond the Notebook

A major goal of the project was translating theoretical machine learning concepts into working software that can be used through an API and desktop application rather than stopping at model training and evaluation.

## Future Improvements

* Cross-validation and hyperparameter tuning
* Input validation and range enforcement in the API
* Automated unit tests and CI
* Containerization with Docker
* Cloud deployment of the API
* Replacement of the Tkinter GUI with a web frontend
* More robust model evaluation and comparison
* Centralized preprocessing pipeline to further reduce training/inference inconsistencies

## License

TBD
