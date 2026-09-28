<div align="center">

# Breast Cancer Classification

### A reproducible machine learning workflow with an interactive model comparison app

[![Python](https://img.shields.io/badge/Python-3.13-1F4E79?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20app-147D72?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)

</div>

---

## Project overview

This project explores two supervised classification approaches for breast tumour diagnosis:

| Model | Approach |
|:--|:--|
| **Logistic Regression** | A linear classifier with regularization |
| **K-Nearest Neighbors (KNN)** | A distance-based classifier using nearby training examples |

The workflow begins with shared data review and exploration, then builds and evaluates each model independently. A Streamlit app loads the saved model pipelines so users can try a manually entered or synthetic feature profile and compare the models' outputs.

> [!IMPORTANT]
> **For education and model comparison only.** This project is not a medical device and must not be used to diagnose, screen, or guide treatment. Model outputs are not clinical advice or an individual’s medical risk. Please consult a qualified healthcare professional about medical concerns.

## What the app does

- Guides users through 30 measurements in three groups: **Mean**, **Variation**, and **Largest**.
- Offers a synthetic profile option for interactive exploration.
- Shows the Logistic Regression and KNN predictions for the same profile after **Compute Output** is selected.
- Provides an optional visual summary of how the profile's largest measurement differences compare with training-set averages.
- Separately compares the models' errors on a held-out test set with known diagnoses.

The app uses “cancer predicted” and “no cancer predicted” for plain-language results, while keeping the dataset labels **malignant (M)** and **benign (B)** visible. Cancer is not an infection. A manual or synthetic profile has no known diagnosis, so the app cannot calculate whether that individual prediction was correct.

## Workflow

~~~mermaid
flowchart TD
    A[Raw diagnosis dataset] --> B[01 · Data overview]
    B --> C[02 · Data visualization]
    C --> D[Stratified data partitions]
    D --> E1[03 · Logistic Regression]
    D --> E2[03 · KNN]
    E1 --> F1[04 · Evaluate Logistic Regression]
    E2 --> F2[04 · Evaluate KNN]
    F1 --> G1[Saved Logistic Regression pipeline]
    F2 --> G2[Saved KNN pipeline]
    G1 --> H[Streamlit model studio]
    G2 --> H
    H --> I[Interactive predictions]
    H --> J[Held-out error comparison]
~~~

### Analysis sequence

1. **Review the dataset** — inspect its shape, feature types, target labels, and data quality.
2. **Explore the measurements** — visualize feature distributions and relationships.
3. **Build the Logistic Regression model** — select model settings using stratified cross-validation on the training partition.
4. **Build the KNN model** — follow the same partitioning and model-selection approach.
5. **Evaluate both models** — use the held-out test partition for final comparison.
6. **Explore with the web app** — load the saved pipelines and interact with both models.

The selected model configurations prioritize **malignant-class recall** during cross-validation. The test set is reserved for evaluation and is not used to fit or select the saved models.

## Held-out comparison

The app computes error counts from the shared, stratified **114-record test set**. The current saved models produced:

| Model | Wrong predictions | Missed malignant cases | False malignant alarms |
|:--|--:|--:|--:|
| Logistic Regression | 1 / 114 | 1 | 0 |
| KNN | 5 / 114 | 4 | 1 |

These are results on one fixed test split; they do not guarantee future performance. The test set contains a limited number of cases, and these figures should not be interpreted as clinical validation.

## Run the app with Docker

Docker packages the app, both saved model pipelines, their feature metadata, and the CSV used by the app. It **loads existing models** at startup; it does not train them.

The image repository is on [Docker Hub](https://hub.docker.com/repository/docker/surajchalise/breast-cancer-model-studio/general). After the image is pushed, pull and run it without building locally:

~~~powershell
docker pull surajchalise/breast-cancer-model-studio:latest
docker run --rm -p 8501:8501 surajchalise/breast-cancer-model-studio:latest
~~~

Then open [http://localhost:8501](http://localhost:8501).

### 1. Build the image

From the project root in PowerShell:

~~~powershell
docker build -t breast-cancer-demo:local .
~~~

### 2. Start the container

~~~powershell
docker run --rm --name breast-cancer-demo -p 127.0.0.1:8501:8501 breast-cancer-demo:local
~~~

### 3. Open the app

Visit [http://localhost:8501](http://localhost:8501). The app is bound to your own computer and is not published to the internet by this command. Stop the foreground container with **Ctrl+C** in the terminal.

If you prefer to run it in the background:

~~~powershell
docker run --rm -d --name breast-cancer-demo -p 127.0.0.1:8501:8501 breast-cancer-demo:local
docker stop breast-cancer-demo
~~~

After changing the app or its bundled files, rebuild the image and recreate the container:

~~~powershell
docker stop breast-cancer-demo
docker build -t breast-cancer-demo:local .
docker run --rm -d --name breast-cancer-demo -p 127.0.0.1:8501:8501 breast-cancer-demo:local
~~~

## Run the notebooks locally

The notebooks support the analysis and model-building workflow. In PowerShell, from the project root:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
~~~

Open notebooks in this order:

| Order | Notebook | Purpose |
|--:|:--|:--|
| 1 | [01 · Data overview](notebooks/01_data_overview.ipynb) | Inspect the dataset and target |
| 2 | [02 · Data visualization](notebooks/02_data_visualization.ipynb) | Explore and visualize measurements |
| 3 | [03 · Logistic Regression](notebooks/Logistic_Regression/03_model_building.ipynb) | Build the Logistic Regression pipeline |
| 4 | [04 · Logistic Regression evaluation](notebooks/Logistic_Regression/04_model_evaluation.ipynb) | Evaluate Logistic Regression |
| 5 | [03 · KNN](notebooks/KNN/03_model_building.ipynb) | Build the KNN pipeline |
| 6 | [04 · KNN evaluation](notebooks/KNN/04_model_evaluation.ipynb) | Evaluate KNN |

The first two notebooks are shared by both model tracks. Execute notebooks yourself to regenerate their saved outputs and artifacts.

> **App dependencies:** app-requirements.txt contains the pinned libraries used in the Docker image. **Notebook dependencies:** requirements.txt contains the data analysis and Jupyter libraries.

## Repository map

~~~text
.
├── app.py                         # Streamlit model studio
├── Dockerfile                     # Container build instructions
├── app-requirements.txt           # Pinned runtime dependencies
├── requirements.txt               # Notebook and analysis dependencies
├── configs/
│   └── config.yaml                # Model and feature configuration
├── data/
│   └── raw/
│       └── Raw_Breast_Cancer_Data.csv
├── models/
│   ├── Logistic_Regression/
│   │   └── logistic_regression_pipeline.joblib
│   └── KNN/
│       └── knn_pipeline.joblib
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_data_visualization.ipynb
│   ├── Logistic_Regression/
│   │   ├── 03_model_building.ipynb
│   │   └── 04_model_evaluation.ipynb
│   └── KNN/
│       ├── 03_model_building.ipynb
│       └── 04_model_evaluation.ipynb
└── reports/
    └── figures/
        ├── Logistic_Regression/   # Metadata, experiment log, evaluation outputs
        └── KNN/                   # Metadata, experiment log, evaluation outputs
~~~

## Data and modeling notes

- The target column is diagnosis: **B** = benign and **M** = malignant.
- The saved pipelines use the same ordered set of 30 numeric features.
- The record ID and empty Unnamed: 32 column are excluded from model inputs.
- The app's generated profiles are synthetic variations around a randomly selected training profile; they are not presented as real patient records.
- Inputs are processed in memory by the app and are not saved.
- The profile chart compares selected measurements with training-set averages. It is descriptive context, not a model explanation.
- A model score for the malignant class is not calibrated as an individual's medical risk.

## License and attribution

No license or dataset citation is declared in this repository. Add the appropriate license and original dataset attribution before redistributing this project or its data.
