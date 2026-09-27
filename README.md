# Breast Cancer Prediction — Docker demo

This Streamlit research demo loads the saved Logistic Regression and KNN pipelines. It offers manual entry for the 30 numeric measurements and random labeled examples from the reserved test split. Manual entries show both predictions and malignant-class model scores; correctness cannot be known without a true diagnosis. Random labeled examples show whether each prediction matches that record's known diagnosis. The comparison table reports both models on the shared 114-record test split.

**Research use only. Not for diagnosis or treatment decisions.**

## Build and run with Docker Desktop

From the repository root in PowerShell, build the image:

```powershell
docker build -t breast-cancer-demo:local .
```

Run the app on your own computer:

```powershell
docker run --rm --name breast-cancer-demo -p 127.0.0.1:8501:8501 breast-cancer-demo:local
```

Open [http://localhost:8501](http://localhost:8501). Stop the app with `Ctrl+C` in the terminal. The image includes the saved model pipelines, matching metadata, and source CSV; it does not install or train models at startup.
