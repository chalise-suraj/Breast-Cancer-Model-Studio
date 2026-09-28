# Breast Cancer Prediction — Docker demo

This Streamlit research demo loads the saved Logistic Regression and KNN pipelines. The **Make a prediction** tab lets a user enter the 30 measurements or generate a synthetic feature vector by perturbing a random training-set profile. Predictions appear only after the user selects **Compute Output**. Synthetic/manual values have no known diagnosis, so they can show model predictions and agreement but not per-input error.

The separate **Compare model errors** tab reports each saved model's measured false negatives, false positives, and total errors on the shared 114-record held-out test split.

**Research use only. Not for diagnosis or treatment decisions.** “Malignant” and “benign” are retained with plain-language descriptions; cancer is not an infection.

## Build and run with Docker Desktop

From the repository root in PowerShell, build the image:

```powershell
docker build -t breast-cancer-demo:local .
```

Run the app on your own computer:

```powershell
docker run --rm --name breast-cancer-demo -p 127.0.0.1:8501:8501 breast-cancer-demo:local
```

Open [http://localhost:8501](http://localhost:8501). Stop the app with `Ctrl+C` in the terminal, or run `docker stop breast-cancer-demo` if it is detached. The image includes saved model pipelines, matching metadata, and the source CSV; it does not train models when the app starts.
