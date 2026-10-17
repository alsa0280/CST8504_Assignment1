# Weather Day Type Prediction Dashboard

CST8504 Assignment 1

GitHub repo: https://github.com/alsa0280/CST8504_Assignment1

## Authors

- Hadi Al-Sayed Ali (041181821)
- Danish Manro (041291839)

## Summary

This project predicts the type of day (Sunny, Cloudy, Rainy, Hot or Cold) from five weather measurements. It includes a data processing pipeline, tuned and compared classification models, and a Streamlit dashboard that makes interactive predictions with the best-performing model.

## Running the project (Docker)

```
docker build -t weather_dashboard .
docker run --rm -p 8501:8501 --name dashboard weather_dashboard
```

Then open http://localhost:8501.

## Data

The raw dataset is in `data/raw/weather_data.csv` and contains 1,500 day records.

- **Features:** `day_of_year`, `day_temp` (degrees Celsius), `humidity` (%), `cloud_cover` (%) and `uv_index`
- **Target:** `day_type`

In `src/data_processor.py`:

- missing feature values are imputed with column medians, and unlabelled rows are dropped
- duplicate rows are removed
- training (70%), validation (15%) and test (15%) sets are created with a stratified split
- a `MinMaxScaler` is fitted on the training set and saved to `models/scaler.pkl`
- the three sets are saved to `data/processed/`

## Models

`src/model_trainer.py` tunes three classifiers, `DecisionTreeClassifier`, `RandomForestClassifier` and `ExtraTreesClassifier`, on combinations of the following parameters:

- `max_depth`
- `n_estimators` (random forest and extra trees only)
- `min_samples_leaf`
- `max_features`

This gives 210 configurations in total. The best configuration of each family is chosen by macro F1 on the validation set, and the three finalists are then compared once on the test set. The best model, a random forest, is saved to `models/best_model.pkl`, and `src/predictor.py` loads it to make predictions in the dashboard.

## Dashboard

The Streamlit dashboard has four tabs:

- **Data Overview:** dataset size, a preview of the raw data and summary statistics.
- **Exploration:** filter by day type and feature ranges, then explore class counts (bar chart), average feature values over the year (line plot) and any two features against each other (scatter plot).
- **Prediction:** set the five weather values and click "Predict" to see the predicted day type and the probability of each class.
- **Model Performance:** the best model's metrics, a comparison of the three model families, a confusion matrix and feature importances.

## Project Structure

```
├── dashboard.py              Streamlit dashboard
├── src/
│   ├── data_processor.py     Cleaning, splitting, scaling
│   ├── model_trainer.py      Tuning, evaluation, saving
│   └── predictor.py          Loads the model and scaler, predicts new inputs
├── notebooks/
│   ├── Preprocessing.ipynb   Data inspection, missing values, duplicates, outliers
│   └── EDA.ipynb             Exploratory data analysis
├── data/
│   ├── raw/                  Original dataset
│   └── processed/            Train, validation and test sets
├── models/
│   ├── best_model.pkl        Trained best model
│   ├── scaler.pkl            Fitted MinMaxScaler
│   ├── metrics.json          Best model's test results (used by the dashboard)
│   ├── results.pkl           Copy of the best model's results
│   ├── model_comparison.csv  Best model of each family, compared on the test set
│   └── tuning_results.csv    Validation F1 of all 210 configurations
├── Dockerfile
├── .dockerignore
└── requirements.txt
```
