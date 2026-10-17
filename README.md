

\# Weather Day Type Prediction Dashboard



CST8504 Assignment 1



GitHub repo link: https://github.com/alsa0280/CST8504\_Assignment1











\## Authors

Hadi Al-Sayed Ali (041181821)



Danish Manro (041291839)



\## Summary



This project predicts the type of day (Sunny, Cloudy, Rainy, Hot, or Cold) from five weather measurements. It includes a data processing pipeline, tuned and compared classification models, and a Streamlit dashboard with interactive prediction capabilities with the best-performing model. 

\## Running the project (Docker)

```

docker build -t weather\_dashboard .

docker run --rm -p 8501:8501 --name dashboard weather\_dashboard

```

Then open ```http://localhost:8501/```. Change the port number if it's already taken.



\## Data



The raw dataset can be found in ```data/raw/weather\_data.csv``` and contains 1,500 day records. 

\* \*\*Features\*\*: ```day\_of\_year```, ```day\_temp``` in degrees celcius, ```humidity``` percentage, ```cloud\_cover``` percentage, and ```uv\_index```

\* \*\*Target\*\*: ```day\_type```



In ```src/data\_processor.py```, 

\* the raw dataset's missing values are imputed with column medians

\* duplicate rows are removed

\* outliers are checked within reasonable ranges (with impossible values removed entirely)

\* training (70%), testing (15%), and validation (15%) splits are cre ated

\* a ```MinMax``` scaler is fitted on the training set and saved to ```models/scaler.pkl```

\* the three sets are saved to ```data/processed```

\## Models



```src/model\_trainer.py``` tunes three classification model families on combinations of the following parameters:

\* ```max\_depth```

\* ```n\_estimators```

\* ```min\_samples\_leaf```

\* ```max\_features```



This results in a total of 210 configurations of the three families. The three classification classes are ```DecisionTreeClassifier```, ```RandomForestClassifier```, and ```ExtraTreesClassifier```. 



The models were compared by their macro F1 score on the validation set, and the saved model, a random forest, was saved to ```src/predictor.py``` for usage in dashboard predictions.

\## Dashboard



The Streamlit dashboard has four tabs:



\*\*Data Overview\*\*: dataset size, a preview of the raw data, and summary statistics.



\*\*Exploration\*\*: filter by day type and feature ranges, explore class counts (bar chart), average feature values over a year (line plot), and any two features against each other (scatter plot).



\*\*Prediction\*\*: set the five weather values and click "Predict" to see the predicted day type and the probability of each class.



\*\*Model Performance\*\*: the best model's metrics, a comparison of the three model families, a confusion matrix, and a feature ranking.

\## Project Structure

```

├── dashboard.py              

├── src/

│   ├── data\_processor.py     cleaning, splitting, scaling

│   ├── model\_trainer.py      tuning, evaluation, saving

│   └── predictor.py          loads model + scaler, predicts new inputs

├── notebooks/

│   ├── Preprocessing.ipynb   Data inspection, missing values, duplicates, outliers

│   └── EDA.ipynb             Exploratory data analysis

├── data/

│   ├── raw/                  Original dataset

│   └── processed/            Train, validation and test sets

├── models/

│   ├── best\_model.pkl        

│   ├── scaler.pkl            Fitted MinMaxScaler

│   ├── metrics.json          Best model's test results (used by the dashboard)

│   ├── model\_comparison.csv  Best model of each family, compared on the test set

│   └── tuning\_results.csv    Validation F1 of all 210 configurations

├── Dockerfile

├── .dockerignore

└── requirements.txt



```

