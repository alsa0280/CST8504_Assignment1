import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support
from src.data_processor import DataProcessor
import joblib
import time
import json

#print(classification_report(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))

class ModelTrainer:
    def __init__(self, processed_dir = 'data/processed'):
        self.FEATURES = ['day_of_year', 'day_temp', 'humidity', 'cloud_cover', 'uv_index']

        self.train = pd.read_csv(processed_dir + '/weather_data_train.csv')
        self.test = pd.read_csv(processed_dir + '/weather_data_test.csv')
        self.val = pd.read_csv(processed_dir + '/weather_data_val.csv')

        self.X_train = self.train.iloc[:,:-1]
        self.y_train = self.train.iloc[:,-1]

        self.X_test = self.test.iloc[:,:-1]
        self.y_test = self.test.iloc[:,-1]

        self.X_val = self.val.iloc[:,:-1]
        self.y_val = self.val.iloc[:,-1]

    def get_candidates(self):
        depths = [3,5,7,9]
        base_names = ["rf", "dt"]
        candidates = {'rf':{}, 'dt':{}}
        for base_name in base_names:
            for depth in depths:
                model_name = base_name + "_" + str(depth)
                if base_name == "rf":
                    model = RandomForestClassifier(max_depth=depth, class_weight='balanced', random_state=42)
                    candidates['rf'][model_name] = model
                else:
                    model = DecisionTreeClassifier(max_depth=depth, class_weight='balanced', random_state=42)
                    candidates['dt'][model_name] = model
        #print(candidates)
        return candidates

    def train_eval(self):
        candidates = self.get_candidates()

        best_model_rf = None
        best_name_rf = None
        best_train_time_rf = None
        best_score_rf = -1

        best_model_dt = None
        best_name_dt = None
        best_train_time_dt = None
        best_score_dt = -1

        # tune dt
        for dt_model_name in candidates['dt']:
            dt_model = candidates['dt'][dt_model_name]
            start_time = time.time()
            dt_model.fit(self.X_train,self.y_train)
            end_time = time.time()
            train_time = end_time - start_time
            f1 = f1_score(self.y_val, dt_model.predict(self.X_val), average='macro')
            if f1 > best_score_dt:
                best_score_dt = f1
                best_model_dt = dt_model
                best_name_dt = dt_model_name
                best_train_time_dt = train_time

        # tune rf
        for rf_model_name in candidates['rf']:
            rf_model = candidates['rf'][rf_model_name]
            start_time = time.time()
            rf_model.fit(self.X_train, self.y_train)
            end_time = time.time()
            train_time = end_time - start_time
            f1 = f1_score(self.y_val, rf_model.predict(self.X_val), average='macro')
            if f1 > best_score_rf:
                best_score_rf = f1
                best_model_rf = rf_model
                best_name_rf = rf_model_name
                best_train_time_rf = train_time
        return {best_name_dt:(best_model_dt, best_train_time_dt), best_name_rf:(best_model_rf,best_train_time_rf)}

    def test_save(self):
        results = None
        model_dict = self.train_eval()
        best_model = None
        best_score = -1

        for model_name in model_dict:
           model = model_dict[model_name][0]
           y_pred = model.predict(self.X_test)
           precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='macro')
           accuracy = accuracy_score(self.y_test, y_pred)
           if f1 > best_score:
               best_model = model
               best_score = f1
               results = {
                       'model': str(model),
                       'name': model_name,
                       'f1': f1,
                       'precision': precision,
                       'recall': recall,
                       'accuracy': accuracy,
                       'train_time': str(model_dict[model_name][1]) + " seconds",
                       'confusion_matrix': confusion_matrix(self.y_test, y_pred, labels=best_model.classes_).tolist(),
                       'feature_ranking': dict(zip(self.FEATURES, best_model.feature_importances_.tolist()))
               }
        with open('models/metrics.json', 'w') as outfile:
            json.dump(results, outfile)
        joblib.dump(best_model, 'models/best_model.pkl')
        joblib.dump(results, 'models/results.pkl')
        print(results)

if __name__ == '__main__':
    model_trainer = ModelTrainer()
    model_trainer.test_save()
