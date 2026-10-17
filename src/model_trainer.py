import json
import time
from itertools import product
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support)


class ModelTrainer:
    def __init__(self, processed_dir='data/processed'):
        self.FEATURES = ['day_of_year', 'day_temp', 'humidity', 'cloud_cover', 'uv_index']

        self.train = pd.read_csv(processed_dir + '/weather_data_train.csv')
        self.test = pd.read_csv(processed_dir + '/weather_data_test.csv')
        self.val = pd.read_csv(processed_dir + '/weather_data_val.csv')

        self.X_train = self.train.iloc[:, :-1]
        self.y_train = self.train.iloc[:, -1]

        self.X_test = self.test.iloc[:, :-1]
        self.y_test = self.test.iloc[:, -1]

        self.X_val = self.val.iloc[:, :-1]
        self.y_val = self.val.iloc[:, -1]

    def get_candidates(self):
        depths = [3, 5, 7, 9, None]
        n_estimators = [100, 200, 300]
        min_samples_leaf = [1, 2, 5]
        max_features = ['sqrt', 1]

        candidates = {'rf': {}, 'dt': {}, 'et': {}}

        for depth, n_est, leaf, max_feat in product(depths, n_estimators, min_samples_leaf, max_features):
            suffix = f"_d{depth}_n{n_est}_leaf{leaf}_mf{max_feat}"
            candidates['rf']['rf' + suffix] = RandomForestClassifier(
                max_depth=depth, n_estimators=n_est, min_samples_leaf=leaf,
                max_features=max_feat, class_weight='balanced',
                random_state=42)
            candidates['et']['et' + suffix] = ExtraTreesClassifier(
                max_depth=depth, n_estimators=n_est, min_samples_leaf=leaf,
                max_features=max_feat, class_weight='balanced',
                random_state=42)

        for depth, leaf, max_feat in product(depths, min_samples_leaf, max_features):
            name = f"dt_d{depth}_leaf{leaf}_mf{max_feat}"
            candidates['dt'][name] = DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=leaf, max_features=max_feat,
                class_weight='balanced', random_state=42)

        return candidates

    def train_eval(self):
        candidates = self.get_candidates()
        best_per_family = {}
        all_rows = []

        for family, models in candidates.items():
            best_name, best_model, best_time, best_score = None, None, None, -1
            print(f"Tuning {family} ({len(models)} candidates)...")

            for model_name, model in models.items():
                start_time = time.time()
                model.fit(self.X_train, self.y_train)
                train_time = time.time() - start_time

                f1 = f1_score(self.y_val, model.predict(self.X_val), average='macro')
                all_rows.append({"family": family, "model": model_name,
                                 "validation_f1": f1, "train_time_seconds": train_time})

                if f1 > best_score:
                    best_name, best_model, best_time, best_score = model_name, model, train_time, f1

            print(f"  best: {best_name} (validation F1 = {best_score:.3f})")
            best_per_family[best_name] = (best_model, best_time, best_score)

        pd.DataFrame(all_rows).sort_values("validation_f1", ascending=False) \
            .to_csv("models/tuning_results.csv", index=False)

        return best_per_family

    def test_save(self):
        results = None
        model_dict = self.train_eval()
        best_model = None
        best_score = -1
        comparison_rows = []

        for model_name, (model, train_time, val_f1) in model_dict.items():
            y_pred = model.predict(self.X_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='macro', zero_division=0)
            accuracy = accuracy_score(self.y_test, y_pred)
            comparison_rows.append({
                "model": model_name,
                "validation_f1": val_f1,
                "test_f1": f1,
                "test_precision": precision,
                "test_recall": recall,
                "test_accuracy": accuracy,
                "train_time_seconds": train_time
            })

            if val_f1 > best_score:
                best_model = model
                best_score = val_f1
                results = {
                    'model': str(model),
                    'name': model_name,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'train_time': train_time,
                    'class_labels': model.classes_.tolist(),
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred, labels=model.classes_).tolist(),
                    'feature_ranking': dict(zip(self.FEATURES, model.feature_importances_.tolist()))
                }

        comparison = pd.DataFrame(comparison_rows)
        comparison.to_csv("models/model_comparison.csv", index=False)
        with open('models/metrics.json', 'w') as outfile:
            json.dump(results, outfile, indent=2)
        joblib.dump(best_model, 'models/best_model.pkl')
        joblib.dump(results, 'models/results.pkl')

        print("\n" + comparison.round(3).to_string(index=False))
        print(f"\nBest model: {results['name']}")

if __name__ == '__main__':
    model_trainer = ModelTrainer()
    model_trainer.test_save()