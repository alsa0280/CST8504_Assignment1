import joblib
import pandas as pd

class Predictor:

    def __init__(self, model_path='models/best_model.pkl', scaler_path='models/scaler.pkl'):
        self.FEATURES = ['day_of_year', 'day_temp', 'humidity', 'cloud_cover', 'uv_index']

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, input_dict):
        input_df_unscaled = pd.DataFrame(input_dict, index=[0])[self.FEATURES]
        input_df = pd.DataFrame(self.scaler.transform(input_df_unscaled), columns=self.FEATURES)
        pred_label = self.model.predict(input_df)[0]
        class_probabilities = dict(zip(self.model.classes_,self.model.predict_proba(input_df)[0]))
        return pred_label, class_probabilities