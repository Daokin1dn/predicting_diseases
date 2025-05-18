from model_handler import ModelHandler
from ensemble_handler import EnsembleHandler

class PredictionHandler:
    def __init__(self, models, models_dir='models/'):
        """
        models: экземпляры ModelHandler
        models_dir: Директория где хранятся обученные модели
        """
        self.models_dir = models_dir
        self.models = models

    def predict(self, input_data):
        """
         Метод для предсказания с выводом результата
        """
        all_predictions = []
        for model in self.models:
            if model.model_path is None:
                model_path = f"{self.models_dir}/{self.model.model_type}_model.pkl"
                model.load_model(model_path)
            prediction = model.predict(input_data)
            all_predictions.append(model.predict(input_data))
            print(f"[INFO] Результат предсказания для модели  {model.model_type}: {prediction}")
        
        majority_predictions = EnsembleHandler(all_predictions).predict_majority_vote()
        print(f"[INFO] Результат предсказания после ансамблирования: {majority_predictions}")
        return majority_predictions

#TODO доделать все и для CLI

# class PredictionHandler:
#     def __init__(self, models_dir='models/', models={}):
#         """
#         models: список экземпляров моделей
#         models_dir: директория, в которой хранятся обученные модели
#         """
#         self.models_dir = models_dir
#         self.models = models

#     def load_models(self, model_names):
#         """
#         Загружает модели из файлов, используя joblib
#         model_names: список названий моделей, которые нужно загрузить
#         """
#         for model_name in model_names:
#             model_path = f"{self.models_dir}/{model_name}_model.pkl"
#             try:
#                 model = joblib.load(model_path)
#                 self.models[model_name] = model
#                 print(f"[INFO] Модель {model_name} успешно загружена из {model_path}")
#             except FileNotFoundError:
#                 print(f"[ERROR] Модель {model_name} не найдена по пути {model_path}")

#     def predict(self, model_name, input_data):
#         """
#         Использует загруженную модель для предсказания
#         model_name: название модели, которую нужно использовать для предсказания
#         input_data: данные, на которых будет сделано предсказание
#         """
#         if model_name not in self.models:
#             raise ValueError(f"[ERROR] Модель {model_name} не загружена.")
        
#         model = self.models[model_name]
#         prediction = model.predict(input_data)
#         return prediction

#     def make_prediction(self, model_name, input_data):
#         """
#         Метод для предсказания с выводом результата
#         """
#         prediction = self.predict(model_name, input_data)
#         print(f"[INFO] Результат предсказания для модели {model_name}: {prediction}")
#         return prediction
