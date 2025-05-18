import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from model_handler import ModelHandler
from data_handler import DataHandler
import numpy as np

class TrainingHandler:
    def __init__(self, models, data_handler, output_dir='models/'):
        """
        models: экземпляры моделей, если их будет больеш двух {'MLP', 'XGBoost', '???'}
        data_handler: экземпляр класса DataHandler с загруженными и подготовленными данными
        output_dir: директория для сохранения обученных моделей
        """
        self.data_handler = data_handler
        self.output_dir = output_dir
        self.models = models

    def train(self):
        """
        Процесс обучения всех моделей
        """
        for model in self.models:
            # обучение модели
            model.train(self.data_handler.X_train, self.data_handler.y_train)
            
            # Оценка модели на тестовых данных
            y_pred = model.predict(self.data_handler.X_test)
            print(f"[INFO] Оценка модели {model.model_type} на тестовых данных:")
            print(classification_report(self.data_handler.y_test, y_pred))

            # Сохранение модели
            model_path = f"{self.output_dir}/{model.model_type}_model.pkl"
            model.save_model(model_path)
            print(f"[INFO] Модель {model.model_type} сохранена по пути: {model_path}")
