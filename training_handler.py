import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
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
        X_train, X_test, y_train, y_test = self.data_handler.get_data()

        for model in self.models:
            # обучение модели
            model.fit(X_train, y_train)
            
            # Оценка модели на тестовых данных
            y_pred = model.predict(X_test)
            print(f"[INFO] Оценка модели {self.models.model_type} на тестовых данных:")
            print(classification_report(y_test, y_pred))

            # Сохранение модели
            model_path = f"{self.output_dir}/{self.models.model_type}_model.pkl"
            model.save_model(model_path)
            print(f"[INFO] Модель {self.models.model_type} сохранена по пути: {model_path}")
