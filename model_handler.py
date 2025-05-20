# Copyright 2025 Abdiraimov Daniyar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import joblib
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

class ModelHandler:
    def __init__(self, model_type="mlp", model_path=None):
        self.model_type = model_type
        self.model_path = model_path
        self.initialize_model()
        self.model = None

    def initialize_model(self):
        if self.model_type.lower() == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=300,
                random_state=6878
            )
        elif self.model_type.lower() == "xgboost":
            self.model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=6878
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")

    def train(self, X_train, y_train):
        if self.model is None:
            self.initialize_model()
        self.model.fit(X_train, y_train)
        print(f"[INFO] Модель '{self.model_type}' обучена.")

    def predict(self, X_test):
        if self.model is None:
            raise RuntimeError("Модель не инициализирована.")
        return self.model.predict(X_test)

    def save_model(self, path):
        joblib.dump(self.model, path)
        self.model_path = path
        print(f"[INFO] Модель сохранена в: {path}")

    def load_model(self, path):
        try:
            self.model = joblib.load(path)
            self.model_path = path
            print(f"[INFO] Модель загружена из: {path}")
        except FileNotFoundError:
            print(f"[ERROR] Модель не найдена по пути {model_path}")
