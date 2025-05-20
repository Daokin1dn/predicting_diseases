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

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

class DataHandler:
    def __init__(self, file_path, target_column="Disease"):
        self.file_path = file_path
        self.target_column = target_column
        self.df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.symptom_list = []
        self.disease_df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(type(self.df))
            print(f"[INFO] Данные загружены: {self.df.shape[0]} строк, {self.df.shape[1]} столбцов.")
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки файла: {e}")

    def preprocess(self):
        # Заполнение пустых значений
        self.df.fillna('', inplace=True)

        # Объединение симптомов
        symptom_cols = [col for col in self.df.columns if col.startswith("Symptom_")]
        self.df["combined_symptoms"] = self.df[symptom_cols].apply(
            lambda row: [sym.strip().lower() for sym in row if sym.strip() != ''], axis=1
        )

        # Все уникальные симптомы
        all_symptoms = sorted(set(sym for sublist in self.df["combined_symptoms"] for sym in sublist))
        self.symptom_list = all_symptoms  

        # One-hot кодирование симптомов
        for symptom in all_symptoms:
            self.df[symptom] = self.df["combined_symptoms"].apply(lambda x: 1 if symptom in x else 0)

        diseases_name = self.df[self.target_column].unique()
        # Удаление временных колонок
        self.df.drop(columns=symptom_cols + ["combined_symptoms"], inplace=True)
        # Кодирование диагнозов
        le_target = LabelEncoder()
        self.df[self.target_column] = le_target.fit_transform(self.df[self.target_column])
        self.label_encoders[self.target_column] = le_target

        # X и y для сплит
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        diseases = self.df[self.target_column].unique()
        # Масштабирование
        X_scaled = self.scaler.fit_transform(X)

        # Сплит
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.01, random_state=6878)

        print(f"[INFO] Предобработка завершена: {X.shape[1]} бинарных признаков.")

        # DataFrame с индексами и названиями для диагнозов
        self.disease_df = pd.DataFrame({
            'code': diseases,
            'disease': diseases_name
        })
        self.save_class_instance()
        self.save_diseases_map()

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_class_instance(self, filename="processed_dataset.csv"):
        output_path = os.path.join(os.path.dirname(self.file_path), filename)
        self.df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"[INFO] Обработанный CSV сохранён: {output_path}")
    
    def save_diseases_map(self, filename="diseases_map.csv"):
        output_path = os.path.join(os.path.dirname(self.file_path), filename)
        self.disease_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"[INFO] CSV для диагнозов сохранён: {output_path}")


    def load_processed(self, proc_dataset_name="processed_dataset.csv", diseases_map_name="diseases_map.csv" ):
        """
        Загружает ранее предобработанный и сохранённый CSV.
        Восстанавливает все необходимые компоненты.
        """
        try:
            proc_dataset_path = os.path.join(os.path.dirname(self.file_path), proc_dataset_name)
            self.df = pd.read_csv(proc_dataset_path)
            diseases_map_path = os.path.join(os.path.dirname(self.file_path), diseases_map_name)
            self.disease_df = pd.read_csv(diseases_map_path)

            # Восстановление списка симптомов: все бинарные столбцы, кроме Disease
            self.symptom_list = [col for col in self.df.columns if col != self.target_column and
                                self.df[col].dropna().value_counts().index.isin([0,1]).all()]

            # Восстановление LabelEncoder для болезни
            le_target = LabelEncoder()
            self.df[self.target_column] = le_target.fit_transform(self.df[self.target_column])
            self.label_encoders[self.target_column] = le_target

            # Разделение признаков и цели
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]

            # Восстановление StandardScaler (fit, затем transform)
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Сплит
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y, test_size=0.01, random_state=6878)

            print(f"[INFO] Загружен обработанный CSV: {proc_dataset_path}")
            print(f"[INFO] Кол-во симптомов: {len(self.symptom_list)}")
            print(f"[INFO] Загружен CSV диагнозов: {diseases_map_path}")

        except Exception as e:
            print(f"[ERROR] Ошибка загрузки обработанных данных: {e}")


    def recover_predictions(self, X_pred_scaled, y_pred):
        """
        Восстановить исходные симптомы и диагнозы из масштабированных и предсказанных данных.
        :param X_pred_scaled: масштабированные one-hot признаки (обычно это X_test или X для предсказания)
        :param y_pred: массив предсказанных классов (чисел)
        :return: список словарей: [{symptoms: [...], disease: "..."}, ...]
        """
        
        if not self.symptom_list:
            self.load_processed()
        X_original = self.scaler.inverse_transform(X_pred_scaled)

        readable_data = []
        for i, row in enumerate(X_original):
            symptoms = [self.symptom_list[j] for j, val in enumerate(row) if round(val) == 1]
            disease = self.label_encoders[self.target_column].inverse_transform([y_pred[i]])[0]
            readable_data.append({
                "symptoms": symptoms,
                "disease": disease
            })
        #TODO надо сделать вывод названии диагноза 
        return readable_data

    def transform_user_input(self, user_symptoms):
        """
        Преобразует список симптомов пользователя в масштабированный вектор признаков.
        :param user_symptoms: список строк, например ["fatigue", "nausea"]
        :return: массив [1 x N] в формате np.array([[...]])
        """
        if not self.symptom_list:
            self.load_processed()

        # Нормализация
        input_set = set([sym.lower().strip() for sym in user_symptoms])

        input_vector = [1 if symptom in input_set else 0 for symptom in self.symptom_list]

        # Преобразование
        input_scaled = self.scaler.transform([input_vector])  # список -> [1 x N] массив
        return input_scaled



# def test():
#     preprocessed_data = DataHandler("data/dataset.csv")
#     preprocessed_data.load_data()
#     preprocessed_data.preprocess()
#     preprocessed_data.save_class_instance()


# test()