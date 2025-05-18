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

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
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
        self.symptom_list = all_symptoms  # сохранить на будущее

        # One-hot кодирование симптомов
        for symptom in all_symptoms:
            self.df[symptom] = self.df["combined_symptoms"].apply(lambda x: 1 if symptom in x else 0)

        # Удаление временных колонок
        self.df.drop(columns=symptom_cols + ["combined_symptoms"], inplace=True)

        # Кодирование диагнозов
        le_target = LabelEncoder()
        self.df[self.target_column] = le_target.fit_transform(self.df[self.target_column])
        self.label_encoders[self.target_column] = le_target

        # X и y
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        # Масштабирование
        X_scaled = self.scaler.fit_transform(X)

        # Сплит
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=6878)

        print(f"[INFO] Предобработка завершена: {X.shape[1]} бинарных признаков.")

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_class_instance(self, filename="processed_dataset.csv"):
        output_path = os.path.join(self.file_path, filename)
        self.df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"[INFO] Обработанный CSV сохранён: {output_path}")


# def test():
#     preprocessed_data = DataHandler("data/dataset.csv")
#     preprocessed_data.load_data()
#     preprocessed_data.preprocess()
#     preprocessed_data.save_class_instance()


# test()