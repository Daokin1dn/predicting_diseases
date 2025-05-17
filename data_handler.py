import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataHandler:
    def __init__(self, file_path, target_column):
        self.file_path = file_path
        self.target_column = target_column
        self.df = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print(f"[INFO] Данные загружены: {self.df.shape[0]} строк, {self.df.shape[1]} столбцов.")

    def preprocess(self):
        self.df.dropna(inplace=True)

        # Кодирование категориальных признаков
        for col in self.df.select_dtypes(include=['object']).columns:
            if col != self.target_column:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        X_scaled = self.scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=6878)

        print(f"[INFO] Данные успешно предобработаны и разделены на обучающую и тестовую выборки.")

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

