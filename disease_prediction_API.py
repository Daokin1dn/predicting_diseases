from data_handler import DataHandler
from model_handler import ModelHandler
from training_handler import TrainingHandler
from prediction_handler import PredictionHandler
from evaluation_handler import EvaluationHandler

class DiseasePredictionAPI:
    def __init__(self, dataset_filepath, model_types):
        self.data_handler = DataHandler(dataset_filepath)
        self.evaluation_handler = EvaluationHandler()
        self.models = self.models_initializer(model_types)
        self.training_handler = None
        self.prediction_handler = None

    def models_initializer(seld, model_types={"MLP", "XGBoost"}):
        models = []
        for model_type in model_types:
            models.append(ModelHandler(model_type))
            print(models)            
        return models 

    def load_data(self):
        self.data_handler.load_data()
        self.data_handler.preprocess()
        return self.data_handler.get_data()

    def train(self, output_dir='models/'):
        self.training_handler = TrainingHandler(self.models, self.data_handler, output_dir)
        self.training_handler.train()
        return self.models

    def predict(self, input_data=None, models_dir='models/'):
        self.prediction_handler = PredictionHandler(self.models, models_dir)

        if input_data is None:
            input_data = self.data_handler.X_test
            print(self.data_handler.X_test)
        return self.prediction_handler.predict(input_data)

    def evaluate(self, y_true=None, y_pred=None, show_plots=False):
        if y_true is None or y_pred is None:
            y_true = self.data_handler.y_test
            y_pred = self.predict()
        results = self.evaluation_handler.evaluate(y_true, y_pred)
        self.evaluation_handler.print_evaluation_report(results)
        if show_plots:
            self.evaluation_handler.plot_confusion_matrix(results["confusion_matrix"], class_names=["Class 0", "Class 1"])
        return results

    def get_models(self):
        return self.models
