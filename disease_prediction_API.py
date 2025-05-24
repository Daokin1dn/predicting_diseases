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

    def models_initializer(self, model_types=["MLP", "XGBoost"]):
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

    def predict(self, user_input=None, models_dir='models/'):
        self.prediction_handler = PredictionHandler(self.models, models_dir)

        if user_input is None:
            input_data = self.data_handler.X_test
        else:
            input_data = self.data_handler.transform_user_input(user_input)
            
        #print(input_data)

        predictions = self.prediction_handler.predict(input_data)
        readable_results = self.data_handler.recover_predictions(input_data, predictions)
        for item in readable_results:
            print("Симптомы:", ", ".join(item["symptoms"]))
            print("Диагноз:", item["disease"])
        return readable_results, predictions

    def evaluate(self, user_input=None, show_plots=False):
        if user_input is None:
            y_true = self.data_handler.y_test
            pred_re, y_pred = self.predict()
        else:
            y_true = self.data_handler.transform_user_input(user_input)
            pred_re, y_pred = self.predict(user_input)

        results = self.evaluation_handler.evaluate(y_true, y_pred)
        self.evaluation_handler.print_evaluation_report(results)
        if show_plots:
            self.evaluation_handler.plot_confusion_matrix(results["confusion_matrix"])
        return results

    def get_models(self):
        return self.models
