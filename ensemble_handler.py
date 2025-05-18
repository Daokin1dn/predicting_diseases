from collections import Counter
import numpy as np
from model_handler import ModelHandler

class EnsembleHandler:
    def __init__(self, all_predictions):
        self.all_predictions = all_predictions

   
    def predict_majority_vote(self):
        """
        Предсказывает класс на основе большинства голосов
        """

        self.all_predictions = np.array(self.all_predictions).T
        majority_votes = [Counter(row).most_common(1)[0][0] for row in self.all_predictions]
        return majority_votes
