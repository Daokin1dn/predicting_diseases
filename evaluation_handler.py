from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class EvaluationHandler:
    def __init__(self):
        pass

    def evaluate(self, y_true, y_pred):
        """
        y_true: истинные значения (метки классов)
        y_pred: предсказанные значения (результаты модели)
        """
        results = {}

        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted')
        results['recall'] = recall_score(y_true, y_pred, average='weighted')
        results['f1'] = f1_score(y_true, y_pred, average='weighted')

        # Метрика для бинарной классификации)
        if len(np.unique(y_true)) == 2:
            results['roc_auc'] = roc_auc_score(y_true, y_pred)

        # Матрица ошибок 
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        return results

    def plot_confusion_matrix(self, cm, class_names):
        """
        Визуализация матрицы ошибок с помощью seaborn.
        cm: матрица ошибок
        class_names: имена классов (для меток на оси)
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Предсказанные метки')
        plt.ylabel('Истинные метки')
        plt.title('Матрица ошибок')
        plt.show()

    def print_evaluation_report(self, results):
        print(f"Точность (Accuracy): {results['accuracy']:.4f}")
        print(f"Прецизионность (Precision): {results['precision']:.4f}")
        print(f"Полнота (Recall): {results['recall']:.4f}")
        print(f"F-метрика: {results['f1']:.4f}")
        if 'roc_auc' in results:
            print(f"AUC: {results['roc_auc']:.4f}")
        print("\nМатрица ошибок:")
        print(results['confusion_matrix'])
