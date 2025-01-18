import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

class Evaluator:
    def __init__(self, rfc, X_test, y_test):
        self.rfc = rfc
        self.y_pred = self.rfc.predict(X_test)
        self.y_test = y_test

        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.conf_matrix = confusion_matrix(self.y_test, self.y_pred)

    def show_matrix(self, ):
        plt.figure(figsize=(10, 7))
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()