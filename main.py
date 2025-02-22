from preprocessor import Preprocessor 
from evaluator import Evaluator
from datasets import load_dataset
import joblib

# Random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    n_estimators = 100
    print("Number of trees:", n_estimators)
    model_path = "./models/model.pty"
    print("Base model path:", model_path)

    # !huggingface-cli login
    # !pip install datasets
    print("Loading dataset...")
    ds = load_dataset("abullard1/steam-reviews-constructiveness-binary-label-annotations-1.5k", "main_data")
    data = ds['base']

    print("Processing data...")
    reviews_dirty = []
    labels = []
    for row in data:
        reviews_dirty.append(row['review'])
        labels.append(row['constructive'])
    preprocessor = Preprocessor()
    vectors = preprocessor.train_preprocess(reviews_dirty)

    try:
        joblib.load(model_path)
        print("Model found at", model_path)
    except FileNotFoundError:
        print("Model not found at", model_path)

        print("Training model...")
        X_tfidf = vectors
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2)

        # Use 100 estimators and default values for other parameters
        rfc = RandomForestClassifier(n_estimators=n_estimators)
        rfc.fit(X_train, y_train)

        print("Saving model at", model_path)
        joblib.dump(rfc, model_path)

        print("Evaluating...")
        ev = Evaluator(rfc, X_test, y_test)
        print("Accuracy: ", ev.accuracy)
