from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from preprocessor import Preprocessor
import os
from datasets import load_dataset

app = FastAPI()

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
_ = preprocessor.train_preprocess(reviews_dirty)


# Load the pre-trained model
rfc = joblib.load("./models/model.pty")
class Review(BaseModel):
    text: str

@app.post("/predict/")
def predict(review: Review):
    review_normalized = preprocessor.preprocess(review.text)
    # Predict using the loaded model
    prediction = rfc.predict(review_normalized)
    # Return the prediction (0 or 1)
    return {"constructive": int(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=13337)
