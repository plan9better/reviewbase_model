import re
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessor:
    def __init__(self):
        pass
    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())  # Handle all white space

        # Additional cleaning steps
        text = re.sub(r'\d+', 'NUM', text)  # Replace numbers with token
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Normalize repeated chars

        # Handle common abbreviations
        abbreviations = {
            "isn't": "is not", "don't": "do not", "its": "it is", "won't": "will not", 
            "can't": "cannot", "you're": "you are", "i'm": "i am", "i'll": "i will", 
            "i've": "i have", "i'd": "i would", "you'll": "you will", "you've": "you have", 
            "you'd": "you would", "he's": "he is", "she's": "she is", "it's": "it is", 
            "we're": "we are", "they're": "they are", "that's": "that is", "there's": "there is", 
            "there're": "there are", "isnt": "is not",
        }
        for abbr, expanded in abbreviations.items():
            text = text.replace(abbr, expanded)

        return text

    def _train_vectorize(self, reviews_normalized: list):
        n_features = 5000

        self.vectorizer = TfidfVectorizer(
            token_pattern=r'(?u)\b\w\w+\b',  # Default pattern: matches words of 2+ chars
            lowercase=True,  # Converts text to lowercase before tokenizing
            strip_accents='unicode'  # Removes accents
        )
        self.vectorizer = TfidfVectorizer(max_features=n_features)
        vectors = self.vectorizer.fit_transform([review for review in reviews_normalized])
        return vectors

    def _vectorize(self, review_normalized):
        vectors = self.vectorizer.transform([review_normalized]).toarray()
        return vectors

    def train_preprocess(self, reviews_dirty: list):
        reviews_normalized = []
        for review in reviews_dirty:
            reviews_normalized.append(self._normalize(review))
        
        reviews_vectorized = self._train_vectorize(reviews_normalized)
        return reviews_vectorized

    def preprocess(self, review_dirty: str):
        review_normalized = self._normalize(review_dirty)
        review_vectorized = self._vectorize(review_normalized)
        return review_vectorized

