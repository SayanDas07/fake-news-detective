import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
loaded_model = pickle.load(open('M:/fake-news-detection/trained_model.sav', 'rb'))
vectorizer = pickle.load(open('M:/fake-news-detection/vectorizer.sav', 'rb'))


new_text = ["Darrell Lucus House Dem Aide: We Didn’t Even S.."]

new_text_transformed = vectorizer.transform(new_text)


prediction = loaded_model.predict(new_text_transformed)


print("Prediction is: ", prediction[0])

if prediction[0] == 1:
    print("This is positive news.")
else:
    print("This is negative news.")
