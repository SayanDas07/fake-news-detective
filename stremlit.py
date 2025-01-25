import numpy as np
import pickle
import streamlit as st

def local_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stTextInput > div > div > input {
        background-color: white;
        border: 2px solid #3498db;
        border-radius: 10px;
        padding: 10px;
        font-size: 1rem;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 1rem;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    .stSuccess {
        background-color: rgba(76, 175, 80, 0.1);
        color: #2c3e50;
        border-radius: 10px;
        padding: 15px;
        font-size: 1.1rem;
        text-align: center;
        border: 2px solid #4CAF50;
    }
    .stSuccess > div {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model and vectorizer
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))

def fake_news_prediction(new_text):

    new_text_transformed = vectorizer.transform([new_text])

    prediction = loaded_model.predict(new_text_transformed)
    print(prediction)

    return prediction[0]

def main():
    
    local_css()
    
 
    st.markdown('<h1 class="stTitle">🕵️ Fake News Detective</h1>', unsafe_allow_html=True)
    

    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; margin-bottom: 20px;'>
    Uncover the truth behind news articles with our advanced detection tool.
    </div>
    """, unsafe_allow_html=True)


    new_text = st.text_area("Enter News Content", 
                             height=200, 
                             placeholder="Paste the news article or headline here...",
                             help="Paste the news content you want to verify.")

    diagnosis = ''


    if st.button('Analyze News'):
        if new_text:
            
            result = fake_news_prediction(new_text)
            
            # Determine diagnosis with different styles
            if result == 1:
                diagnosis = "✅ Positive News (Real News)"
                st.success(diagnosis)
            else:
                diagnosis = "❌ Negative News (Fake News)"
                st.error(diagnosis)
        else:
            st.warning("Please enter news content to analyze.")

    

if __name__ == '__main__':
    main()