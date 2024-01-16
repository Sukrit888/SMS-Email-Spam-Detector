import streamlit as st
from model_load import load_model
from nlp_helpers import transform_text
from PIL import Image
from os import getcwd

# Model Path
model_path = getcwd() + "/models"

# IMG Path
img_path = getcwd() + "/img"

# Load Models
tfidf = load_model(f'{model_path}/vectorizer.pkl')
model = load_model(f'{model_path}/mnb_model.pkl')

st.markdown("<h1 style='text-align:center'> Email/SMS Spam Classifier </h1>", unsafe_allow_html=True)


# Create Two Tab Layout
tab_1, tab_2 = st.tabs(['Introduction', 'Make Predictions'])

with tab_1:
    st.markdown("<h2 style='text-align:center'> Project Motivation </h3>", unsafe_allow_html=True)

    # Get Image as BytesIO
    img_io = Image.open(f'{img_path}/spam_not_spam_img.png')
    st.image(img_io, use_column_width=True)

    st.write("In this project, I put my machine learning skills to the test by leveraging techniques in NLP to create features for classifying whether or not input text can be bucketed into spam or not spam by using several supervised classifier models.")

with tab_2:
    input_sms = st.text_area("Enter the message: ")

    # Column Layout
    col_1, col_2 = st.columns(2, gap='medium')

    if st.button('Predict'):

        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0][0]
        probability = round(probability*100, 2)

        with col_1:
            # Display Label
            if result == 1:
                st.error("Spam")
            else:
                st.success("Not Spam")
        
        with col_2:
            # Display Probability
            st.success(f'Model Confidence: {probability}%')
        
        