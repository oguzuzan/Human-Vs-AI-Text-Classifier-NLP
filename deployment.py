import streamlit as st
import joblib

# Model
pipeline = joblib.load('pipeline_xgboost.joblib')

info_text = """
**Purpose:**
  - This application is designed to predict whether texts are written by humans or artificial intelligence.

**How it Works:**
  - Enter at least 20 words of text, press the "Predict" button, and see the result. It offers a simple, fast, and user-friendly experience.

**Features:**
  - **Quick Predictions:** Enter text, press the "Predict" button, and see instant results.
  - **Visual Support:** Prediction results present human and artificial intelligence probabilities with visual graphs. This provides a clear insight and simplifies the decision-making process.
"""


def main():
    # Page layout settings
    st.set_page_config(page_title="Human vs AI Text Classifier", page_icon="🤖")

    # Title
    st.title('Human vs AI Text Classifier(Only English)')
    st.markdown(info_text)

    input_text = st.text_area("Enter text (at least 20 words):", "", height=200)

    # Word count
    word_count = len(input_text.split())
    st.text(f"Word Count: {word_count}")

    # Prediction button
    if st.button('Predict'):
        # Check word count of the text
        word_count = len(input_text.split())
        if word_count < 20:
            st.error("Please enter at least 20 words.")
        else:
            # Making predictions
            predicted_class = pipeline.predict([input_text])
            predicted_probabilities = pipeline.predict_proba([input_text])

            # Display prediction results
            st.write("Prediction Results:")
            result = "HUMAN" if predicted_class[0] == 0 else "AI"
            st.info(f"**{result}**")

            st.write("Probability:")
            st.info(f"Probability of being Human: **{predicted_probabilities[0][0]:.2f}**")
            st.info(f"Probability of being AI: **{predicted_probabilities[0][1]:.2f}**")

            labels = ['Human', 'AI']
            probabilities = predicted_probabilities[0]
            st.bar_chart({labels[i]: probabilities[i] for i in range(len(labels))})

            input_text = ""
    

if __name__ == '__main__':
    main()
