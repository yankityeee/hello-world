from transformers import pipeline
import streamlit as st
from PIL import Image

# function part
def imgClassifier(image_name, modelName):
    # Load the age classification pipeline
    # The code below should be placed in the main part of the program
    age_classifier = pipeline("image-classification",
                              model=modelName)
    
    image_name = Image.open(image_name).convert("RGB")
    st.image(image_name, caption="Uploaded Image", use_column_width=True)
    
    # Classify age
    age_predictions = age_classifier(image_name)

    return age_predictions

def output_msg():
    # Display results
    st.write(age_predictions)
    age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)

    st.write(f"Age range: {age_predictions[0]['label']}")
    st.write("done")

def main():
    # Streamlit UI
    st.title("Age Classification using ViT")
    
    # age_predictions = imgClassifier("middleagedMan.jpg", "prithivMLmods/Age-Classification-SigLIP2")
    age_predictions = imgClassifier("middleagedMan.jpg", "dima806/fairface_age_image_detection")

    output_msg()

# main part
if __name__ == "__main__":
    main()
