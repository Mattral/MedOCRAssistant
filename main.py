import streamlit as st
import easyocr
import numpy as np
from huggingface_hub import InferenceClient
import random
import time
from PIL import Image
import pdfplumber  # pdfplumber for PDF extraction
from io import BytesIO
from googletrans import Translator  # For translation

# Initialize EasyOCR and Huggingface InferenceClient
reader = easyocr.Reader(['en'])
model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
access_token = "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"
client = InferenceClient(model=model, token=access_token)

# Embedded system prompt for AI
system_prompt_text = (
    "You are Doctor's Assistant reporting to patient"
)

# Function to process and send OCR extracted text to AI
def get_medical_ai_response(ocr_text, history=None):
    if history is None:
        history = []
    prompt = f"This information is the OCR extracted from Medical Document or Report: {ocr_text}\n\nPlease identify if this is medical related, if it is medical explain in details and if it's not, provide a summary of the content in less than 50 words."
    
    # Send the OCR text to the AI model
    history, output = chat_inf(prompt, history, random.randint(1, 1111111111111111), 0.9, 3840, 0.9, 1.0)
    return output, history

# Function to generate AI responses
def chat_inf(prompt, history, seed, temp, tokens, top_p, rep_p):
    generate_kwargs = dict(
        temperature=temp,
        max_new_tokens=tokens,
        top_p=top_p,
        repetition_penalty=rep_p,
        do_sample=True,
        seed=seed,
    )

    formatted_prompt = format_prompt_mixtral(prompt, history)

    for attempt in range(5):
        try:
            stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
            output = ""
            for response in stream:
                output += response.token.text

            if not output:
                return history, "No response."

            history.append((prompt, output))
            return history, output
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return history, "An error occurred during processing."

# Function to format the prompt for Mixtral model
def format_prompt_mixtral(message, history):
    prompt = "<s>"
    prompt += f"{system_prompt_text}\n\n"  # Add the system prompt

    if history:
        for user_prompt, bot_response in history:
            prompt += f"[INST] {user_prompt} [/INST] {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to process and display OCR text and image
def process_image_or_pdf(uploaded_file):
    extracted_text = ""
    
    if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
        # Process image
        image = Image.open(uploaded_file)
        img_array = np.array(image)  # Now numpy is imported, this will work

        # Perform OCR on the image
        result = reader.readtext(img_array)

        # Extract text from OCR result
        for detection in result:
            extracted_text += detection[1] + "\n"

        # Display extracted text and send to AI
        st.subheader("Extracted Text from Image: (please wait for AI to analyze it)")
        st.text(extracted_text)

        # Convert image to byte format for Streamlit to display
        try:
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            st.image(img_byte_arr, caption='Uploaded Image', use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying image: {e}")

    elif uploaded_file.type == "application/pdf":
        # Process PDF
        extracted_text = extract_text_from_pdf(uploaded_file)

        # Display extracted text and send to AI
        st.subheader("Extracted Text from PDF:")
        st.text(extracted_text)

    else:
        st.error("Invalid file format. Please upload an image or PDF.")

    # Get AI response based on the extracted text
    ai_response, history = get_medical_ai_response(extracted_text)
    
    # Display AI response with a black background and white text
    st.markdown(f"""
        <div style="background-color: black; padding: 20px; border-radius: 10px;">
            <p style="color: white; word-wrap: break-word; max-width: 700px; margin: 0; font-size: 18px;">{ai_response}</p>
        </div>
    """, unsafe_allow_html=True)

    return history, extracted_text

# Translate text to Hawaiian using googletrans
def translate_to_hawaiian(text):
    translator = Translator()
    
    # Detect the language of the input text
    detected_language = translator.detect(text).lang
    st.write(f"Detected Language: {detected_language}")
    
    # Translate the text from the detected language to Hawaiian
    translated = translator.translate(text, src=detected_language, dest='haw')
    
    return translated.text

# Main page with Streamlit interface
def contributors_page():
    st.balloons()
    st.write("""<h1 style="text-align: center; color:#FFF6F4;">Ask me about your Medical Documents</h1><hr>
                <p style="text-align:center;">Upload an image or PDF to extract text and analyze if it relates to medical documents.</p>
            """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an Image or PDF", type=['png', 'jpg', 'jpeg', 'pdf'])

    if uploaded_file is not None:
        # Process the uploaded file and get the AI response and history
        history, extracted_text = process_image_or_pdf(uploaded_file)

        # Ask for follow-up question from the user
        follow_up_question = st.text_input("Ask a follow-up question:")

        if follow_up_question:
            # Combine the follow-up question with the history to maintain context
            follow_up_prompt = f"Based on the information: {extracted_text}\nUser's follow-up question: {follow_up_question}"
            ai_response, history = get_medical_ai_response(follow_up_prompt, history)

            # Display AI response to the follow-up question
            st.markdown(f"""
                <div style="background-color: black; padding: 20px; border-radius: 10px;">
                    <p style="color: white; word-wrap: break-word; max-width: 700px; margin: 0; font-size: 18px;">{ai_response}</p>
                </div>
            """, unsafe_allow_html=True)

        # Translate to Hawaiian button
        if st.button("Translate to Hawaiian"):
            if extracted_text:
                hawaiian_translation = translate_to_hawaiian(extracted_text)
                st.write(f"Translation in Hawaiian: {hawaiian_translation}")
            else:
                st.write("No text to translate. Please upload a document or enter some text.")

# Run the Streamlit app
if __name__ == "__main__":
    contributors_page()
