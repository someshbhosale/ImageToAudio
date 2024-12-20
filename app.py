# Import necessary libraries and modules
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from gtts import gTTS
from langchain import PromptTemplate, LLMChain, HuggingFaceHub
import requests
import os
import streamlit as st


# Load environment variables from a .env file
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define the Streamlit application main function
def main():
    # Configure Streamlit app settings
    st.set_page_config(page_title="Image to Audio Story", page_icon="📖", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f5f5f5;
        }
        .main-header {
            text-align: center;
            padding: 20px;
            background-color: #4b7bec;
            color: white;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .uploaded-img {
            display: flex;
            justify-content: center;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            color: #999;
        }
        </style>
    """, unsafe_allow_html=True)

    # Add a custom header
    st.markdown("<div class='main-header'><h1>📖 Turn Images into Audio Stories</h1><p>Transform your imagination by converting images into creative short stories and audio</p></div>", unsafe_allow_html=True)

    # Layout columns
    col1, col2 = st.columns([1, 2])

    # Allow user to upload an image file
    with col1:
        st.subheader("Upload Your Image")
        uploaded_file = st.file_uploader("Choose an image file:", type=["jpg", "png", "jpeg"])

    # Check if an image has been uploaded
    if uploaded_file is not None:
        # Read and save the uploaded image file
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)

        # Display the uploaded image in the second column
        with col2:
            st.image(uploaded_file, caption='Uploaded Image', use_container_width =True)

        # Step 1: Convert the image to text using an image-to-text pipeline
        with st.spinner("Analyzing image and extracting text..."):
            scenario = img2text(uploaded_file.name)
        st.success("Text extracted successfully!")

        # Step 2: Generate a short story based on the extracted text
        with st.spinner("Generating a creative story..."):
            story = generate_story(scenario)
        st.success("Story created successfully!")

        # Step 3: Convert the generated story text to audio
        with st.spinner("Converting story to audio..."):
            audio_file = text2speech(story)
        st.success("Audio generated successfully!")

        # Display the extracted scenario and generated story in expandable sections
        st.markdown("### Results")
        with st.expander("📜 Extracted Scenario"):
            st.write(scenario)
        with st.expander("📖 Generated Story"):
            st.write(story)

        # Play the generated audio
        st.markdown("### 🎧 Listen to the Story")
        st.audio(audio_file)


# Step 1: Convert an image to text using an image-to-text pipeline
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]['generated_text']
    print(text)
    return text

# Step 2: Generate a short story based on a scenario using a language model
def generate_story(scenario):
    # Define the template with clear instructions
    template = """
    you are a story teller;
    you can generate a short story based on a simple narrative, the story should be no more than 100 words;

    CONTEXT: {scenario}
    STORY: 
    """
    # Load the HuggingFace language model
    repo_id = "tiiuae/falcon-7b-instruct"
    llm = HuggingFaceHub(
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        repo_id=repo_id,
        verbose=False,
        model_kwargs={"temperature": 0.1, "max_new_tokens": 1500},
    )
    # Use the template to create a prompt
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(llm=llm, prompt=prompt)

    # Generate the story using the provided scenario
    full_output = story_llm.predict(scenario=scenario)
    print("Full Output from LLM:\n", full_output)

    # Extract only the story part from the output
    story = full_output.split("STORY:")[1].strip()
    print("Extracted Story:\n", story)
    return story


# Step 3: Convert text to speech using an external API
def text2speech(message):
    tts = gTTS(text=message, lang='en')

    # Save the audio file
    audio_file = "audio.mp3"
    tts.save(audio_file)

    return audio_file


# Execute the main function when the script is run directly
if __name__ == '__main__':
    main()
