import streamlit as st
from langchain.prompts import PromptTemplate
from ctransformers import AutoModelForCausalLM  # Adjusted import


# Function to load the model and get responses
def getLLamaresponse(input_text, no_words, blog_style):
    # Initialize the Llama model
    model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML", model_type="llama")

    # Create prompt with the specified style and word count
    prompt = f"Write a blog for {blog_style} with {no_words} words on the topic: {input_text}"

    # Generate response
    response = model(prompt)  # Adjusted response generation method
    return response


# Streamlit configuration and UI
st.set_page_config(
    page_title="Generate Blogs",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.header("Generate Blogs 🤖")
input_text = st.text_input("Enter the Blog Topic")

# Creating two columns for additional inputs
col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input("No of Words")
with col2:
    blog_style = st.selectbox("Writing the blog for",
                              ("Researchers", "Data Scientist", "Common People"), index=0)

# Button to submit and generate blog
submit = st.button("Generate")

# Final Response
if submit:
    if input_text and no_words and blog_style:
        # Call the function and display the response
        response = getLLamaresponse(input_text, no_words, blog_style)
        st.write(response)
    else:
        st.error("Please fill in all fields to generate the blog.")
