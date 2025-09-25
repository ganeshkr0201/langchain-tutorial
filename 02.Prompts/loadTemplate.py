from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


st.header("Research Assistent")

template = load_prompt('template.json')

title_input = st.selectbox("Select Research Paper Name", ["Attention is all you need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Begineer-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])


if st.button("Generate"):
    prompt = template.invoke({
        "title_input": title_input,
        "style_input": style_input,
        "length_input": length_input
    })
    result = model.invoke(prompt)

    st.write(result.content)