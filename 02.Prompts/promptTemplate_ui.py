from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template = PromptTemplate(
template="""
please summarize the reasearch paper titled "{title_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details:
    - Include relevent mathemetical equations if present in the paper.
    - Explain the mathemetical concepts using simple, intutive code snippets where applicable.
2. Analogies:
    - Use relatable analogies to simplify complex ideas.
    If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
    insure the summery is clear, accurate, and aligned with the provided style and length.
""",
input_variables=["title_input", "style_input", "length_input"],
validate_template=True
)
 
st.header("Research Assistent")

title_input = st.selectbox("Select Research Paper Name", ["Attention is all you need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Begineer-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])


if st.button("Generate"):
    prompt = template.invoke({
        'title_input': title_input,
        'style_input': style_input,
        'length_input': length_input
    })
    result = model.invoke(prompt)

    st.write(result.content)