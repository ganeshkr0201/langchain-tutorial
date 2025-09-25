from langchain_core.prompts import PromptTemplate

template = PromptTemplate (
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

template.save('template.json')