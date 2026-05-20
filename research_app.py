import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

# Research papers
papers = {
    "Transformer Paper": """
    Transformers introduced self-attention mechanisms
    allowing parallel processing in NLP tasks.
    """,

    "BERT Paper": """
    BERT introduced bidirectional transformer training
    for contextual understanding.
    """,

    "GPT Paper": """
    GPT used autoregressive transformer decoding
    for text generation.
    """
}

# UI
st.title("Research Tool")

selected_paper = st.selectbox(
    "Select Research Paper",
    list(papers.keys())
)

style = st.selectbox(
    "Explanation Style",
    [
        "Beginner Friendly",
        "Technical",
        "Simple"
    ]
)

length = st.selectbox(
    "Explanation Length",
    [
        "Short",
        "Medium",
        "Detailed"
    ]
)

# Button
if st.button("Summarize"):

    # Loading message
    with st.spinner("Generating summary..."):

        try:

            # Local model
            llm = Ollama(model="phi3")

            # Prompt template
            template = """
            Summarize the following research paper.

            Paper:
            {paper}

            Explanation Style:
            {style}

            Explanation Length:
            {length}
            """

            prompt = PromptTemplate(
                input_variables=["paper", "style", "length"],
                template=template
            )

            final_prompt = prompt.format(
                paper=papers[selected_paper],
                style=style,
                length=length
            )

            # Generate response
            response = llm.invoke(final_prompt)

            # Output
            st.subheader("Summary")
            st.write(response)

        except Exception as e:
            st.error(f"Error: {e}")