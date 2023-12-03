import streamlit as st
from transformers import pipeline, AutoTokenizer

# Customizing the Streamlit page layout
def set_page_layout():
    st.markdown(
        """
         <style>
        body {
            background-color: #F0F0F0; /* Light gray */
            color: #333A56; /* Dark blue text */
        }
        .stTextInput, .stTextArea {
            background-color: #333A56; /* Dark blue */
            color: white; /* White text */
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 8px;
            box-shadow: none;
        }
        .stButton {
            color: #F0F0F0; /* Light gray text */
            background-color: #70A9A1; /* Light blue button */
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        }
        .stButton:hover {
            background-color: #50877D; /* Darker shade of blue on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to get the answer using the Transformer model
def run_question_answering(question, context):
    model_name = "deepset/roberta-base-squad2"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    result = nlp({'question': question, 'context': context})
    return result

# Main function to render the Streamlit app
def main():
    st.set_page_config(page_title="AnswerGenie")
    set_page_layout()

    st.title("🔮 AnswerGenie")
    st.subheader("Enter the context and ask a question to get answers!")

    context = st.text_area("Enter the context:")
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if question and context:
            st.markdown("---")
            st.write("🤔 Question:", question)
            st.write("💡 Context:", context)
            st.markdown("---")
            st.write("📜 **Answer:**")
            result = run_question_answering(question, context)
            st.success(result['answer'])
        else:
            st.warning("⚠️ Please provide both a question and context.")

if __name__ == "__main__":
    main()
