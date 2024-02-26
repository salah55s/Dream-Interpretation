import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Set environment variables for API keys
os.environ["COHERE_API_KEY"] = "QGmj2joeTmbF9x6qj7j6xpjsWLvX2Q5OCdUW3qHD"
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyB_HEc-iwJd3X9qohdi8RtV5Y1_yQMpMbk"

# Load PDF document
def load_document(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

# Create FAISS index
def create_faiss_index(pages):
    faiss_index = FAISS.from_documents(pages, CohereEmbeddings())
    return faiss_index

# Streamlit app
def main():
    st.title("Dream Interpretation with Langchain")
    st.write("Enter your dream in Arabic and let Langchain interpret it!")

    # Get user input
    user_ask = st.text_input("ما هو حلمك؟")

    # Load PDF document
    pages = load_document("Noortryyy.pdf")

    # Create FAISS index
    faiss_index = create_faiss_index(pages)

    if st.button("Interpret Dream"):
        text_contents = []

        # Similarity search
        docs = faiss_index.similarity_search(user_ask, k=40)

        # Extract text content from each document
        for doc in docs:
            text_content = doc.page_content[:]
            text_contents.append(text_content)

        # Concatenate text contents
        concatenated_texts = [text_content + "رد بمساعده المعلومات المعطاه في تفسير الحلم ودائما انهى الرد بكلمة والله اعلم  واجعل الكلام شيق ومثير واحكى ككقصه ممتتاليه من الاحداث" for text_content in text_contents]

        # Initialize Google GenAI model
        model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

        # Invoke the model
        res = model.invoke([
            SystemMessage(content=concatenated_texts),
            HumanMessage(content=user_ask)
        ])

        # Display result
        st.write(eval(f'f"""{res}"""'))

if __name__ == "__main__":
    main()
