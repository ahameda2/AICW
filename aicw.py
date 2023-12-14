import streamlit as st
from huggingface_hub import notebook_login
from PyPDF2 import PdfReader
import io
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings, SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader
import tempfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# Setting up the Streamlit page
st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")


# Define a class to hold the text and metadata with the expected attributes
class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}


# Define the function to read and extract text from a PDF byte stream
def read_pdf(file_stream):
    reader = PdfReader(file_stream)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ""  # Adding a fallback for pages with no text
    return text


# Initialize session state variables
#if 'chain' not in st.session_state:
    #st.session_state.chain = None
#if 'documents_processed' not in st.session_state:
    #st.session_state.documents_processed = False

# Sidebar for Hugging Face Login and PDF Upload
with st.sidebar:
    st.subheader("Hugging Face Login")
    hf_token = st.text_input("Enter your Hugging Face token", type="password")
    submit_button = st.button("Login")

    if submit_button:
        try:
            notebook_login(hf_token)
            st.success("Connected successfully to Hugging Face Hub.")
        except Exception as e:
            st.error(f"Failed to connect: {e}")

    st.subheader("Your Documents")
    uploaded_files = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type='pdf')
    if uploaded_files:
        documents = []
        for file in uploaded_files:
            file_extensions = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extensions == '.pdf':
                loader = PyPDFLoader(temp_file_path)
            else:
                st.error(f"Unsupported file type: {file_extensions}")
                st.stop()

            if loader:
                documents.extend(loader.load())
                os.remove(temp_file_path)

    #process_button = st.button("Process PDFs")

# Main Page Interface
st.header("Chat with Multiple PDFs 📚")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # For Apple Silicon GPUs
    else:
        return "cpu"


DEVICE = get_device()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Processing PDFs
if uploaded_files:

    st.session_state.documents_processed = True
    st.success("PDFs processed successfully!")

    # Combine all texts and split into chunks
    combined_text = " ".join([doc.page_content for doc in documents])
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    split_text = text_splitter.split_documents(documents)
    st.write(f"Number of text chunks: {len(split_text)}")

    # Creating database of embeddings
    db = Chroma.from_documents(split_text, embeddings, persist_directory="db")
    st.success("Embeddings processed and database created.")

    # Initialize the model and tokenizer for conversational AI
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, torch_dtype='auto')
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype='auto').to(DEVICE).half()
    max_seq_length = 128

    # Set up the text generation model
    text_generator = pipeline('text-generation', model=model_name, device=DEVICE, tokenizer=tokenizer,
                              max_new_tokens=20, max_length=128)

    # Set up the conversational retrieval chain
    retriever = db.as_retriever()
    chain = ConversationalRetrievalChain(llm=model, retriever=retriever, return_source_documents=True)

    def ans_gen(instruction):
        response = ''
        instruction = instruction
        qa = chain
        generated_text = qa(instruction)
        answer = generated_text['result']
        return answer

    user_input = st.text_input("Ask a question about your documents:", key="input")

    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hello: I am a chatbot. Ask me a question."]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hello: I am a chatbot. Ask me a question."]

    if user_input:
        answer = ans_gen({"query": user_input})
        st.session_state["past"].append(user_input)
        response = answer
        st.session_state["generated"].append(response)

    if st.session_state["generated"]:
        st.write("Chatbot:")
        st.write(st.session_state["generated"][-1])

    # Chat interface
    #if st.session_state.documents_processed:
        #st.subheader("Chat with your PDFs")
        #user_query = st.text_input("Ask a question about your documents:", key="user_query")
        #if st.button("Submit"):
            #if chain and user_query:
                #result = chain({'question': user_query, 'chat_history': []})
                #st.write('Answer:', result['answer'])
            #else:
                #st.warning("Please process PDFs before asking questions.")
    #else:
        #st.write("Please upload and process PDFs to enable the chat feature.")
