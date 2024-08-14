import os
import logging
import PyPDF2
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain, LLMChain
from langchain.chains import ConversationalRetrievalChain, LLMChain


app = Flask(__name__)
app.secret_key = os.urandom(24).hex()  # Generate a random secret key

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
vector_store = None
chat_history = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = []
        for i, page in enumerate(reader.pages):
            content = page.extract_text()
            if content:
                text.append((content, i + 1))
    return text

def process_pdfs(pdf_paths):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    all_splits = []
    for pdf_path in pdf_paths:
        text_with_pages = extract_text_from_pdf(pdf_path)
        for text, page_num in text_with_pages:
            splits = text_splitter.split_text(text)
            for i, split in enumerate(splits):
                doc = Document(
                    page_content=split,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num,
                        "chunk": i,
                        "text": text  # Store full page text for context
                    }
                )
                all_splits.append(doc)
    return all_splits

def create_vector_store(splits):
    return FAISS.from_documents(splits, OpenAIEmbeddings())

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    api_key = request.json.get('api_key')
    if not api_key:
        return jsonify({"error": "No API key provided"}), 400
    
    session['openai_api_key'] = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    logging.info("API key set successfully")
    return jsonify({"message": "API key set successfully"}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    global vector_store
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({"error": "No file part"}), 400
    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        logging.error("No selected file")
        return jsonify({"error": "No selected file"}), 400
    
    pdf_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            pdf_paths.append(file_path)
    
    splits = process_pdfs(pdf_paths)
    vector_store = create_vector_store(splits)
    logging.info(f"Processed {len(pdf_paths)} PDFs successfully")
    return jsonify({"message": "Files uploaded and processed successfully"}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    global vector_store, chat_history
    if vector_store is None:
        logging.error("No documents uploaded yet")
        return jsonify({"error": "No documents uploaded yet"}), 400
    
    question = request.json.get('question')
    strict_pdf = request.json.get('strict_pdf', False)
    if not question:
        logging.error("No question provided")
        return jsonify({"error": "No question provided"}), 400
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    condense_question_prompt = PromptTemplate.from_template("""
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. If you don't know the answer, just say that you don't know. Don't try to make up an answer.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:""")
    
    qa_prompt = PromptTemplate.from_template("""
    You are an AI assistant for answering questions based on the given documents. You are given the following extracted parts of long documents and a question. Provide a conversational answer based on the context provided. If you don't know the answer, just say that you don't know. Don't try to make up an answer.

    {context}

    Question: {question}
    Answer:""")

    question_generator = LLMChain(llm=llm, prompt=condense_question_prompt)
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)

    qa_chain = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True
    )
    
    result = qa_chain({"question": question, "chat_history": chat_history})
    answer = result['answer']
    source_documents = result['source_documents']

    sources = []
    for doc in source_documents:
        source = doc.metadata['source']
        page = doc.metadata['page']
        chunk = doc.metadata['chunk']
        full_text = doc.metadata['text']
        context = get_context(full_text, doc.page_content)
        sources.append({
            "source": source,
            "page": page,
            "chunk": chunk,
            "context": context
        })

    chat_history.append((question, answer))
    logging.info(f"Question answered successfully: {question[:50]}...")
    return jsonify({"answer": answer, "sources": sources}), 200

# The get_context function remains the same
def get_context(full_text, chunk_text):
    # Find the chunk in the full text and return some context around it
    start_index = full_text.index(chunk_text)
    context_start = max(0, start_index - 100)
    context_end = min(len(full_text), start_index + len(chunk_text) + 100)
    return full_text[context_start:context_end]


@app.route('/clear_history', methods=['POST'])
def clear_history():
    global chat_history
    chat_history = []
    return jsonify({"message": "Chat history cleared successfully"}), 200

if __name__ == '__main__':
    logging.info("Starting the application. Please open http://127.0.0.1:5000 in your web browser.")
    app.run(debug=True)
