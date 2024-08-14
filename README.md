# RAG Chat Application for Multiple PDFs

## Description

This RAG (Retrieval-Augmented Generation) Chat Application is a Flask-based web application that allows users to upload PDF documents and ask questions about their content. The application uses OpenAI's GPT-3.5 model and LangChain to provide accurate answers based on the uploaded documents.

## Features

- Upload and process multiple PDF documents
- Chat interface for asking questions about the uploaded documents
- Option to strictly answer from PDFs or use general language model knowledge
- Display of source references with context for each answer
- Continuous chat history
- OpenAI API key management through the UI
![image](https://github.com/user-attachments/assets/a015eb42-67d8-43cc-b29d-633fe47b1552)


## Prerequisites

- Python 3.7 or later
- pip (Python package installer)
- OpenAI API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/rag-chat-app.git
   cd rag-chat-app
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install flask werkzeug PyPDF2 langchain langchain-community langchain-openai openai tiktoken faiss-cpu
   ```

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000`

3. Enter your OpenAI API key in the provided field and click "Set API Key"

4. Upload PDF documents using the file input field

5. Once the files are processed, you can start asking questions in the input field at the bottom

6. Toggle the "Strictly answer from PDFs" checkbox to control whether the AI should only use information from the uploaded documents or if it can use its general knowledge

## Project Structure

- `app.py`: Main application file containing the Flask routes and RAG logic
- `templates/index.html`: HTML template for the web interface
- `uploads/`: Directory where uploaded PDF files are stored (created automatically)

## Important Notes

- This application is for demonstration purposes and may require additional security measures for production use
- Large PDF files may take some time to process
- Ensure your OpenAI API key has sufficient credits for continuous usage

## Contributing

Contributions to improve the RAG Chat Application are welcome. Please feel free to submit a Pull Request.


