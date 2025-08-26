# Retrieval-Augmented Generation (RAG) System

This project implements a local Retrieval-Augmented Generation (RAG) system. The system extracts text from a PDF, chunks the text, generates embeddings, stores them in a FAISS index, and retrieves relevant chunks to generate responses using a language model.

## Features
- Extract text from PDF files using PyMuPDF.
- Chunk the text into manageable pieces.
- Generate embeddings for the chunks using Sentence Transformers.
- Store embeddings in a FAISS index for efficient similarity search.
- Query the FAISS index to retrieve relevant chunks.
- Generate responses using a pre-trained language model (e.g., GPT-2).

## Requirements
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage
1. Place your PDF file in the project directory.
2. Update the `main.py` script with the name of your PDF file.
3. Run the script:
   ```bash
   python3 main.py
   ```
4. Enter a query to retrieve relevant information and generate a response.

## Project Structure
- `main.py`: Main script containing the RAG pipeline.
- `requirements.txt`: List of dependencies.
- `embeddings.index`: FAISS index file (generated after running the script).

## Example Query
- **Input**: "What is the purpose of this document?"
- **Output**: A generated response based on the retrieved chunks.

## Future Improvements
- Add support for multiple file formats (e.g., Word, plain text).
- Implement a web-based interface using Flask or FastAPI.
- Optimize performance for large datasets.

## License
This project is open source and available under the MIT License.
