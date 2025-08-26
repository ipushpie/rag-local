import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

def load_pdf_text(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits the text into chunks of a specified size with optional overlap.
    
    Args:
        text (str): The text to be chunked.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of overlapping characters between chunks.
    
    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks



def embed_text(chunks):
    """
    Converts text chunks into embeddings using a pre-trained model.
    
    Args:
        chunks (list): A list of text chunks.
    
    Returns:
        list: A list of embeddings.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained model
    embeddings = model.encode(chunks)  # Generate embeddings for all chunks
    return embeddings




def store_embeddings(embeddings, index_file="embeddings.index"):
    """
    Stores embeddings in a FAISS index for efficient similarity search.
    
    Args:
        embeddings (list): A list of embeddings.
        index_file (str): The file path to save the FAISS index.
    """
    # Convert embeddings to a NumPy array
    embeddings = np.array(embeddings).astype('float32')
    
    # Create a FAISS index
    dimension = embeddings.shape[1]  # Embedding size
    index = faiss.IndexFlatL2(dimension)  # L2 distance metric
    index.add(embeddings)  # Add embeddings to the index
    
    # Save the index to a file
    faiss.write_index(index, index_file)
    print(f"Embeddings stored in {index_file}")

def load_embeddings(index_file="embeddings.index"):
    """
    Loads a FAISS index from a file.
    
    Args:
        index_file (str): The file path to the FAISS index.
    
    Returns:
        faiss.Index: The loaded FAISS index.
    """
    return faiss.read_index(index_file)

def query_embeddings(index, query, model, top_k=5):
    """
    Queries the FAISS index to find the most similar embeddings to the query.
    
    Args:
        index (faiss.Index): The FAISS index containing the embeddings.
        query (str): The input query string.
        model (SentenceTransformer): The embedding model to encode the query.
        top_k (int): The number of top results to retrieve.
    
    Returns:
        list: Indices of the most similar embeddings.
        list: Distances of the most similar embeddings.
    """
    # Convert the query into an embedding
    query_embedding = model.encode([query]).astype('float32')
    
    # Search the FAISS index
    distances, indices = index.search(query_embedding, top_k)
    return indices[0], distances[0]

def generate_response(chunks, query):
    """
    Generates a response using the retrieved chunks and the query.
    
    Args:
        chunks (list): The retrieved text chunks.
        query (str): The input query.
    
    Returns:
        str: The generated response.
    """
    # Combine the retrieved chunks into a single context
    context = " ".join(chunks)
    
    # Load a pre-trained language model pipeline
    generator = pipeline("text-generation", model="gpt2")
    
    # Generate a response using the query and context
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = generator(prompt, max_length=200, num_return_sequences=1)
    return response[0]["generated_text"]

# Example usage
if __name__ == "__main__":
    # Step 1: Extract text from the PDF
    extracted_text = load_pdf_text("ATLAN - MSA - v1.0 - signed_unlocked.pdf")
    
    # Step 2: Chunk the extracted text
    chunks = chunk_text(extracted_text)
    print(f"First 3 chunks: {chunks[:3]}")  # Print the first 3 chunks
    
    # Step 3: Generate embeddings for the chunks
    embeddings = embed_text(chunks)
    print(f"First 3 embeddings: {embeddings[:3]}")  # Print the first 3 embeddings
    
    # Step 4: Store the embeddings in a FAISS index
    store_embeddings(embeddings)
    
    # Step 5: Load the FAISS index
    index = load_embeddings()
    
    # Step 6: Define a query
    query = "What is the purpose of this document?"
    
    # Step 7: Query the index
    indices, distances = query_embeddings(index, query, SentenceTransformer('all-MiniLM-L6-v2'))
    print(f"Top indices: {indices}")
    print(f"Distances: {distances}")
    
    # Step 8: Generate a response
    top_chunks = [chunks[i] for i in indices]
    response = generate_response(top_chunks, query)
    print(f"Response: {response}")