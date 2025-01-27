import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(PACKAGE_ROOT))

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class RAGPipeline():

    def __init__(self):
        # Initialize the paths to the knowledge base and the vectore store
        self.knowledge_base_dir = os.path.join(PACKAGE_ROOT, "knowledge_base")
        self.vector_store_dir   = os.path.join(PACKAGE_ROOT, "vector_store")
        
        # Connect to OpenAI in order to use the embeddings model
        openai_api_key = os.getenv('OPENAI_API_KEY')
        self.embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Load the chroma db that holds the embeddings and the chunks
        self.chroma_db = self.load_db()

    def load_pdfs(self):
        """
            Load and combine text from all PDF files in the specified knowledge base directory.

            This method scans the knowledge base directory for PDF files, loads the content of each PDF,
            combines the text from all pages of a PDF into a single string, and creates a Document object
            with the combined content. Metadata about the source file is also added to each Document.

            Returns:
                list[Document]: A list of Document objects, where each Document represents the combined
                text of all pages from a single PDF file.
        """
        # Grab all PDF files in the directory
        pdf_files = [f for f in os.listdir(self.knowledge_base_dir) if f.lower().endswith(".pdf")]

        # This list will hold exactly 1 Document per PDF
        all_docs = []

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.knowledge_base_dir, pdf_file)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()  # This typically returns multiple Document objects (one per page)
            
            # Combine text from all pages into one big string
            combined_text = "\n".join(page.page_content for page in pages)
            
            # Create a single Document with all pages combined
            single_doc = Document(
                page_content=combined_text,
                metadata={"source": pdf_file}  # or any other metadata
            )
            all_docs.append(single_doc)
        return all_docs

    def load_db(self):
        """
            Load an existing Chroma vector store or create a new one using PDF documents and an embedding model.

            If a vector store already exists in the specified directory, it is loaded and returned.
            Otherwise, this method:
                1. Loads PDF files from the knowledge base directory.
                2. Creates a Chroma vector store by embedding the loaded documents using the specified
                embedding model.
                3. Persists the newly created vector store to disk.

            Returns:
                Chroma: A Chroma vector store object representing the embedded documents.
        """
        if os.path.exists(self.vector_store_dir) and os.listdir(self.vector_store_dir):
            # Load a Chroma vector store
            print(f'[RAGPipeline] Loading existing index from {self.vector_store_dir}')
            return Chroma(embedding_function=self.embeddings_model, persist_directory=self.vector_store_dir)

        else:
            print("[RAGPipeline] No existing index found. Creating a new one...")
            
            # Load the pdfs
            pdfs = self.load_pdfs()
            
            # Create a Chroma vector store using the documents and embeddings
            db = Chroma.from_documents(documents=pdfs,
                                       embedding=self.embeddings_model,
                                       persist_directory=self.vector_store_dir)
            
            # Persist the database to disk (so you don’t have to re-embed every time)
            db.persist()
            return db

    def retrieve(self, query, k=2):
        """
            Retrieves relevant document chunks from the vector database based on a query.

            Args:
                query (str): The search query to find relevant documents.
                k (int, optional): Number of document chunks to retrieve. Defaults to 2.

            Returns:
                str: A concatenated string of all retrieved document chunks, prefixed with
                    "Retrieved chunks from the knowledge base: ".
        """
        # Create a retriever from the vector store
        retriever = self.chroma_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        # Retrieve the most similar chunks compared to the query
        retrieved_docs = retriever.get_relevant_documents(query)

        # Concatenate all chunks into a single string and return the results
        combined_chunks = "\n".join((f'\nDocument {i}\n' + doc.page_content) for i, doc in enumerate(retrieved_docs))
        combined_chunks += "Above are the retrieved documents from the knowledge base"
        return combined_chunks
