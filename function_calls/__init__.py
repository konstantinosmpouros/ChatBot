import datetime
from googlesearch import search

from pathlib import Path
import sys
import os

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

def current_datetime():
    """Get the current local time as a string."""
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_date

def google_search_top_5(query: str) -> list:
    """
    Perform a Google search and return the top 5 links.
    
    Args:
        query: The search query string.
    Returns:
        A list of the top 5 URLs from the Google search results.
    """
    try:
        # Perform a Google search and return the top 5 links
        results = list(search(query, num_results=5))
        return results
    except Exception as e:
        return [f"Error during search: {e}"]

def rag_retrieve(query: str):
    """
    Retrieves relevant document chunks from the vector database based on a query.

    Args:
        query: The search query to find relevant documents.

    Returns:
        str: A concatenated string of all retrieved document chunks, prefixed with
            "Retrieved chunks from the knowledge base: ".
    """
    # Create a retriever from the vector store
    retriever = chroma_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )
    # Retrieve the most similar chunks compared to the query
    retrieved_docs = retriever.get_relevant_documents(query)

    # Concatenate all chunks into a single string and return the results
    combined_chunks = "Retrieved chunks from the knowledge base, answer according to this in the user query: "
    combined_chunks += "\n\n".join(doc.page_content for doc in retrieved_docs)
    return combined_chunks