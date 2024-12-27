import datetime
from googlesearch import search


def current_datetime():
    """Get the current local time as a string."""
    return str(datetime.datetime.now())

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
        return list(search(query, num_results=5))
    except Exception as e:
        return [f"Error during search: {e}"]