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