from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(PACKAGE_ROOT))

from chatbot import hf_login
import ui


# Log in to the hugging face
hf_login()

# Start the ChatBot UI
ui.start()