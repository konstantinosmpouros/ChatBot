from pathlib import Path
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import login

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(PACKAGE_ROOT))

import ui


# Load .env file and pass the keys to os as env variables
load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')

# Log in to hugging face
if token:
    login(token=token)
else:
    raise ValueError("HUGGINGFACE_TOKEN is not set. Please ensure the .env file exists and contains the token. Else please create an enviroment variable named 'HUGGINGFACE_TOKEN' and place your token inside.")


# Start the ChatBot UI
ui.start()