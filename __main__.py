from pathlib import Path
import sys
import os

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from dotenv import load_dotenv
from huggingface_hub import login

from ui import ChatbotInterface


# Load .env file and pass the keys to os as env variables
load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')

# Log in to hugging face
if token:
    login(token=token)
else:
    raise ValueError("HUGGINGFACE_TOKEN is not set. Please ensure the .env file exists and contains the token. Else please create an enviroment variable named 'HUGGINGFACE_TOKEN' and place your token inside.")


# Start the ChatBot UI
chatbot = ChatbotInterface()
interface = chatbot.create_interface()
interface.launch()