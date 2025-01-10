# ChatBot

## Overview

This project implements an interactive chat interface using **Gradio** to enable conversations with the **Llama 3.1 8B Instruct** model. The interface provides a user-friendly way to input messages and receive AI-generated responses in a chat-like environment.

## Features

- **Interactive Chat UI**: A Gradio-based chatbot interface for real-time conversations.

- **Llama 3.1 8B Instruct Model**: Utilizes the Llama 3.1 8B Instruct model for generating high-quality, context-aware responses.

- **Dynamic Response Handling**: Supports maintaining chat history to provide context-aware replies.

## Technologies Used

- **Gradio**: For creating the interactive user interface.
- **Transformers Library**: For loading and interacting with the Llama model.
- **PyTorch**: For GPU-accelerated model execution.

## How It Works

1. **Model Initialization**: The `load_model` function loads the Llama 3.1 8B Instruct model along with its tokenizer.

2. **Chat Logic**: User inputs are processed and sent to the model to generate a response, maintaining a history of the conversation.

3. **Gradio Interface**: The Gradio UI includes a text box for user input, a "Send" button to trigger responses, and a chatbot display for interaction.

## Usage

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have an enviroment variable named **HUGGINGFACE_TOKEN** with you token inside.

3. Run the Chatbot project from the parent directory:

   ```bash
   python3 ChatBot
   ```
