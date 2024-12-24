from pathlib import Path
import os
import sys
import gradio as gr

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(PACKAGE_ROOT))

from chatbot import init_prompts, load_model, llm_response


models_available = {
    'Llama 3.1' : 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'Gemma 2' : 'google/gemma-2-9b-it'
}


def start():
    # Load the model and initialize the chat history
    history = init_prompts()
    tokenizer, model = load_model(models_available["Llama 3.1"])

    # Initialize a method for the chating with the llm
    def gradio_chat(user_input, history):
        if not tokenizer or not model:
            return "Model not loaded. Please check the configuration.", history

        history = llm_response(tokenizer, model, history, user_input)
        chat_history = [
            (entry["content"], None) if entry["role"] == "user" else (None, entry["content"]) 
            for entry in history if entry["role"] != "system"
        ]
        return chat_history, history

    # Set the Gradio UI
    with gr.Blocks() as chat_interface:
        gr.Markdown("### Aegean AI Chat Assistant", elem_id="title")

        chatbot = gr.Chatbot()
        user_input = gr.Textbox(label="Your Message", placeholder="Type your message here...", lines=1)
        send_button = gr.Button("Send")

        # Logic for handling user input
        def handle_chat(input_text, history):
            chat_output, updated_history = gradio_chat(input_text, history)
            return chat_output, updated_history, ""

        # Bind send button and text box to chat logic
        send_button.click(handle_chat, inputs=[user_input, gr.State(history)], outputs=[chatbot, gr.State(history), user_input])
        user_input.submit(handle_chat, inputs=[user_input, gr.State(history)], outputs=[chatbot, gr.State(history), user_input])

    # Launch the Gradio interface
    chat_interface.launch()
