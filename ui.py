from pathlib import Path
import os
import sys
import gradio as gr

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(PACKAGE_ROOT))

from chatbot import init_prompts, load_model, llm_response, add_to_history


models_available = {
    'Llama 3.1' : 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'Gemma 2' : 'google/gemma-2-9b-it'
}


def start():
    # Initialize a method for the chating with the llm
    def gradio_chat(history, user_input):
        if not tokenizer or not model:
            raise ValueError("Model not loaded. Please check the configuration.")

        # Add the user's prompt in the history and show it immediately
        history = add_to_history(history, role='user', prompt=user_input)
        chat_history = [
            (entry["content"], None) if entry["role"] == "user" else (None, entry["content"]) 
            for entry in history if entry["role"] != "system"
        ]

        yield chat_history, history, ''

        # Generate the response and stream it back to the chatbox
        history = add_to_history(history, role='assistant', prompt='')

        # Iterate over the llm_response generator
        for text in llm_response(tokenizer, model, history):
            history[-2]['content'] += text  # Append the streamed token to the assistant's response

            chat_history = [
                (entry["content"], None) if entry["role"] == "user" else (None, entry["content"]) 
                for entry in history if entry["role"] != "system"
            ]
            yield chat_history, history, ''

    # Load the model and initialize the chat history
    history = init_prompts()
    tokenizer, model = load_model(models_available["Llama 3.1"])

    # Set the Gradio UI
    with gr.Blocks() as chat_interface:
        # Title
        gr.Markdown("### Aegean AI Chat Assistant", elem_id="title")

        # Add a dropdown for selecting models
        model_dropdown = gr.Dropdown(
            label="Select Model",
            choices=list(models_available.keys()),
            value=list(models_available.keys())[0],
            interactive=True
        )

        # Chatbox, Input, Send Button
        chatbot = gr.Chatbot()
        user_input = gr.Textbox(label="Your Message", placeholder="Type your prompt here...", lines=1)
        send_button = gr.Button("Send")

        # Bind send button and text box to chat logic
        send_button.click(gradio_chat, 
                          inputs=[gr.State(history), user_input], 
                          outputs=[chatbot, gr.State(history), user_input])
        
        user_input.submit(gradio_chat, 
                          inputs=[gr.State(history), user_input], 
                          outputs=[chatbot, gr.State(history), user_input])

    # Launch the Gradio interface
    chat_interface.launch()
