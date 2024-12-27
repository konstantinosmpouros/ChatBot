import gradio as gr
from llama import Llama_3_8B
from mistral import Mistral_7B


models_available = {
    'Llama 3.1' : Llama_3_8B(),
    'Mistral 7B' : Mistral_7B()
}

def start():
    # Initialize a method for the chating with the llm
    def gradio_chat(user_input):
        if not llm.tokenizer or not llm.model:
            raise ValueError("Model not loaded. Please check the configuration.")

        # Add the user's prompt in the history and show it immediately
        llm.add_to_history(role='user', prompt=user_input)
        yield llm.history, ''

        # Iterate over the llm_response generator
        for text in llm.llm_response():
            llm.history[-1]['content'] += text  # Append the streamed token to the assistant's response
            yield llm.history, ''

    # Set the Gradio UI
    with gr.Blocks() as chat_interface:
        # Load the model and initialize the chat history
        llm = models_available['Llama 3.1']
        llm.load_model()
        
        # Title
        gr.Markdown("### Aegean AI Chat Assistant")

        # Add a dropdown for selecting models
        model_dropdown = gr.Dropdown(
            label="Select Model",
            choices=list(models_available.keys()),
            value=list(models_available.keys())[0],
            interactive=True
        )

        # Chatbox, Input, Send Button
        chatbot = gr.Chatbot(type='messages', autoscroll=True)
        user_input = gr.Textbox(label="Your Message", placeholder="Type your prompt here...", lines=1)
        send_button = gr.Button("Send")

        # Bind send button and text box to chat logic
        send_button.click(gradio_chat, 
                          inputs=[user_input], 
                          outputs=[chatbot, user_input])
        
        user_input.submit(gradio_chat, 
                          inputs=[user_input], 
                          outputs=[chatbot, user_input])

    # Launch the Gradio interface
    chat_interface.launch()
