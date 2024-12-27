import gradio as gr
from llama import Llama_3_8B
from mistral import Mistral_7B


class ChatbotInterface:
    def __init__(self):
        self.models_available = {
            'Llama 3.1': Llama_3_8B(),
            'Mistral 7B': Mistral_7B()
        }
        self.llm = None
        self.pass_history = None
        
    def initialize_default_model(self):
        # Initialize with the first model in the list
        default_model = list(self.models_available.keys())[0]
        self.llm = self.models_available[default_model]
        self.llm.load_model()
        
    def gradio_chat(self, user_input):
        if not self.llm.tokenizer or not self.llm.model:
            raise ValueError("Model not loaded. Please check the configuration.")

        # Add the user's prompt in the history and show it immediately
        self.llm.add_to_history(role='user', prompt=user_input)
        yield self.llm.history, ''

        # Iterate over the llm_response generator
        for text in self.llm.llm_response():
            self.llm.history[-1]['content'] += text  # Append the streamed token to the assistant's response
            yield self.llm.history, ''

    def change_model(self, selected_model):
        if self.llm is not None:
            self.llm.unload_model()  # Unload the currently loaded model
        self.llm = self.models_available[selected_model]  # Update the llm instance
        self.llm.load_model()  # Load the selected model

    def create_interface(self):
        with gr.Blocks() as chat_interface:
            # Initialize the default model
            self.initialize_default_model()
            
            # Title
            gr.Markdown("### Aegean AI Chat Assistant")

            # Add a dropdown for selecting models
            model_dropdown = gr.Dropdown(
                label="Select Model",
                choices=list(self.models_available.keys()),
                value=list(self.models_available.keys())[0],
                interactive=True
            )

            # Chatbox, Input, Send Button
            chatbot = gr.Chatbot(type='messages', autoscroll=True)
            user_input = gr.Textbox(
                label="Your Message", 
                placeholder="Type your prompt here...", 
                lines=1
            )
            send_button = gr.Button("Send")

            # Bind dropdown to model update logic
            model_dropdown.change(
                self.change_model,
                inputs=[model_dropdown],
                outputs=[]
            )

            # Bind send button and text box to chat logic
            send_button.click(
                self.gradio_chat, 
                inputs=[user_input], 
                outputs=[chatbot, user_input]
            )
            
            user_input.submit(
                self.gradio_chat, 
                inputs=[user_input], 
                outputs=[chatbot, user_input]
            )

        return chat_interface
