from pathlib import Path
import sys
import os

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import gradio as gr
from LLMs import Llama_3_8B

class ChatbotInterface:
    def __init__(self):
        self.llm = Llama_3_8B()
        self.llm.load_model()
        self.search_in_rag = False

    def toggle_rag(self):
        # Toggle the boolean value
        self.search_in_rag = not self.search_in_rag
        
        # Return an update with changed color based on RAG search state
        return gr.update(
            value="RAG Search", 
            variant="secondary" if not self.search_in_rag else "primary"
        )

    def gradio_chat(self, user_input):
        if not self.llm.tokenizer or not self.llm.model:
            raise ValueError("Model not loaded. Please check the configuration.")

        # Modify user_input if the toggle is active
        if self.search_in_rag:
            user_input = 'Search in the RAG for the following: ' + user_input

        # Add the user's prompt in the history and show it immediately
        self.llm.add_to_history(role='user', prompt=user_input)
        
        # Create a display history that removes only the RAG search prefix
        display_history = [
            {**msg, 'content': msg['content'].replace('Search in the RAG for the following: ', '')} 
            for msg in self.llm.history
        ]
        
        yield display_history, gr.update(value='')

        # Iterate over the llm_response generator
        for text in self.llm.llm_response():
            self.llm.history[-1]['content'] += text  # Append the streamed token to the assistant's response
            
            # Create a display history that removes only the RAG search prefix
            display_history = [
                {**msg, 'content': msg['content'].replace('Search in the RAG for the following: ', '')} 
                for msg in self.llm.history
            ]
            yield display_history, gr.update()

    def create_interface(self):
        with gr.Blocks() as chat_interface:
            # Title
            gr.Markdown("### AI Agent")

            # Chatbox, Input, Send Button, and Toggle Button
            chatbot = gr.Chatbot(type='messages', autoscroll=True, height=600)
            user_input = gr.Textbox(
                label="Your Message", 
                placeholder="Type your prompt here...", 
                lines=1
            )
            send_button = gr.Button("Send")
            toggle_button = gr.Button("RAG Search")

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

            # Bind toggle button to toggle logic
            toggle_button.click(
                self.toggle_rag, 
                inputs=[], 
                outputs=[toggle_button]
            )

        return chat_interface