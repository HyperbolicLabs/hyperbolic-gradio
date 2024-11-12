import gradio as gr
import hyperbolic_gradio

gr.load(
    name='meta-llama/Meta-Llama-3-70B-Instruct',
    src=hyperbolic_gradio.registry,
    title='Hyperbolic-Gradio Integration',
    description="Chat with Meta-Llama-3-70B-Instruct model.",
    examples=["Explain quantum gravity to a 5-year old.", "How many R are there in the word Strawberry?"]
).launch()