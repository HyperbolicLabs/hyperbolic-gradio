import gradio as gr
import hyperbolic_gradio

gr.load(
    name='meta-llama/Meta-Llama-3-70B-Instruct',
    src=hyperbolic_gradio.registry,
).launch()