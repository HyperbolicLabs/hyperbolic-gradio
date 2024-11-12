import gradio as gr
import hyperbolic_gradio

with gr.Blocks() as demo:
    with gr.Tab("Meta-Llama-3-70B-Instruct"):
        gr.load('meta-llama/Meta-Llama-3-70B-Instruct', src=hyperbolic_gradio.registry)
    with gr.Tab("Llama-3.2-3B-Instruct"):
        gr.load('meta-llama/Llama-3.2-3B-Instruct', src=hyperbolic_gradio.registry)

demo.launch()