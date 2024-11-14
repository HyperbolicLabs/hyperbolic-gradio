# `hyperbolic-gradio`

is a Python package that makes it very easy for developers to create machine learning apps that are powered by Hyperbolic AI's API.

# Installation

You can install `hyperbolic-gradio` directly using pip:

```bash
pip install hyperbolic-gradio
```

That's it! 

# Basic Usage

Just like if you were to use the `hyperbolic` API, you should first save your Hyperbolic API key to this environment variable:

```
export HYPERBOLIC_API_KEY=<your token>
```

Then in a Python file, write:

```python
import gradio as gr
import hyperbolic_gradio

gr.load(
    name='meta-llama/Meta-Llama-3-70B-Instruct',
    src=hyperbolic_gradio.registry,
).launch()
```

Run the Python file, and you should see a Gradio Interface connected to the model on Hyperbolic AI!

![ChatInterface](https://raw.githubusercontent.com/HyperbolicLabs/hyperbolic-gradio/master/chatinterface.png)

# Customization 

Once you can create a Gradio UI from an Hyperbolic API endpoint, you can customize it by setting your own input and output components, or any other arguments to `gr.Interface`. For example, the screenshot below was generated with:

```py
import gradio as gr
import hyperbolic_gradio

gr.load(
    name='meta-llama/Meta-Llama-3-70B-Instruct',
    src=hyperbolic_gradio.registry,
    title='Hyperbolic-Gradio Integration',
    description="Chat with Meta-Llama-3-70B-Instruct model.",
    examples=["Explain quantum gravity to a 5-year old.", "How many R are there in the word Strawberry?"]
).launch()
```
![ChatInterface with customizations](https://raw.githubusercontent.com/HyperbolicLabs/hyperbolic-gradio/master/hyperbolic-gradio.png)

# Composition

Or use your loaded Interface within larger Gradio Web UIs, e.g.

```python
import gradio as gr
import hyperbolic_gradio

with gr.Blocks() as demo:
    with gr.Tab("Meta-Llama-3-70B-Instruct"):
        gr.load('meta-llama/Meta-Llama-3-70B-Instruct', src=hyperbolic_gradio.registry)
    with gr.Tab("Llama-3.2-3B-Instruct"):
        gr.load('meta-llama/Llama-3.2-3B-Instruct', src=hyperbolic_gradio.registry)

demo.launch()
```

# Under the Hood

The `hyperbolic-gradio` Python library has two dependencies: `hyperbolic` and `gradio`. It defines a "registry" function `hyperbolic_gradio.registry`, which takes in a model name and returns a Gradio app.

# Supported Models in Hyperbolic AI

All chat API models supported by Hyperbolic AI are compatible with this integration. For a comprehensive list of available models and their specifications, please refer to the [Hyperbolic AI Models documentation](https://platform.hyperbolic.ai/docs/models).

-------

Note: if you are getting a 401 authentication error, then the Hyperbolic API Client is not able to get the API token from the environment variable. This happened to me as well, in which case save it in your Python session, like this:

```py
import os

os.environ["HYPERBOLIC_API_KEY"] = ...
```
