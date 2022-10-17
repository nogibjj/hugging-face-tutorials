from transformers import pipeline
import gradio as gr

#define the model
model = pipeline("text-generation")

#define the predict function
def predict(prompt):
    completion = model(prompt)[0]["generated_text"]
    return completion

#define the gradio interface
gr.Interface(fn=predict, inputs="text", outputs="text").launch()