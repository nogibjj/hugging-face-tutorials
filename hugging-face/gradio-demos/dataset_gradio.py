import gradio as gr

# title and description are optional
title = "Supersoaker Defective Product Prediction"
description = "This model predicts Supersoaker production line failures. Drag and drop any slice from dataset or edit values as you wish in below dataframe component."

gr.Interface.load(
    "huggingface/scikit-learn/tabular-playground", title=title, description=description
).launch()
