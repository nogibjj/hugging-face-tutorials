import gradio as gr
import pandas as pd

# title and description are optional
title = "NBA Dataset"
description = "display the nba dataset"

df = df["train"].to_pandas()
df.dropna(axis=0, inplace=True)


#gr.Interface.load("huggingface/scikit-learn/tabular-playground", title=title, description=description).launch()