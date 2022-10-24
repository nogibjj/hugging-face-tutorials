#!/usr/bin/env python

"""
Fine Tuning Example with HuggingFace

Based on official tutorial
"""

from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import click


def tokenize_function(examples):
    """Tokenize Function"""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def setup():
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )
    return tokenizer, tokenized_datasets, model


def compute_metrics(eval_pred):
    """Evaluation Metric"""

    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train_model(model, tokenized_datasets):

    training_args = TrainingArguments(
        output_dir="hf_fine_tune_hello_world",
        evaluation_strategy="epoch",
        push_to_hub=True,
        push_to_hub_model_id="hf_fine_tune_hello_world",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    return trainer


@click.command()
def main():
    """Main Function

    Example:
    python fineTuneCli.py
    """
    tokenizer, tokenized_datasets, model = setup()
    trainer = train_model(model, tokenized_datasets)
    trainer.train()
    trainer.push_to_hub()
    tokenizer.push_to_hub(repo_id="hf_fine_tune_hello_world")


if __name__ == "__main__":
    main()
