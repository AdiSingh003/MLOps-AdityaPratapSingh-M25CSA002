import argparse
import os

import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from data import download_dataset, split_data, GoodreadsDataset
from utils import (
        build_label_maps,
        encode_labels,
        decode_labels,
        compute_metrics,
        save_results,
        save_confusion_matrix,
        print_classification_report,
    )

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
OUTPUT_DIR = "./results"
CACHED_MODEL_DIR = "distilbert-reviews-genres"
CACHE_PATH = "genre_reviews_dict.pickle"
os.environ["WANDB_DISABLED"] = "true"


def main(hf_repo: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    genre_reviews_dict = download_dataset(CACHE_PATH)
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews_dict)
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")

    label2id, id2label = build_label_maps(train_labels)
    label_names = [id2label[i] for i in sorted(id2label)]
    train_labels_enc = encode_labels(train_labels, label2id)
    test_labels_enc = encode_labels(test_labels, label2id)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_enc = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = GoodreadsDataset(train_enc, train_labels_enc)
    test_dataset = GoodreadsDataset(test_enc, test_labels_enc)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=10,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model(CACHED_MODEL_DIR)
    tokenizer.save_pretrained(CACHED_MODEL_DIR)

    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    pred_output = trainer.predict(test_dataset)
    pred_ids    = pred_output.predictions.argmax(-1).flatten().tolist()
    pred_labels = decode_labels(pred_ids, id2label)

    print("\nClassification Report:")
    report = print_classification_report(test_labels, pred_labels,save_path="./results/local_classification_report.txt")
    eval_results["classification_report"] = report

    save_confusion_matrix(
        true_labels=test_labels,
        predicted_labels=pred_labels,
        label_names=label_names,
        path=f"results/local_eval_results_confusion_matrix.png",
        normalize=False,
    )
    save_confusion_matrix(
        true_labels=test_labels,
        predicted_labels=pred_labels,
        label_names=label_names,
        path=f"results/local_eval_results_confusion_matrix.png",
        normalize=True,
    )
    save_results(eval_results, "results/local_eval_results.json")

    if hf_repo:
        print(f"Pushing model to Hugging Face Hub: {hf_repo}")
        model.push_to_hub(hf_repo)
        tokenizer.push_to_hub(hf_repo)

    return trainer, tokenizer, id2label, test_dataset, test_texts, test_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", type=str, default=None,
                        help="HuggingFace repo id, e.g. username/goodreads-genre-classifier")
    args = parser.parse_args()
    main(hf_repo=args.hf_repo)
