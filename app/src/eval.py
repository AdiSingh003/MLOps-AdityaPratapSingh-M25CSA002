import argparse
import json
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

os.environ["WANDB_DISABLED"] = "true"
MAX_LENGTH = 512
CACHE_PATH = "genre_reviews_dict.pickle"


def evaluate(model_path: str, results_filename: str = "eval_results.json") -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating model: {model_path}  |  device: {device}")

    genre_reviews_dict = download_dataset(CACHE_PATH)
    _, train_labels_raw, test_texts, test_labels = split_data(genre_reviews_dict)

    label2id, id2label = build_label_maps(train_labels_raw)
    label_names = [id2label[i] for i in sorted(id2label)]
    test_labels_enc = encode_labels(test_labels, label2id)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)

    test_enc = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_dataset = GoodreadsDataset(test_enc, test_labels_enc)

    eval_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=16,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    eval_results = trainer.evaluate()
    print("\nEvaluation results:")
    for k, v in eval_results.items():
        print(f"  {k}: {v}")

    pred_output = trainer.predict(test_dataset)
    pred_ids    = pred_output.predictions.argmax(-1).flatten().tolist()
    pred_labels = decode_labels(pred_ids, id2label)

    print("\nClassification Report:")
    report = print_classification_report(test_labels, pred_labels,save_path="./results/hub_classification_report.txt")
    eval_results["classification_report"] = report

    stem = os.path.splitext(results_filename)[0]
    save_confusion_matrix(
        true_labels=test_labels,
        predicted_labels=pred_labels,
        label_names=label_names,
        path=f"results/{stem}_confusion_matrix.png",
        normalize=False,
    )
    save_confusion_matrix(
        true_labels=test_labels,
        predicted_labels=pred_labels,
        label_names=label_names,
        path=f"results/{stem}_confusion_matrix_normalised.png",
        normalize=True,
    )

    save_results(eval_results, f"./results/{results_filename}")
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   type=str, default="./distilbert-reviews-genres",
                        help="Local path or HF Hub repo id")
    parser.add_argument("--results_file", type=str, default="eval_results.json")
    parser.add_argument("--compare",      action="store_true",
                        help="Compare local vs hub results after evaluation")
    args = parser.parse_args()

    evaluate(args.model_path, results_filename=args.results_file)