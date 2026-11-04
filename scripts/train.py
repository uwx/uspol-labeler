"""Fine-tune all-MiniLM-L6-v2 as a binary US-politics classifier."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

BASE = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 128


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class Posts(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.enc = tokenizer(
            [r["text"] for r in rows],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_token_type_ids=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
        self.weights = torch.tensor([r.get("confidence", 1.0) for r in rows], dtype=torch.float)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        item["weights"] = self.weights[i]
        return item


def best_threshold(labels, probs):
    best = (0.5, -1.0)
    for threshold in [i / 100 for i in range(5, 96, 5)]:
        f1 = metrics(labels, probs, threshold)["f1"]
        if f1 > best[1]:
            best = (threshold, f1)
    return best[0]


def metrics(labels, probs, threshold=0.5):
    pred = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for y, p in zip(labels, pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, pred) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(labels, pred) if y == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / len(labels) if labels else 0.0
    pos = [p for p, y in zip(probs, labels) if y == 1]
    neg = [p for p, y in zip(probs, labels) if y == 0]
    auc = None
    if pos and neg:
        wins = sum(1 for a in pos for b in neg if a > b) + 0.5 * sum(1 for a in pos for b in neg if a == b)
        auc = wins / (len(pos) * len(neg))
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc, "n": len(labels)}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    labels = []
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
        ).logits
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        labels.extend(batch["labels"].tolist())
    return labels, probs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.jsonl")
    parser.add_argument("--out", default="model/minilm-uspol")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2).to(device)

    rows = load_rows(args.data)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    gold_rows = [r for r in rows if r["split"] == "gold"]
    if args.max_train:
        train_rows = train_rows[: args.max_train]

    train_ds = Posts(train_rows, tokenizer)
    val_ds = Posts(val_rows, tokenizer)
    test_ds = Posts(test_rows, tokenizer)
    gold_ds = Posts(gold_rows, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)
    gold_loader = DataLoader(gold_ds, batch_size=128)

    positives = sum(r["label"] for r in train_rows)
    negatives = len(train_rows) - positives
    class_weights = torch.tensor([len(train_rows) / (2 * max(negatives, 1)), len(train_rows) / (2 * max(positives, 1))], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    print(json.dumps({"train": len(train_rows), "val": len(val_rows), "test": len(test_rows), "gold": len(gold_rows), "positives": positives}), flush=True)
    best_f1 = -1.0
    history = []
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        running = 0.0
        for step, batch in enumerate(train_loader):
            logits = model(
                input_ids=batch["input_ids"].to(device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                token_type_ids=batch["token_type_ids"].to(device, non_blocking=True),
            ).logits
            loss = loss_fn(logits, batch["labels"].to(device, non_blocking=True))
            loss = (loss * batch["weights"].to(device, non_blocking=True)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(train_loader)} loss {running / (step + 1):.4f}", flush=True)
        val_labels, val_probs = predict(model, val_loader, device)
        val_metrics = metrics(val_labels, val_probs)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": running / max(1, len(train_loader)),
            "val": val_metrics,
            "seconds": round(time.time() - start, 1),
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics), flush=True)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)

    model = AutoModelForSequenceClassification.from_pretrained(args.out).to(device)
    result = {"history": history}
    val_labels, val_probs = predict(model, val_loader, device)
    threshold = best_threshold(val_labels, val_probs)
    result["threshold"] = threshold
    for name, loader in (("val", val_loader), ("test", test_loader), ("gold", gold_loader)):
        labels, probs = predict(model, loader, device)
        result[name] = {
            "at_0.5": metrics(labels, probs, 0.5),
            f"at_{threshold:.2f}": metrics(labels, probs, threshold),
            "at_0.85": metrics(labels, probs, 0.85),
        }
        if name == "gold":
            rows_out = []
            for row, prob in zip(gold_rows, probs):
                rows_out.append({
                    "uri": row["uri"],
                    "text": row["text"],
                    "label": row["label"],
                    "prob": prob,
                    "pred": int(prob >= threshold),
                })
            with open("data/gold_predictions.jsonl", "w", encoding="utf-8") as f:
                for row in rows_out:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    Path(args.out, "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
