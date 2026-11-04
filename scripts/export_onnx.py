"""Export the fine-tuned MiniLM classifier to ONNX and verify parity."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 128


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/minilm-uspol")
    parser.add_argument("--out", default="model/onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()

    texts = [
        "Trump just announced new tariffs on Canada.",
        "My cat knocked over the trash can again lol.",
        "The Senate voted 51-49 to advance the budget bill.",
        "The UK parliament is debating its budget today.",
    ]
    enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt", return_token_type_ids=True)
    input_ids, attention_mask, token_type_ids = enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]

    with torch.no_grad():
        torch_logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits.numpy()

    onnx_path = out / "model.onnx"
    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "token_type_ids": token_type_ids.numpy(),
    })[0]
    diff = float(np.max(np.abs(torch_logits - onnx_logits)))
    print(json.dumps({"onnx": str(onnx_path), "max_abs_diff": diff, "shape": list(onnx_logits.shape)}))

    # int8 dynamic quantization for CPU-only deployments
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        int8_path = out / "model.int8.onnx"
        quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
        print(json.dumps({"int8": str(int8_path), "bytes": int8_path.stat().st_size}))
    except Exception as exc:  # quantization is optional
        print(f"int8 quantization skipped: {exc}")

    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        src = Path(args.model) / name
        if src.exists():
            shutil.copy2(src, out / name)
    (out / "export.json").write_text(json.dumps({
        "base": args.model,
        "max_len": MAX_LEN,
        "max_abs_diff": diff,
        "input_names": ["input_ids", "attention_mask", "token_type_ids"],
        "output_names": ["logits"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
