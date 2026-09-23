#!/usr/bin/env python3
"""
Train a SetFit classifier for the Hermes Router Proxy.
Uses the same pretrain traces as the surrogate, fine-tunes all-MiniLM-L6-v2
with a classification head on our labeled samples.

Output: .router/setfit/ (saved model directory)
"""
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRACE_DIR = SCRIPT_DIR / "traces"
OUTPUT_DIR = SCRIPT_DIR / ".router" / "setfit"

def _load_all_traces():
    """Load classifier-labeled traces from all trace files.

    Only traces labeled by an LLM teacher (not surrogate/fasttext/setfit
    predictions) are used as training data.  This matches the surrogate's
    extract_classifier_traces() logic.
    """
    import glob
    all_samples = []
    # Models whose predictions are ground-truth labels (LLM teachers)
    # Anything starting with surrogate/, fasttext/, setfit/ is a local
    # prediction and excluded.
    skip_prefixes = ("surrogate/", "fasttext/", "setfit/")
    for tf in sorted(glob.glob(str(TRACE_DIR / "router-trace-*.jsonl"))):
        for line in open(tf):
            try:
                d = json.loads(line)
                if d.get("event") != "classify":
                    continue
                model = d.get("model", "")
                if model.startswith(skip_prefixes):
                    continue
                if not d.get("user_message_preview") or not d.get("classifier_result"):
                    continue
                all_samples.append(d)
            except:
                pass
    return all_samples

def main():
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    # Load training data from all trace files
    samples = _load_all_traces()
    if not samples:
        print("ERROR: No traces found. Run pretrain_surrogate.py or label_hermes_sessions.py first.")
        sys.exit(1)
    print(f"Loaded {len(samples)} training samples")

    # Build dataset
    texts = [s["user_message_preview"] for s in samples]
    labels = [s["classifier_result"] for s in samples]

    # Create label-to-id mapping
    unique_labels = sorted(set(labels))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    label_ids = [label2id[l] for l in labels]

    print(f"Labels: {unique_labels}")
    dist = {}
    for l in labels:
        dist[l] = dist.get(l, 0) + 1
    print(f"Distribution: {dist}")

    # Create HuggingFace dataset
    ds = Dataset.from_dict({"text": texts, "label": label_ids})

    # Load SetFit model (same base as our zero-shot: all-MiniLM-L6-v2)
    print("\nLoading SetFit model (all-MiniLM-L6-v2)...")
    model = SetFitModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        labels=unique_labels,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_epochs=3,
        batch_size=8,
        num_iterations=1,
        body_learning_rate=2e-5,
        head_learning_rate=0.01,
        sampling_strategy="oversampling",
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
    )

    # Train
    print("\nTraining SetFit model...")
    t0 = time.time()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
    )
    trainer.train()
    train_time = time.time() - t0
    print(f"Trained in {train_time:.1f}s")

    # Evaluate on training data
    print("\nEvaluating on training data...")
    preds = model.predict(texts)
    correct = sum(1 for p, t in zip(preds, label_ids) if p == t)
    accuracy = correct / len(label_ids)
    print(f"Training accuracy: {accuracy:.4f} ({correct}/{len(label_ids)})")

    # Save model
    print(f"\nSaving model to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))

    # Save label mapping
    meta = {"labels": unique_labels, "n_samples": len(samples), "label2id": label2id, "id2label": id2label}
    (OUTPUT_DIR / "labels.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved label mapping: {meta['labels']}")

    # Quick inference speed test
    test_texts = [
        "hello how are you",
        "write a python function to sort a list",
        "deploy to proxmox lxc container",
        "research paper on transformer architectures",
        "configure zigbee2mqtt home assistant",
        "fix the nginx 502 bad gateway",
        "compare llama.cpp vs vllm for inference",
        "你好，今天天气怎么样",
    ]

    print("\nInference test:")
    for text in test_texts:
        t0 = time.time()
        pred = model([text])
        elapsed = (time.time() - t0) * 1000
        label = pred[0]
        # Get probability
        probs = model.predict_proba([text])[0]
        confidence = float(max(probs))
        print(f"  {elapsed:.1f}ms | {label:15s} ({confidence:.3f}) | {text}")

    print(f"\nDone. Model saved at {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
