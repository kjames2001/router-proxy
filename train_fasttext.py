#!/usr/bin/env python3
"""
Train a fastText supervised classifier for the Hermes Router Proxy.
Uses the same pretrain traces as the surrogate, converts to fastText format,
trains, quantizes, and saves the model.

Output: .router/fasttext/model.bin, model.ftz (quantized)
"""
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRACE_DIR = SCRIPT_DIR / "traces"
OUTPUT_DIR = SCRIPT_DIR / ".router" / "fasttext"

def _load_all_traces():
    """Load classify events from all trace files (pretrain + live + hermes-sessions)."""
    import glob
    all_samples = []
    for tf in sorted(glob.glob(str(TRACE_DIR / "router-trace-*.jsonl"))):
        for line in open(tf):
            try:
                d = json.loads(line)
                if d.get("event") == "classify" and d.get("user_message_preview"):
                    all_samples.append(d)
            except: pass
    return all_samples

def main():
    import fasttext

    # Load training data from all trace files
    samples = _load_all_traces()
    if not samples:
        print("ERROR: No traces found. Run pretrain_surrogate.py or label_hermes_sessions.py first.")
        sys.exit(1)
    print(f"Loaded {len(samples)} training samples")

    # Convert to fastText format: __label__category text
    train_file = OUTPUT_DIR / "train.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(train_file, "w", encoding="utf-8") as f:
        for s in samples:
            label = s["classifier_result"]
            text = s["user_message_preview"].replace("\n", " ").strip()
            f.write(f"__label__{label} {text}\n")

    print(f"Wrote {len(samples)} samples to {train_file}")

    # Train fastText supervised model
    print("\nTraining fastText model...")
    t0 = time.time()
    model = fasttext.train_supervised(
        input=str(train_file),
        epoch=50,
        lr=0.5,
        wordNgrams=2,
        dim=100,
        minCount=1,
        loss="softmax",
        verbose=2,
    )
    train_time = time.time() - t0
    print(f"Trained in {train_time:.2f}s")

    # Evaluate on training data (quick sanity check)
    n, p, r = model.test(str(train_file))
    print(f"Training accuracy: {p:.4f} ({n} samples)")

    # Save model
    model_bin = OUTPUT_DIR / "model.bin"
    model.save_model(str(model_bin))
    print(f"Saved model: {model_bin} ({model_bin.stat().st_size / 1024:.0f} KB)")

    # Quantize model for smaller size
    print("\nQuantizing model...")
    model.quantize(input=str(train_file), qnorm=True, retrain=True, cutoff=50000)
    model_ftz = OUTPUT_DIR / "model.ftz"
    model.save_model(str(model_ftz))
    print(f"Quantized model: {model_ftz} ({model_ftz.stat().st_size / 1024:.0f} KB)")

    # Test quantized model accuracy
    n, p, r = model.test(str(train_file))
    print(f"Quantized training accuracy: {p:.4f}")

    # Quick inference speed test
    test_texts = [
        "hello how are you",
        "write a python function to sort a list",
        "deploy to proxmox lxc container",
        "research paper on transformer architectures",
        "configure zigbee2mqtt home assistant",
    ]
    print("\nInference speed test (quantized model):")
    for text in test_texts:
        t0 = time.time()
        for _ in range(1000):
            label, prob = model.predict(text)
        elapsed = (time.time() - t0) / 1000 * 1000  # ms
        print(f"  {elapsed:.3f}ms — '{text[:40]}' → {label[0].replace('__label__', '')} ({prob[0]:.3f})")

    # Save label list
    labels_file = OUTPUT_DIR / "labels.json"
    labels = list(set(s["classifier_result"] for s in samples))
    labels_file.write_text(json.dumps({"labels": labels, "n_samples": len(samples)}, indent=2))
    print(f"\nLabels: {labels}")
    print(f"Done. Model saved at {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
