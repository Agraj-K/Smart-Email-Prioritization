"""
End-to-end pipeline orchestrator for the Email Priority Classification System.

Subcommands:
  python pipeline.py preprocess   -- Download, parse, clean, save Enron data
  python pipeline.py train        -- Generate labels + fine-tune DistilBERT
  python pipeline.py index        -- Build FAISS index from historical emails
  python pipeline.py classify     -- Classify a new email (full 3-phase pipeline)
  python pipeline.py evaluate     -- Evaluate the fine-tuned model
  python pipeline.py demo         -- Interactive demo
"""

import argparse, sys, os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    PROCESSED_CSV, LABELED_CSV, MODEL_DIR, FAISS_INDEX_FILE,
    SAMPLE_SIZE, RAG_TOP_K, LABEL2ID,
)


def cmd_preprocess(args):
    from preprocessing.data_loader import DataLoader
    sample = args.sample if args.sample else SAMPLE_SIZE
    loader = DataLoader(sample_size=sample)
    df = loader.run()
    print(f"\n[OK] {len(df)} emails saved to {PROCESSED_CSV}")


def cmd_train(args):
    from fine_tuning.label_generator import LabelGenerator
    from fine_tuning.trainer import FineTuner

    if not os.path.exists(PROCESSED_CSV):
        print(f"[Error] Run `python pipeline.py preprocess` first."); sys.exit(1)

    df = pd.read_csv(PROCESSED_CSV)
    df = df.dropna(subset=["clean_body_classify"]).reset_index(drop=True)

    labeler = LabelGenerator()
    df = labeler.generate_and_save(df)

    trainer = FineTuner()
    history = trainer.train(df["clean_body_classify"].tolist(), df["priority_label"].tolist())
    print(f"\n[OK] Model saved to {MODEL_DIR}")


def cmd_index(args):
    from rag.embedder import Embedder
    from rag.vector_store import VectorStore

    if not os.path.exists(LABELED_CSV):
        print(f"[Error] Run `python pipeline.py train` first."); sys.exit(1)

    df = pd.read_csv(LABELED_CSV)
    df = df.dropna(subset=["clean_body_classify"]).reset_index(drop=True)

    embedder = Embedder()
    embeddings = embedder.embed_batch(df["clean_body_classify"].tolist())

    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            "subject": str(row.get("subject", "")),
            "body_snippet": str(row.get("clean_body_summary", ""))[:200],
            "priority_label": str(row.get("priority_label", "")),
            "sender": str(row.get("from", "")),
        })

    store = VectorStore()
    store.build_index(embeddings, metadata)
    store.save()
    print(f"\n[OK] FAISS index built with {len(df)} emails")


def cmd_classify(args):
    from inference.contextual_classifier import ContextualClassifier

    if not args.email:
        print("[Error] Provide --email \"...\""); sys.exit(1)

    classifier = ContextualClassifier()
    result = classifier.classify(args.email, top_k=RAG_TOP_K)

    print("\n" + "=" * 60)
    print(f"  Priority:    {result['priority']}")
    print(f"  Confidence:  {result['confidence']:.1%}")
    print(f"  Reasoning:   {result['reasoning']}")
    print(f"  Model Pred:  {result['model_prediction']} ({result['model_confidence']:.1%})")

    # Chain-of-Thought Steps
    cot_steps = result.get("cot_steps", {})
    if cot_steps:
        print("\n  Chain-of-Thought Reasoning:")
        step_names = {
            "urgency": "Step 1 - Urgency",
            "action": "Step 2 - Action Required",
            "sender": "Step 3 - Sender Context",
            "historical": "Step 4 - Historical Pattern",
            "model_agreement": "Step 5 - Model Agreement",
        }
        for key, name in step_names.items():
            if key in cot_steps:
                print(f"    {name}: {cot_steps[key]}")

    print("  RAG Context:")
    for i, ctx in enumerate(result["rag_context"], 1):
        print(f"    {i}. [{ctx['priority_label']}] {ctx['subject'][:50]} (sim={ctx['similarity_score']:.3f})")
    print("=" * 60)


def cmd_evaluate(args):
    from fine_tuning.trainer import FineTuner
    from fine_tuning.evaluator import Evaluator

    if not os.path.exists(LABELED_CSV):
        print("[Error] Run train first."); sys.exit(1)

    df = pd.read_csv(LABELED_CSV).dropna(subset=["clean_body_classify"]).reset_index(drop=True)
    trainer = FineTuner()
    trainer.load_model(MODEL_DIR)

    y_true, y_pred = [], []
    for _, row in df.iterrows():
        label, _, _ = trainer.predict(row["clean_body_classify"])
        y_pred.append(LABEL2ID[label])
        y_true.append(LABEL2ID[row["priority_label"]])

    Evaluator.print_report(y_true, y_pred)


def cmd_demo(args):
    from inference.contextual_classifier import ContextualClassifier
    classifier = ContextualClassifier()
    print("\nType an email and press Enter. Type 'quit' to exit.\n")
    while True:
        email = input("Email > ").strip()
        if email.lower() in ("quit", "exit", "q"):
            break
        if not email:
            continue
        r = classifier.classify(email)
        print(f"  Priority: {r['priority']} | Confidence: {r['confidence']:.1%} | {r['reasoning']}\n")


def main():
    parser = argparse.ArgumentParser(description="Email Priority Classification System")
    sub = parser.add_subparsers(dest="command")

    sp = sub.add_parser("preprocess")
    sp.add_argument("--sample", type=int, default=None)
    sp.set_defaults(func=cmd_preprocess)

    sub.add_parser("train").set_defaults(func=cmd_train)
    sub.add_parser("index").set_defaults(func=cmd_index)

    sp = sub.add_parser("classify")
    sp.add_argument("--email", type=str, required=True)
    sp.set_defaults(func=cmd_classify)

    sub.add_parser("evaluate").set_defaults(func=cmd_evaluate)
    sub.add_parser("demo").set_defaults(func=cmd_demo)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help(); sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
