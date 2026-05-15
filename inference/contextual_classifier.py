"""
Contextual Classifier — Phase 3 of the architecture.

Orchestrates the full inference pipeline:
  1. New Email Text arrives
  2. Fine-tuned DistilBERT produces prediction + confidence
  3. RAG retriever finds similar historical emails
  4. Generative LLM (flan-t5-base) combines everything into a
     final classification with confidence score and reasoning

Architecture flow:
  New Email → Retrieved Historical Context from RAG
            → Contextual Input
            → Generative LLM (Fine-Tuned Model / Prompt)
            → Final Classification: HIGH PRIORITY with Confidence Score
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    LLM_MODEL_NAME, LLM_MAX_INPUT_LENGTH, LLM_MAX_NEW_TOKENS,
    MODEL_DIR, PRIORITY_LABELS,
)
from fine_tuning.trainer import FineTuner
from rag.retriever import Retriever
from inference.llm_prompt import PromptBuilder


class ContextualClassifier:
    """
    End-to-end contextual email priority classifier.

    Combines:
      - Fine-tuned DistilBERT (Phase 1 output)
      - FAISS-based RAG retrieval (Phase 2 output)
      - Generative LLM reasoning (Phase 3)
    """

    def __init__(self, load_all: bool = True):
        """
        Initialize all three components.

        Args:
            load_all: If True, load the fine-tuned model and FAISS index
                      from disk immediately. Set False for lazy loading.
        """
        # Phase 1: Fine-tuned classifier
        self.fine_tuner = FineTuner()
        if load_all:
            self.fine_tuner.load_model(MODEL_DIR)

        # Phase 2: RAG retriever
        self.retriever = Retriever()
        if load_all:
            self.retriever.load_index()

        # Phase 3: Generative LLM
        self.device = self.fine_tuner.device
        print(f"[ContextualClassifier] Loading LLM: {LLM_MODEL_NAME}...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(
            LLM_MODEL_NAME
        ).to(self.device)
        self.llm_model.eval()
        print("[ContextualClassifier] All components loaded.")

    # ── LLM Generation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _generate_llm_response(self, prompt: str) -> str:
        """Send a prompt to flan-t5-base and return the generated text."""
        inputs = self.llm_tokenizer(
            prompt,
            truncation=True,
            max_length=LLM_MAX_INPUT_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=2,
        )

        return self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ── Parse LLM Output ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_llm_output(
        llm_text: str, fallback_label: str, fallback_confidence: float
    ) -> dict:
        """
        Parse the LLM's structured output.
        Falls back to the fine-tuned model's prediction if parsing fails.
        """
        result = {
            "priority": fallback_label,
            "confidence": fallback_confidence,
            "reasoning": "Based on fine-tuned model prediction.",
            "source": "model_fallback",
        }

        # Try to extract Priority
        priority_match = re.search(
            r"Priority:\s*(High|Medium|Low)", llm_text, re.IGNORECASE
        )
        if priority_match:
            label = priority_match.group(1).capitalize()
            if label in PRIORITY_LABELS:
                result["priority"] = label
                result["source"] = "llm"

        # Try to extract Confidence
        conf_match = re.search(
            r"Confidence:\s*([\d.]+)", llm_text
        )
        if conf_match:
            try:
                conf = float(conf_match.group(1))
                if 0.0 <= conf <= 1.0:
                    result["confidence"] = conf
            except ValueError:
                pass

        # Try to extract Reasoning
        reason_match = re.search(
            r"Reasoning:\s*(.+)", llm_text, re.IGNORECASE
        )
        if reason_match:
            result["reasoning"] = reason_match.group(1).strip()

        return result

    # ── Main Classification API ───────────────────────────────────────────────

    def classify(self, email_text: str, top_k: int = 3) -> dict:
        """
        Classify a single email using the full 3-phase pipeline.

        Args:
            email_text: The email body text to classify.
            top_k:      Number of similar emails to retrieve from RAG.

        Returns:
            Dictionary with:
              - priority:       Final label (High/Medium/Low)
              - confidence:     Confidence score (0-1)
              - reasoning:      LLM's explanation
              - source:         "llm" or "model_fallback"
              - model_prediction: Raw fine-tuned model output
              - model_confidence: Raw model confidence
              - model_probabilities: Full probability distribution
              - rag_context:    Retrieved similar emails
        """
        # Step 1: Fine-tuned model prediction
        model_label, model_conf, model_probs = self.fine_tuner.predict(
            email_text
        )

        # Step 2: RAG retrieval
        rag_results = self.retriever.query(email_text, top_k=top_k)

        # Step 3: Build prompt and get LLM response
        prompt = PromptBuilder.build_prompt(
            email_text=email_text,
            retrieved_emails=rag_results,
            model_prediction=model_label,
            model_confidence=model_conf,
            model_probabilities=model_probs,
        )

        llm_response = self._generate_llm_response(prompt)

        # Step 4: Parse LLM output (with model fallback)
        parsed = self._parse_llm_output(llm_response, model_label, model_conf)

        return {
            **parsed,
            "model_prediction": model_label,
            "model_confidence": model_conf,
            "model_probabilities": model_probs,
            "rag_context": rag_results,
            "llm_raw_response": llm_response,
        }

    # ── Batch Classification ──────────────────────────────────────────────────

    def classify_batch(
        self, emails: list[str], top_k: int = 3
    ) -> list[dict]:
        """Classify a batch of emails."""
        results = []
        for i, email in enumerate(emails):
            print(f"[ContextualClassifier] Classifying {i+1}/{len(emails)}...")
            results.append(self.classify(email, top_k=top_k))
        return results
