"""
Prompt templates for the generative LLM (Phase 3).

Builds a structured prompt that includes:
  1. The new email text
  2. Retrieved historical context from RAG (similar past emails + their priorities)
  3. The fine-tuned classifier's prediction + confidence
  4. Instructions for the LLM to produce a final classification
"""


class PromptBuilder:
    """Build prompts for contextual email priority classification."""

    CLASSIFICATION_TEMPLATE = """You are an expert email priority classifier. Your task is to determine the priority level of a new email using context from similar historical emails and a machine learning model's prediction.

## New Email
{email_text}

## Historical Context (Similar Past Emails)
{rag_context}

## ML Model Prediction
- Predicted Priority: {model_prediction}
- Confidence: {model_confidence:.1%}
- Full Distribution: {model_probabilities}

## Instructions
Based on the new email content, the historical context, and the ML model's prediction, determine the final priority classification.

Consider:
1. Urgency keywords (urgent, ASAP, deadline, immediately)
2. Action requests (please review, need approval, schedule meeting)
3. Sender importance (executives, managers, automated systems)
4. Similarity to past high-priority emails
5. The ML model's confidence level

Respond with EXACTLY this format:
Priority: [High/Medium/Low]
Confidence: [0.0 to 1.0]
Reasoning: [One sentence explaining why]"""

    @staticmethod
    def format_rag_context(retrieved_emails: list[dict]) -> str:
        """Format retrieved historical emails into a readable context block."""
        if not retrieved_emails:
            return "No similar historical emails found."

        lines = []
        for i, email in enumerate(retrieved_emails, 1):
            subject = email.get("subject", "N/A")
            snippet = email.get("body_snippet", "N/A")
            label = email.get("priority_label", "N/A")
            score = email.get("similarity_score", 0.0)

            lines.append(
                f"{i}. [Priority: {label}] (Similarity: {score:.3f})\n"
                f"   Subject: {subject}\n"
                f"   Snippet: {snippet}"
            )

        return "\n".join(lines)

    @classmethod
    def build_prompt(
        cls,
        email_text: str,
        retrieved_emails: list[dict],
        model_prediction: str,
        model_confidence: float,
        model_probabilities: dict,
    ) -> str:
        """
        Assemble the full prompt for the generative LLM.

        Args:
            email_text:          The new email to classify.
            retrieved_emails:    List of similar historical emails from RAG.
            model_prediction:    Fine-tuned model's predicted label.
            model_confidence:    Fine-tuned model's confidence (0-1).
            model_probabilities: Dict of {label: probability}.

        Returns:
            Formatted prompt string.
        """
        rag_context = cls.format_rag_context(retrieved_emails)

        prob_str = ", ".join(
            f"{k}: {v:.1%}" for k, v in model_probabilities.items()
        )

        return cls.CLASSIFICATION_TEMPLATE.format(
            email_text=email_text,
            rag_context=rag_context,
            model_prediction=model_prediction,
            model_confidence=model_confidence,
            model_probabilities=prob_str,
        )
