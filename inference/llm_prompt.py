"""
Prompt templates for the generative LLM (Phase 3).

Uses Chain-of-Thought (CoT) prompting (Wei et al., 2022) to elicit
step-by-step reasoning from the LLM before it commits to a final
classification. This makes the decision process transparent and
interpretable.

Builds a structured prompt that includes:
  1. The new email text
  2. Retrieved historical context from RAG (similar past emails + their priorities)
  3. The fine-tuned classifier's prediction + confidence
  4. Instructions for step-by-step reasoning followed by a final answer
"""


class PromptBuilder:
    """Build prompts for contextual email priority classification."""

    CLASSIFICATION_TEMPLATE = """You are an expert email priority classifier. Analyze the email step-by-step before deciding.

## New Email
{email_text}

## Historical Context (Similar Past Emails)
{rag_context}

## ML Model Prediction
- Predicted Priority: {model_prediction}
- Confidence: {model_confidence:.1%}
- Full Distribution: {model_probabilities}

## Instructions
Think step by step. For each step, write a short analysis.

Step 1 - Urgency: Are there urgency keywords like urgent, ASAP, deadline, immediately, critical? How time-sensitive is this email?
Step 2 - Action Required: Does the email request a specific action like review, approve, respond, schedule, or submit?
Step 3 - Sender Context: Does the sender appear to be an executive, manager, or automated system?
Step 4 - Historical Pattern: Based on the similar past emails above, what priority did similar emails receive?
Step 5 - Model Agreement: The ML model predicted {model_prediction} with {model_confidence:.1%} confidence. Do you agree or disagree based on the above analysis?

After your analysis, give the final answer in EXACTLY this format:
Priority: [High/Medium/Low]
Confidence: [0.0 to 1.0]
Reasoning: [One sentence summary]

Begin your step-by-step analysis:"""

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
