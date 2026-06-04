"""
Email text cleaner.

Provides two cleaning pipelines:
  - clean_for_summarization: Minimal cleaning (remove replies, signatures, URLs)
  - clean_for_classification: Aggressive cleaning (lowercase, strip special chars)
"""

import re


class Cleaner:
    """Stateless email body cleaning utilities."""

    # ── Reply / Forward Removal ───────────────────────────────────────────────
    def remove_replies(self, text: str) -> str:
        """Strip quoted reply blocks, forwarded-message headers, and
        Enron-specific internal forwarding patterns."""
        # Split on standard patterns AND Enron-internal forwarding headers
        # e.g., "Sharron Westbrook @ ENRON 08/18/2000 04:22 PM"
        # e.g., "From: Name/ENRON@enronXgate on MM/DD/YYYY HH:MM AM"
        text = re.split(
            r'-----Original Message-----'
            r'|On [^\n]+ wrote:'
            r'|(?:-{5,}\s*Forwarded by)'
            r'|\S+\s*@\s*ENRON\s+\d{2}/\d{2}/\d{4}'
            r'|From:\s+[a-zA-Z0-9.\s-]+/ENRON'
            r'|From:\s+[a-zA-Z0-9.\s-]+@enron',
            text,
            flags=re.IGNORECASE,
        )[0]
        text = re.sub(r'>.*', '', text, flags=re.MULTILINE)
        return text

    # ── Email Metadata Removal ────────────────────────────────────────────────
    def remove_email_metadata(self, text: str) -> str:
        """Remove residual email header lines and Enron routing paths."""
        # Remove lines starting with From:, To:, cc:, Subject:, Sent:, Date:
        text = re.sub(
            r'^\s*(From|To|cc|bcc|Subject|Sent|Date):\s*.*$',
            '', text, flags=re.MULTILINE | re.IGNORECASE,
        )
        # Remove Enron-style routing paths (e.g., /HOU/ECT@ECT, /Corp/Enron@ENRON)
        text = re.sub(r'/[a-zA-Z0-9]+/[a-zA-Z0-9]+@[a-zA-Z0-9]+', '', text)
        # Remove standalone email addresses
        text = re.sub(r'\b[\w.+-]+@[\w.-]+\.\w+\b', '', text)
        # Remove timestamp patterns (MM/DD/YYYY HH:MM AM/PM)
        text = re.sub(
            r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s*(?:AM|PM)?',
            '', text, flags=re.IGNORECASE,
        )
        return text

    # ── Signature Removal ─────────────────────────────────────────────────────
    def remove_signature(self, text: str) -> str:
        """Remove trailing signature blocks using common marker phrases."""
        lines = text.split("\n")
        sig_markers = re.compile(
            r'^(--|best regards|regards|thanks|thank you|sincerely|cheers|'
            r'sent from my|get your (private|free).*(email|e-mail)|'
            r'share information about yourself)',
            re.IGNORECASE,
        )
        cutoff = len(lines)
        for i in range(len(lines) - 1, max(len(lines) - 15, -1), -1):
            if sig_markers.search(lines[i].strip()):
                cutoff = i
                break
        return "\n".join(lines[:cutoff])

    # ── URL Removal ───────────────────────────────────────────────────────────
    def remove_urls(self, text: str) -> str:
        """Remove http/https/www URLs."""
        return re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)

    # ── Special Character Removal ─────────────────────────────────────────────
    def remove_special_chars(self, text: str) -> str:
        """Keep only alphanumeric characters and whitespace."""
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # ── Usability Check ───────────────────────────────────────────────────────
    def is_usable(self, text: str) -> bool:
        """Reject empty, near-empty, or system/boilerplate emails."""
        if not isinstance(text, str) or len(text.split()) < 5:
            return False
        boilerplate = re.search(
            r'(immediate action required|do not delete|'
            r'please click on the following link|unsubscribe|'
            r'unique id.*participant)',
            text,
            re.IGNORECASE,
        )
        return not boilerplate

    # ── Pipeline: Summarization Cleaning ──────────────────────────────────────
    def clean_for_summarization(self, text: str) -> str:
        """Minimal cleaning — preserves natural language structure."""
        if not isinstance(text, str):
            return ""
        text = self.remove_replies(text)
        text = self.remove_email_metadata(text)
        text = self.remove_signature(text)
        text = self.remove_urls(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\n+', ' ', text)
        return text if self.is_usable(text) else ""

    # ── Pipeline: Classification Cleaning ─────────────────────────────────────
    def clean_for_classification(self, text: str) -> str:
        """Aggressive cleaning — lowercased, no special chars."""
        text = self.clean_for_summarization(text)
        if not text:
            return ""
        text = text.lower()
        text = self.remove_special_chars(text)
        return text
