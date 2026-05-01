import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import os
import re
import warnings

# Silence the noisy per-row length conflict warnings
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*min_new_tokens.*min_length.*")

# Import your latest Preprocessing module
from Preprocessing import Preprocessing

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_FILE  = "final_analytics_data_bart.csv"
MODEL_NAME   = "facebook/bart-large-cnn"
BATCH_SIZE   = 8      # ← Tune this: 8 for 8GB VRAM, 16 for 16GB+, 4 for 6GB
MAX_IN_LEN   = 512    # Encoder token cap (512 is sweet spot for BART-large speed)
MAX_NEW_TOK  = 60     # Decoder token cap — enough for a full sentence
MIN_NEW_TOK  = 15     # Prevent empty/trivial outputs


class ThirdPersonSummarizer:
    def __init__(self):
        # ── 1. CUDA Memory Config ──────────────────────────────────────────────
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU] {gpu_name} | {vram_gb:.1f} GB VRAM")
        else:
            print("[WARNING] No CUDA GPU found — running on CPU (will be slow)")

        # ── 2. Load Model ──────────────────────────────────────────────────────
        print(f"[Stage 4] Loading {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
        )

        if self.device.type != "cuda":
            self.model = self.model.to(self.device)

        # ── FIX: Clear BART's hardcoded length defaults to silence spam warnings
        self.model.generation_config.max_length          = None
        self.model.generation_config.min_length          = None
        self.model.generation_config.forced_bos_token_id = 0

        self.model.eval()

        if hasattr(torch, "compile"):
            print("[GPU] torch.compile() enabled — first batch will be slow (warmup).")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.bos_id = self.model.config.decoder_start_token_id

    # ── Helpers ────────────────────────────────────────────────────────────────
    def get_clean_name(self, email_addr: str) -> str:
        if not email_addr or "@" not in str(email_addr):
            return "The sender"
        local = email_addr.split("@")[0]
        return re.sub(r"[._\-]+", " ", local).title().strip()

    # ── FIX 2: Garbage email detector ─────────────────────────────────────────
    def _is_garbage(self, text: str) -> bool:
        """
        Reject emails that are system logs, raw data dumps, or too short.
        These cause the model to hallucinate or copy-paste junk.
        """
        if not isinstance(text, str):
            return True
        words = text.split()
        if len(words) < 5:
            return True
        # If more than 20% of tokens are long number strings → skip
        digit_tokens = sum(1 for w in words if re.search(r'\d{4,}', w))
        if digit_tokens / len(words) > 0.2:
            return True
        # If more than 30% of tokens are ALL-CAPS (system log style) → skip
        caps_tokens = sum(1 for w in words if w.isupper() and len(w) > 2)
        if caps_tokens / len(words) > 0.3:
            return True
        return False

    # ── FIX 1: Pronoun swap that skips quoted speech ───────────────────────────
    def _fix_sentence(self, text: str, sender: str) -> str:
        if not text:
            return ""

        # Split on quoted parts — don't touch pronouns inside quotes
        parts = re.split(r'(".*?")', text)
        fixed = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                fixed.append(part)   # inside quotes → leave alone
            else:
                part = re.sub(r"\bI\b",   sender,  part)
                part = re.sub(r"\bmy\b",  "their", part, flags=re.IGNORECASE)
                part = re.sub(r"\bme\b",  sender,  part, flags=re.IGNORECASE)
                part = re.sub(r"\bwe\b",  sender,  part, flags=re.IGNORECASE)
                part = re.sub(r"\bour\b", "their", part, flags=re.IGNORECASE)
                fixed.append(part)

        text = "".join(fixed).strip()
        if text:
            text = text[0].upper() + text[1:]

        # Trim incomplete ending
        last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last_punct != -1 and last_punct < len(text) - 1:
            trailing = text[last_punct + 1:].strip()
            if len(trailing) < 4 or not re.search(r"[aeiou]", trailing, re.I):
                text = text[: last_punct + 1]

        # Prepend sender if BART didn't naturally include them
        if not text.lower().startswith(sender.lower()):
            text = f"{sender} {text[0].lower() + text[1:]}"

        return text.strip()


    # ── Batch summarization ────────────────────────────────────────────────────
    @torch.inference_mode()
    def summarize_batch(self, texts: list[str], senders: list[str]) -> list[str]:
        enc = self.tokenizer(
            texts,
            max_length=MAX_IN_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        results = []
        for i in range(len(texts)):
            single_input_ids      = enc["input_ids"][i].unsqueeze(0)
            single_attention_mask = enc["attention_mask"][i].unsqueeze(0)

            # FIX: use torch.amp.autocast — torch.cuda.amp.autocast is deprecated
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                summary_ids = self.model.generate(
                    input_ids=single_input_ids,
                    attention_mask=single_attention_mask,
                    max_new_tokens=MAX_NEW_TOK,
                    min_new_tokens=MIN_NEW_TOK,
                    num_beams=4,
                    repetition_penalty=2.0,       # ← CHANGED (was 2.5)
                    no_repeat_ngram_size=3,       # ← CHANGED (was 4)
                    do_sample=False,
                    length_penalty=1.0,           # ← CHANGED (was 0.8)
                    early_stopping=True,
)

            decoded = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            results.append(self._fix_sentence(decoded, senders[i]))

        return results

    # ── Public entry point ─────────────────────────────────────────────────────
    def process_dataframe(self, df: pd.DataFrame) -> pd.Series:
        summaries = [""] * len(df)
        texts     = df["clean_body_summary"].tolist()
        senders   = [self.get_clean_name(e) for e in df["from"].tolist()]

        for start in tqdm(range(0, len(df), BATCH_SIZE), desc="[Stage 4] Summarising"):
            end         = min(start + BATCH_SIZE, len(df))
            batch_texts = texts[start:end]
            batch_sndrs = senders[start:end]

            # FIX 2: Use garbage detector instead of just checking length
            valid_mask  = [not self._is_garbage(t) for t in batch_texts]
            valid_texts = [t for t, v in zip(batch_texts, valid_mask) if v]
            valid_sndrs = [s for s, v in zip(batch_sndrs, valid_mask) if v]

            if valid_texts:
                batch_results = self.summarize_batch(valid_texts, valid_sndrs)

            result_iter = iter(batch_results if valid_texts else [])
            for j, (_, valid) in enumerate(zip(batch_texts, valid_mask)):
                summaries[start + j] = next(result_iter) if valid else ""

            if self.device.type == "cuda" and (start // BATCH_SIZE) % 50 == 0:
                torch.cuda.empty_cache()

        return pd.Series(summaries, index=df.index)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Preprocessing
    p = Preprocessing(sample_size=1000)
    p.apply_parse()
    p.apply_cleaning()
    df = p.df

    # 2. Summarizer
    s = ThirdPersonSummarizer()
    print(f"\n[Stage 4] Processing {len(df)} emails in batches of {BATCH_SIZE}...")
    df["summary"] = s.process_dataframe(df)

    # 3. GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[GPU] Peak VRAM used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # 4. Export
    final_cols = ["date", "from", "subject", "summary"]
    df[final_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\n[✓ Success] Saved → {OUTPUT_FILE}")