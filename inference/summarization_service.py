import os
from config import settings

class SummarizationService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.error = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from peft import PeftModel, PeftConfig
            
            # Using flan-t5-base as our base summarizer
            model_name = "google/flan-t5-base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Just use base model (zero-shot) because the 50-sample LoRA hurts performance
            self.model = base_model
                
        except ImportError as e:
            self.error = "PyTorch or Transformers not found. Please fix the local DLL crash."
        except Exception as e:
            self.error = f"Error loading summarizer: {e}"

    def summarize(self, email_body: str) -> str:
        if self.error:
            return f"❌ Summarization unavailable: {self.error}"
            
        try:
            prompt = "Extract the key action items and summary from this email: " + email_body
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(**inputs, max_length=150, min_length=15, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            return f"Error during summarization: {e}"
