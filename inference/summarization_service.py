import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
from config import settings

class SummarizationService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = settings.SUMMARIZATION_MODEL_NAME
        self.lora_dir = settings.LORA_WEIGHTS_DIR
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the base model and LoRA adapter weights if available."""
        try:
            print(f"Loading summarization tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            print(f"Loading base summarization model: {self.model_name}")
            # Load base model
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32 if self.device.type == 'cpu' else torch.float16
            )
            
            # Check if LoRA weights exist
            if os.path.exists(self.lora_dir) and os.listdir(self.lora_dir):
                print(f"Loading LoRA adapter weights from: {self.lora_dir}")
                self.model = PeftModel.from_pretrained(base_model, self.lora_dir)
            else:
                print("No LoRA weights found. Falling back to zero-shot summarization with base model.")
                self.model = base_model
                
            self.model.to(self.device)
            self.model.eval()
            print("Summarization model loaded successfully.")
        except Exception as e:
            print(f"Error loading summarization model: {e}")

    def generate_summary(self, email_text, max_length=150, min_length=30):
        """Generates a summary for the given email text."""
        if not self.model or not self.tokenizer:
            return "Summarization model not loaded."

        # Truncate text if it's too long
        prompt = f"Summarize the following email:\n\n{email_text}\n\nSummary:"
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Failed to generate summary."
