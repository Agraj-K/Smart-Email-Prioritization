import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType

def main():
    print("Loading cnn_dailymail dataset...")
    # Load the CNN/DailyMail dataset (for paragraph/bullet summaries)
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Preprocessing function
    def preprocess_function(examples):
        inputs = ["summarize: " + doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)

        labels = tokenizer(text_target=examples["highlights"], max_length=64, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    print("Loading base model (flan-t5-base)...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Setup LoRA for extremely efficient fine-tuning
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Define training arguments
    output_dir = os.path.join("data", "summarizer_model")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=False, # Set to True if using GPU
        push_to_hub=False,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"].select(range(50)),
        eval_dataset=tokenized_datasets["validation"].select(range(10)),
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving fine-tuned LoRA weights to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
