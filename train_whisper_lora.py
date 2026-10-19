import argparse
import json
import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Cut boss token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def train_lora(dataset_dir="dataset", model_id="openai/whisper-small", output_dir="whisper-lora-adapter", epochs=3, lr=1e-3, lang="german", task="translate"):
    print("=" * 65)
    print("WHISPER LoRA FINE-TUNING ON APPLE SILICON")
    print("=" * 65)
    print(f"[*] Base Model:      {model_id}")
    print(f"[*] Dataset:         {dataset_dir}/train.jsonl")
    print(f"[*] Target Language: {lang}")
    print(f"[*] Task:            {task}")
    print(f"[*] Epochs:          {epochs}")

    jsonl_path = os.path.join(dataset_dir, "train.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"[Error]: {jsonl_path} not found. Run youtube_dataset_builder.py first!", file=sys.stderr)
        return

    # Load dataset
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # Make audio path absolute
            data["audio"] = os.path.abspath(os.path.join(dataset_dir, data["audio"]))
            records.append({"audio": data["audio"], "sentence": data["sentence"]})

    print(f"[*] Loaded {len(records)} audio-sentence training pairs.")

    raw_dataset = Dataset.from_list(records)
    raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))

    processor = WhisperProcessor.from_pretrained(model_id, language=lang, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # Freeze base model
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # Configure LoRA (Low-Rank Adaptation)
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    print("[*] Pre-processing audio features and tokenizing transcripts...")
    processed_dataset = raw_dataset.map(prepare_dataset, remove_columns=["audio", "sentence"])

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Use Apple Silicon GPU (MPS) if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[*] Training Device: {device.upper()}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        warmup_steps=10,
        max_steps=epochs * (len(records) // 4 + 1),
        gradient_checkpointing=True,
        fp16=False,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_steps=5,
        report_to=["none"],
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=processed_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    print("\n[*] Starting LoRA fine-tuning...")
    trainer.train()

    print(f"\n[*] Training complete! Saving LoRA adapter to '{output_dir}'...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"[*] Successfully saved adapter! You can now load '{output_dir}' for inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA on an M1 Mac.")
    parser.add_argument("-d", "--dataset-dir", type=str, default="dataset", help="Path to dataset directory.")
    parser.add_argument("-m", "--model", type=str, default="openai/whisper-small", help="Hugging Face model ID.")
    parser.add_argument("-o", "--output-dir", type=str, default="whisper-lora-adapter", help="Output path for LoRA adapter.")
    parser.add_argument("-e", "--epochs", type=int, default=3, help="Training epochs (default: 3).")
    parser.add_argument("-l", "--lang", type=str, default="german", help="Language name (default: german).")
    parser.add_argument("-t", "--task", type=str, default="translate", choices=["translate", "transcribe"], help="Task.")
    args = parser.parse_args()

    train_lora(
        dataset_dir=args.dataset_dir,
        model_id=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lang=args.lang,
        task=args.task,
    )
