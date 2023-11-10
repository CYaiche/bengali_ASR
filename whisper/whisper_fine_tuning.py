
import os
import torch
import torchaudio
import pandas as pd
import jiwer 

from whisper.WhisperCustomDataset import WhisperCustomDataset, DataCollatorSpeechSeq2SeqWithPadding
from whisper.whisper_eval import compute_metrics
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration

from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

# *************** Load whisper bengali transformers  *************** # 
model_path          = "bangla-speech-processing/BanglaASR"
whisper_processor           =  WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# *************** Prepare dataloaders   *************** # 
audio_folder_pth         = "/disk3/clara/bengali/train_mp3s"
df_train    = pd.read_csv("/home/nxp66145/clara/whisper_train.csv")
df_val      = pd.read_csv("/home/nxp66145/clara/whisper_test.csv")


train_dataset = WhisperCustomDataset(audio_folder_pth, df_train, whisper_processor)
val_dataset     = WhisperCustomDataset(audio_folder_pth, df_val, whisper_processor)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=whisper_processor)

# *************** Training : fine-tuning   *************** # 


training_args = Seq2SeqTrainingArguments(
    output_dir="/disk2/clara/whisper",  # change to a repo name of your choice
    per_device_train_batch_size=8,# 16 
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=50, # 500,
    max_steps=500, # 4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100, # 1000,
    eval_steps=100, # 1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=whisper_processor.feature_extractor,
)

trainer.train()