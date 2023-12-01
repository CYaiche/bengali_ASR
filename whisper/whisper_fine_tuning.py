
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
from datetime import datetime

# *************** Load whisper bengali transformers  *************** # 
model_path                      = "bangla-speech-processing/BanglaASR"
whisper_processor               =  WhisperProcessor.from_pretrained(model_path)
model                           = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
model.config.suppress_tokens    = []

# *************** Prepare dataloaders   *************** # 
audio_folder_pth         = "/disk3/clara/bengali/train_mp3s"
df_train                 = pd.read_csv("/home/nxp66145/clara/whisper_train.csv")
df_val                   = pd.read_csv("/home/nxp66145/clara/whisper_val.csv")

dry_run = True
if dry_run : 
    df_train                 = pd.read_csv("/home/nxp66145/clara/whisper_train_sample_100.csv")
    df_val                   = pd.read_csv("/home/nxp66145/clara/whisper_val_sample_100.csv")

train_dataset   = WhisperCustomDataset(audio_folder_pth, df_train, whisper_processor)
val_dataset     = WhisperCustomDataset(audio_folder_pth, df_val, whisper_processor)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=whisper_processor,bos_id=50258 ) # in benglaASR bos as been overwritten

# *************** Training : fine-tuning   *************** # 
test_explanation = "dry_run_librosa_resampling_no_remove_bos_momasking"
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"/disk2/clara/whisper/{test_explanation}_{run_name}"

if not os.path.isdir(save_dir) :
    os.mkdir(save_dir)
print(f"save_dir : {save_dir}")

training_args = Seq2SeqTrainingArguments(
    output_dir=save_dir,  # change to a repo name of your choice
    per_device_train_batch_size=8,# 16 
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500, # 500,
    max_steps=2000, # 4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500, # 1000,
    eval_steps=500, # 1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

if dry_run : 
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,  # change to a repo name of your choice
        per_device_train_batch_size=4,# 16 
        gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
        learning_rate=0,
        warmup_steps=10, # 500,
        max_steps=100, # 4000,
        gradient_checkpointing=True,
        fp16=False,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=1024,
        save_steps=50, # 1000,
        eval_steps=10, # 1000,
        logging_steps=2,
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
    compute_metrics=lambda x: compute_metrics(x, whisper_processor.tokenizer),
    tokenizer=whisper_processor.feature_extractor,
)

trainer.train()



trainer.save_model(save_dir)
