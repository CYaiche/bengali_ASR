import jiwer 
from transformers import WhisperProcessor

def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    print(pred_str)
    print(label_str)
    wer =  jiwer.wer(label_str, pred_str)

    return {"wer": wer}


if __name__ == "__main__" : 
    proc = WhisperProcessor.from_pretrained("bangla-speech-processing/BanglaASR")
    pred = {"predictions" : [[50258, 50363, 29045, 227, 50257,50257]],
            "label_ids" : [[50258, 50363, 29045, 227, 50257, 50257, 50257, 50257, 50257]]}
    pred = type('allMyFields', (object,), pred)
    w  = compute_metrics(pred, proc.tokenizer)
    print(w)

