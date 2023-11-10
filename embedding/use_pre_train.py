
import json
import pyctcdecode
import os
import torch 
from common.common_params import BENG_DIR
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import Wav2Vec2Processor, Wav2Vec2Tokenizer,  Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC

def get_vocab_pretrained() : 
    processor = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")

    vocab_dict = processor.tokenizer.get_vocab()


    print(vocab_dict)
    #add missing character 

    vocab_dict["\u09dd"] = len(vocab_dict)  # ঢ়'

    # Writing to sample.json
    emb_path = os.path.join(BENG_DIR,"embedding")
    assert os.path.isdir(emb_path), f"path does not exist : {emb_path}"

    filepath = os.path.join(emb_path,"vocab_indicwav2vec.json")

    with open(filepath, "w") as outfile:
        outfile.write(json.dumps(vocab_dict))

    sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}


def test_tokenizer():
    vocab_path = os.path.join(BENG_DIR, "embedding", "vocab_indicwav2vec.json")
    tokenizer = ByteLevelBPETokenizer(
        vocab_path
    )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)
    code = tokenizer.encode(" ভঙ্গীমার বিষয়ে সন্দেহ থাকায় তার ")
    
    print(code)
    print(code.tokens)
    
from transformers import PreTrainedTokenizer
def test():
    processor =  Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")

    # vocab_path = os.path.join(BENG_DIR, "embedding", "vocab_indicwav2vec.json")
    # tokenizer = PreTrainedTokenizer(tokenizer_file=vocab_path)
    # tokenizer.enable_truncation(max_length=512)
    code = processor(" ভঙ্গীমার বিষয়ে সন্দেহ থাকায় তার ")
    
    print(code)
    print(code.tokens)
    
from bltk.langtools.banglachars import (vowels,
                                            vowel_signs,
                                            consonants,
                                            digits,
                                            operators,
                                            punctuations,
                                            others)
def test_bltk():
    print(f'Vowels: {vowels}')
    print(f'Vowel signs: {vowel_signs}')
    print(f'Consonants: {consonants}')
    print(f'Digits: {digits}')
    print(f'Operators: {operators}')
    print(f'Punctuation marks: {punctuations}')
    print(f'Others: {others}')
# test_tokenizer()

if __name__ == "__main__" : 
    get_vocab_pretrained()
    # test_bltk().replace("", " ")
    
    vocabulary_location = os.path.join(BENG_DIR, "embedding","vocab_indicwav2vec.json")
    with open(vocabulary_location) as vocabulary_file:
        char_voc = vocabulary_file.read()
    vocabulary_to_id = json.loads(char_voc) 
    null_index = vocabulary_to_id["<pad>"]
    vocabulary_size = len(vocabulary_to_id) 

    torch.save(vocabulary_size , os.path.join(BENG_DIR, "embedding",'vocabulary_size.pt'))
