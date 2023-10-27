from transformers import Wav2Vec2Processor, Wav2Vec2Tokenizer,  Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
import json
import pyctcdecode
import os
from common.common_params import STT_DIR


def get_vocab_pretrained() : 
    processor = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec_v1_bengali")

    vocab_dict = processor.tokenizer.get_vocab()


    print(vocab_dict)
    # Writing to sample.json
    filepath = os.path.join(STT_DIR,"embedding","vocab_indicwav2vec.json")

    with open(filepath, "w") as outfile:
        outfile.write(json.dumps(vocab_dict))

    sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
def test_tokenizer():
    vocab_path = os.path.join(STT_DIR, "embedding", "vocab_indicwav2vec.json")
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

    # vocab_path = os.path.join(STT_DIR, "embedding", "vocab_indicwav2vec.json")
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
test_bltk()s.replace("", " ")