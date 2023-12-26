# Speech Recognition 🗣️ in Bengali 🇧🇩🇮🇳

## Overview

ASR project for Bengali, exploring Automatic Speech Recognition (ASR) techniques.

ASR system transcribes speech into text, emphasizing the probabilistic sequence of words and sounds.

Two approaches: statistical and end-to-end. Acoustic Model and Lexicon play key roles.

I focus here on the **end-to-end** approach. 

### Bengali Language

- 09 vowels, 39 consonants
- Official language of Bangladesh
- 2nd most spoken language in India

Spoken by 200M+ people, featuring diverse dialects.

## Competition Objective

Handling out-of-distribution data and linguistic nuance, improve WER.

### Training Database

- 1180 hours of audio
- Data cleaning: maintain WER < 70%
- Preprocessing: Mel-Log Spectrograms, tokenized embeddings

## Models Explored

### RRN-T Transducer 🔄

PyTorch implementation with transducer approach.

### Whisper by OpenAI 🤫

40% WER on Common Voice, 10% in French.

### Results 📚

- Baseline vs. Whisper vs. BenglaASR
- Whisper-small fine-tuned: **WER 67%** achieve lower WER then the baseline.

More details  presentation slides in French [here](https://github.com/CYaiche/bengali_ASR/blob/master/Presentation.pdf)

**Sources:**
- [ASR Overview](https://people.irisa.fr/Gwenole.Lecorve/lectures/ASR.pdf)
- [Transducer Implementation](https://lorenlugosch.github.io/posts/2020/11/transducer/)
- [Speech Recognition in ML](https://maelfabien.github.io/machinelearning/speech_reco/#statistical-historical-approach-to-asr)




