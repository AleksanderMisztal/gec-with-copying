# Project proposal

## Introduction

Grammatical error correction (GEC) is the task of automatically correcting grammatical errors in written text. It is useful as feedback for language learners as well as to correct inadvertent mistakes made by proficient speakers.

The aim of this project is to create a GEC system based on neural machine translation (NMT) - automatically translating the incorrect sentence into a correct one - extended with a coping mechanism - instead of translating a word, the system can copy a word from the input.

While at first it might seem surprising to phrase GEC as NMT, it offers many benefits. It can handle all types of errors simultaneously and is very general which means that the techniques from other tasks can be transferred directly.

Copying is beneficial because the translation only modifies the parts of the sentence that are incorrect and therefore most of the input stays unchanged.

## Starting point

### Existing research

This project will develop a system similar to that in [Zhao et al., 2019](https://aclanthology.org/N19-1014.pdf). They use a transformer-based encoder-decoder. The copying mechanism is implemented as an additional attention layer from the decoder's current hidden state to the encoder's states. The probability distribution of the next word is a mixture of the generation and copying distributions.

### Software

I will use the following libraries: nltk or similar for data preprocessing, hugging face tokenizers for tokenization, pytorch to implement the models, and possibly others, including pretrained models.

### Datasets

I will use the [BEA-2019 GEC dataset](https://aclanthology.org/W19-4406.pdf). For training the larger models, it may also be necessary to add the private Cambridge Assessment data and/or synthetic data.

## Project structure

### Objectives

The goal of the project is to build an NMT GEC system based on transformer architecture extended with copying, such as in Zhao et al., while also gaining an understanding of different aspects of developing NLP models and techniques used to optimise performance.

I will also implement a transformer from scratch and aim to match the performance of the library implementations.

I will evaluate the models using objective metrics such as GLEU or $M^2$, as well as visually explore the generated corrections to understand the limitations of my models.

### Possible extensions

Possible extensions include benchmarking against other models such as LSTMs or sth like Levenshtein transformers and applying the same technique to other tasks, such as summarization or paraphrase generation.

## Success criteria

The project will be considered successful if:

1. The nmt + copying model has been implemented,

2. The performance of the model has been evaluated qunatitatively using a metric such as GLEU or $M^2$, and qualitatively.

## Timetable (milestones and deadlines)

Do the background reading, familiarise with datasets and previous work, plan the exact model architecture 15-31 October

Implement the data preprocessing, evaluation, and a simple baseline model to compare against, possibly using a library implementation 1-20 November

Implement and train the transformer based nmt model 21 November - 15 December

Implement the copying mechanism 15 December - 31 December

Finish any outstanding success criteria, reevaluate the feasibility of extensions and choose the most promising ones, create a working document and draft the introduction 1 January - 3 February

Progress report deadline - 4 February

Implement the chosen extensions, draft the rest of the disserations 4 February - 31 March

Iteratively apply corrections to the dissertation based on feedback, incorporate results from extensions 1 March - 13 May

Dissertation deadline - 13 May

## Resource declaration

I will use my own machine for development and running tasks that don't require heavy compute on specialized hardware. I will use google colab for training the smaller models/running experiments. As an alternative, in case colab's usage limits fluctuate, and for training the larger models towards the end of the project, I will request access to the HPC.

I will use github for storing the version history,

I accept full responsibility for this machine and I have made contingency plans to protect myself against hardware and/or software failure.
