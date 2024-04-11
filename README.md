# CS 224N Project

## Abstract

Pre-trained BERT (Devlin et al., 2018) has demonstrated incredible performances in numerous downstream language tasks when the model is fine-tuned using pre- trained weights on downstream tasks. Techniques to further improve BERT embed- dings are desired to not only improve the performance of downstream tasks, but also to gain insight into how the BERT model behaves and explore avenues for fur- ther improvement of language models. This paper explores a number of techniques applied to BERT on four downstream tasks: sentiment classification, paraphrase detection, semantic textual similarity, and linguistic acceptability. The techniques explored include: per-layer learning rate, LoRA, further pre-training, and multitask fine-tuning (Sun et al., 2019) (Hu et al., 2021). We find that on difficult tasks with low performance such as sentiment classification, further pre-training has a big impact on performance, and we find that pre-training on different domain datasets and multitask fine-tuning can exhibit transfer learning properties by improving performance of related tasks. Additionally, we find that LoRA applied on a general pre-trained model achieves similar performance as full fine-tuning, but achieves much lower performance on a less general model that is pre-trained on task-specific data. The experiments from this paper support the prospects and potential of im- proved performance from scaling exhibited by many recent LLM research which pursue scaling model sizes, dataset diversity, and modalities diversity (Aghajanyan et al., 2023).

Read the [paper](Report.pdf)

# CS 224N Default Final Project 2024 - Multitask BERT

This is the default final project for the Stanford CS 224N class. Please refer to the project handout on the course website for detailed instructions and an overview of the codebase.

This project comprises two parts. In the first part, you will implement some important components of the BERT model to better understand its architecture. 
In the second part, you will use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection, and semantic similarity. You will implement extensions to improve your model's performance on the three downstream tasks.

In broad strokes, Part 1 of this project targets:
* bert.py: Missing code blocks.
* classifier.py: Missing code blocks.
* optimizer.py: Missing code blocks.

And Part 2 targets:
* multitask_classifier.py: Missing code blocks.
* datasets.py: Possibly useful functions/classes for extensions.
* evaluation.py: Possibly useful functions/classes for extensions.

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.

## Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
