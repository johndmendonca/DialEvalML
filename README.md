# DialEvalML

This repo implements the Paper [*Towards Multilingual Dialogue Evaluation*](). It also includes competition code for DSTC11 Track 4 *Robust and Multilingual Automatic Evaluation Metrics for Open-Domain Dialogue Systems*, which is introduced in the [*Simple LLM Prompting is State-of-the-Art for Robust and Multilingual Dialogue Evaluation]() paper.



| Section |
|-|
| [Apply DialEvalML to your own dialogues](#apply-dialevalml-to-your-own-dialogues) |
| [Training encoder models from scratch](#training-encoder-models-from-scratch) |
| [Citation](#citation) |


## Apply DialEvalML to your own dialogues

### 1. Download models

* Obtain the models directly from [Drive]().

### 2. Apply Tokenization

Raw dialog corpora may have very different data structures, so we leave to the user to convert their own data to the DialQualityML format.

The format is straightforward:

* The tokenizer receives as input `res` and optionally `ctx` (context is needed to evaluate context dependent metrics). 
* `ctx` can be multi-turn, the only limitation relates to `max_length`. 
* Who said what is determined by appending the speaker token at the start of the sentence.

~~~
A: Gosh, you took all the word right out of my mouth. Let's go out and get crazy tonight.
B: Let's go to the new club on West Street .
A: I'm afraid I can't.


ctx = "<speaker1>Gosh , you took all the word right out of my mouth . Let's go out and get crazy tonight .</s><s><speaker2>Let's go to the new club on West Street ."
res = "<speaker1>I ' m afraid I can ' t ."
~~~

### 3. Run prediction code

* Adjust the script `predict.py` to your requirements.

## Training encoder models from scratch
### 1. Download/format data

* You can download the [preprocessed data](https://drive.google.com/file/d/1MQyLWVKRBmKy3eZG0aVEh23hYmyz2heC/view?usp=sharing) used to train our models. This data is already preprocessed with seperate columns for context and response and follows the DialEvalML format.  

### 2. Training

* Adjust and run `train_xxx.sh`.

## Citation

If you use this work, please consider citing:

~~~
to do
~~~
