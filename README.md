# DialEvalML

This repo implements the Paper [*Towards Multilingual Automatic Open-Domain Dialogue Evaluation*](). It also includes competition code for DSTC11 Track 4 *Robust and Multilingual Automatic Evaluation Metrics for Open-Domain Dialogue Systems*, which is introduced in the [*Simple LLM Prompting is State-of-the-Art for Robust and Multilingual Dialogue Evaluation*]() paper.



| Section |
|-|
| [Apply DialEvalML to your own dialogues](#apply-dialevalml-to-your-own-dialogues) |
| [Training encoder models from scratch](#training-encoder-models-from-scratch) |
| [Citation](#citation) |


## Apply DialEvalML to your own dialogues

### 1. Download models

Obtain the models used in both papers from Drive.

* *Towards Multilingual Automatic Open-Domain Dialogue Evaluation*: [VSP-ML5](https://ulisboa-my.sharepoint.com/:u:/g/personal/ist425406_tecnico_ulisboa_pt/ESVlq-NJPwRFolZNjg0gnZMB5F4d9z_BNCJeLtk24UXtKA?e=hlrrdW), [NSP-ML75](https://ulisboa-my.sharepoint.com/:u:/g/personal/ist425406_tecnico_ulisboa_pt/EX45rKN3eVFGuWJz5bIu3AkBglITGjG3eywCF30QqCZzAQ?e=HWj9L8);
* *Simple LLM Prompting is State-of-the-Art for Robust and Multilingual Dialogue Evaluation*:
  * [VSP-EN](https://ulisboa-my.sharepoint.com/:u:/g/personal/ist425406_tecnico_ulisboa_pt/EeJUZeUvj7dCswVQuz9P2TUB9AeT19o0_ebety6uoTZeSQ?e=IgUurZ), [VSP-PA](https://ulisboa-my.sharepoint.com/:u:/g/personal/ist425406_tecnico_ulisboa_pt/EbdlLnFSCeJAm6olwnKqhUABO2iD-5MgM17LRLZUT0Bd_w?e=tjTbAj), [VSP-ML5](https://ulisboa-my.sharepoint.com/:u:/g/personal/ist425406_tecnico_ulisboa_pt/EQlgnN-U6_tAi2tkwAuawsIBjP5jkMrQ1ll-YXBgC91pSw?e=UXquWB);
  * [NSP-PA](https://ulisboa-my.sharepoint.com/:u:/g/personal/ist425406_tecnico_ulisboa_pt/EfWac59rENdCg8-vEAiEw3oBtP1bd8WkKgkDG1Xn3KoKZQ?e=1xuHNG), [NSP-ML75](https://ulisboa-my.sharepoint.com/:u:/g/personal/ist425406_tecnico_ulisboa_pt/EVJt8rEcBihEsAgoNXDjp9kBGNG1vcRy6MXGEiXxbG_T3g?e=JjvUwD), [NSP-Sia](https://ulisboa-my.sharepoint.com/:u:/g/personal/ist425406_tecnico_ulisboa_pt/Ed1qVQWdazpBrjXaDAx4BIUBId39enX1mP-yytWndOWVJw?e=vSGKzX);
  * [ENG-ML10](https://ulisboa-my.sharepoint.com/:u:/g/personal/ist425406_tecnico_ulisboa_pt/EQx0PUFia3JFvwt-JB85fvQBahmBJyNKG3QYNPwu6_ILVw?e=QOOTLw), [ENG-ML20](https://ulisboa-my.sharepoint.com/:u:/g/personal/ist425406_tecnico_ulisboa_pt/ERnAmgrswh9EgVGxVqoOHzEBWnIYUgiB5Nd8uyBeM8IpdQ?e=gJamVa), [ENG-ML50](https://ulisboa-my.sharepoint.com/:u:/g/personal/ist425406_tecnico_ulisboa_pt/EfFH8viNge9FhJ-53WgDzbcBdLJ1okxWgZer_emB6QvUvw?e=DnQzfy).

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

* You can download the [preprocessed data](https://drive.google.com/file/d/1_kZU08Vo2-qSHUt_qm7QNcWcnn2ligC5/view?usp=sharing) used to train our best models. This data is already preprocessed with seperate columns for context and response and follows the DialEvalML format. To train the multilingual models, simply train with the subsets concatenated.

### 2. Training

* Adjust and run `train_xxx.sh`.

## Citation

If you use this work, please consider citing:

~~~
@inproceedings{mendoncaetal2023towards,
    title = "Towards Multilingual Automatic Open-Domain Dialogue Evaluation",
    author = "Mendonça, John and Lavie, Alon and Trancoso, Isabel",
    booktitle = "Proceedings of the 24th Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = sep,
    year = "2023",
    address = "Prague, Czechia",
    publisher = "Association for Computational Linguistics",
}
~~~

~~~
@inproceedings{mendoncaetal2023simple,
    author    = "Mendonça, John and Pereira, Patricia and Moniz, Helena and Carvalho, João Paulo and Lavie, Alon and Trancoso, Isabel",
    title     = "Simple LLM Prompting is State-of-the-Art for Robust and Multilingual Dialogue Evaluation",
    booktitle = "DSTC11: The Eleventh Dialog System Technology Challenge",
    series    = "24th Meeting of the Special Interest Group on Discourse and Dialogue (SIGDIAL)",
    year      = 2023,
    month     = "September",
    address   = "Prague, Czechia"
}
~~~



