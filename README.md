## Clickbait? Sensational Headline Generation with Auto-tuned Reinforcement Learning
<img src="plot/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

<img align="right" src="plot/HKUST.jpg" width="12%">

This is the PyTorch implementation of the paper:

**Clickbait? Sensational Headline Generation with Auto-tuned Reinforcement Learning**. [**Peng Xu**](https://scholar.google.com/citations?user=PQ26NTIAAAAJ&hl=en), Chien-Sheng Wu, Andrea Madotto, Pascale Fung  ***EMNLP 2019*** [[PDF]](https://arxiv.org/abs/1909.03582)

This code has been written using python3 and PyTorch >= 0.4.0 and its built on top of https://github.com/atulkum/pointer_summarizer. If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex is listed below:
<pre>
@inproceedings{xu2019clickbait,
  title={Clickbait? Sensational Headline Generation with Auto-tuned Reinforcement Learning},
  author={Xu, Peng and Wu, Chien-Sheng and Madotto, Andrea and Fung, Pascale},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={3056--3066},
  year={2019}
}
</pre>

## Dependency
Check the packages needed or simply run the command
```console
❱❱❱ pip install -r requirements.txt
```

## Data Preprocessing
`data_preprocess.ipynb` contains necessary preprocessing steps. Note that same_*.txt are the one with score(produced by CNN scorer) and should be placed in `\sensational_headline\dataset\sensation_lcsts`, and segment_*.txt are the one without score, which should be placed in `\sensational_headline\dataset\lcsts`.

## Resources
Fine-tuned model on Yusan data can be downloaded at [**here**](https://drive.google.com/file/d/1m8gQt3G7rSrZeoMUPoTNt4sSVT5TpLu1/view?usp=sharing) and unzip to the sensational_headline\sensation_save\Rl directory.

(Optional) To train and run your model from scratch, you need [**datasets**](https://drive.google.com/open?id=1ufGjlp2yGQ7Z--scYVEkvlu3hm-ec3dD) and unzip to the project home directory

Pretrained Chinese embedding download: [**click me**](https://github.com/Embedding/Chinese-Word-Vectors) and unzip to the sensational_headline\ directory.

## Experiment

***Fine-tune***

Pointer-Gen+ARL-SEN
```console
❱❱❱ python fine_tune_sensation_generation.py -rl_model_path save/Rl/Pointer_Gen_ARL_SEN/ -sensation_scorer_path save/sensation/512_0.9579935073852539/ -thd 0.1 -use_rl True -use_s_score 1 -batch_size 4 -eval_step 300

```

***Generation***

Pointer-Gen+ARL-SEN
```console
❱❱❱ python sensation_save.py -rl_model_path sensation_save/Rl/50000_4_350_500_0.0_1_1_birnn_pointer_attn_0.0001_adam_1.0_False_False_False_rl_no_cov_0.0_True_0.1_0.6436150074005127  -sensation_scorer_path save/sensation/512_0.9579935073852539/ -use_s_score 0 -thd 0.0 -use_rl True

```
