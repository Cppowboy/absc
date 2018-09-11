# ABSC, a project for aspect-based sentiment classification

## Requirements

+ python 3.5
+ pytorch 0.4
+ tensorboardX
+ absl-py
+ nltk
+ tqdm

## Usage

``` shell
# prepro
python -m lstm.main --mode prepro
# train
python -m lstm.main --mode train
# test
python -m lstm.main --mode test
# You can set different parameters or use different models and datasets.
```

## Experiment Result

| Models | Restaurant_category | Restaurant | Laptop |
| ------ | :-----------: | -----:| -----: |
| lstm | - | - | - |
| atae_lstm | - | 77.86/65.59 | 68.34/62.64 |
| acsa_gcae | - | 78.12/65.59 | 70.85/64.66 |
| bilstm_att_g | - | 76.34/63.65 | 69.91/63.20 |
| acsa_gcae_g | - | - | - |
| ram | - | 78.66/66.66 | 73.82/68.80 |
| tnet | - | 78.93/63.65 | 72.57/65.13 |

## Direcotory

+ data: the semeval2014 dataset
+ lstm: lstm model
+ atae_lstm: Wang, Yequan, Minlie Huang, and Li Zhao. "Attention-based lstm for aspect-level sentiment classification." Proceedings of the 2016 conference on empirical methods in natural language processing. 2016.
+ bilstm_att_g: Liu, Jiangming, and Yue Zhang. "Attention modeling for targeted sentiment." Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers. Vol. 2. 2017.
+ acsa_gcae: Xue, Wei, and Tao Li. "Aspect Based Sentiment Analysis with Gated Convolutional Networks." arXiv preprint arXiv:1805.07043 (2018).
+ acsa_gcae: acsa_gcae + gates
+ tnet: Li, Xin, et al. "Transformation Networks for Target-Oriented Sentiment Classification." arXiv preprint arXiv:1805.01086 (2018).

## Note

Not every implementation works exactly the same as the original paper, if you find any problems in the implementation, please tell me(panyx93@163.com).
