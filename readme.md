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

| Model | Restaurant(aspect category) | Restaurant(aspect term) | Laptop(aspect term) |
| - | - | - |
| lstm | 82.73 | 75.00 | 66.93 |
| atae_lstm | 83.86 | 78.39 | 69.75 |
| acsa_gcae | 82.53 | 78.84 | 71.63 |
| bilstm_att_g | - | 80.39 | 73.09 |
| acsa_gcae_g | - | 79.85 | 73.58 |
| tnet | - | 79.85 | 76.54 |

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
