# Clustering Algorithms Implementation

# Datasets

All datasets and results can be downloaded [here](https://www.dropbox.com/sh/fzbcjf7sb4zesjx/AABApeN8sryYH1eRtkBIUjZMa?dl=0). Please put all datasets to `./datasets/`. The output will be save in `./outputs/`. The name of datasets are `data_<dataset-name>.mat` and the name of the outputs are `output_<dataset-name>.mat`

# Compile and Run

USPEC.py is also the only source file. We can run the code by the following commands, 

```zsh
python3 USPEC.py <dataset-name>
```

For example,

```zsh
python3 USPEC.py Letters
```

It will generate an output file `output_Letters.mat` in `./outputs/`.

The code is tested on a Linux system.

# Reference

1. Wu, Lingfei, Pin-Yu Chen, Ian En-Hsu Yen, Fangli Xu, Yinglong Xia, and Charu Aggarwal. "Scalable spectral clustering using random binning features." In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 2506-2515. 2018. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3219819.3220090)]

2. Huang, Dong, Chang-Dong Wang, Jian-Sheng Wu, Jian-Huang Lai, and Chee-Keong Kwoh. "Ultra-scalable spectral clustering and ensemble clustering." IEEE Transactions on Knowledge and Data Engineering 32, no. 6 (2019): 1212-1226. [[PDF](https://ieeexplore.ieee.org/iel7/69/4358933/08661522.pdf)]
