# Indonesian small BigBird model

**Disclaimer:** This is work in progress. Additional information about the source code will be added in the future.

This GitHub repository contain source code to train/finetune the model, while HuggingFace hub contain model checkpoint and some technical explanation. Link to HuggingFace hub,

* https://huggingface.co/ilos-vigil/bigbird-small-indonesian
* https://huggingface.co/ilos-vigil/bigbird-small-indonesian-nli

## Environment

Software:
* Debian-based OS
* Miniconda
* Python 3.10.6
* Nvidia CUDA 11.2
* Nvidia cuDNN 11.2
* Nvidia NCCL 11.2

Hardware:
* 6C/12T CPU
* 32GB RAM
* RTX 3060 12GB
* SSD storage

Run this command to create Anaconda/Miniconda environment.

```
conda env create -f environment.yaml
```

## Download dataset

Since some dataset require additional step, seperate script which only used to download dataset is created. (1) Downloading Wikipedia dataset for first time will run Apache Beam to parse Wikipedia Dump file. ~14GB RAM used to parse Indonesian Wikipedia dump file from 2022-10-20. (2) Liputan6 news need to be downloaded seperately. Based on https://github.com/fajri91/sum_liputan6/, you must either fill a form or run their code manually to obtain the dataset. Below snippet contain Linux command i used to obtain the dataset.

```sh
git clone https://github.com/fajri91/sum_liputan6/
python3.7 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python 0_download.py
python 1_preprocessing.py
python 2_create_extractive_label.py
mv data/clean data/canonical
```

If you receive lots of error message about processing news, try to reduce value of `THREAD` variable on `0_download.py` file. Take these news have been removed from Liputan6. If you try to open those URL from browser, you'll redirected to their homepage.

```
Failed to proceed  https://www.liputan6.com/news/read/81044/peringatan-hari-antimadat-di-bandung-dihadiri-pecandu . Potentially the news has been deleted from Liputan6.
Failed to proceed  https://www.liputan6.com/news/read/130420/jamiyatul-islamiyah-sesat- . Potentially the news has been deleted from Liputan6.
```

Dataset to perform benchmark/test on NLI models should be downloaded from [meisaputri21/Indonesian-Twitter-Emotion-Dataset/](https://github.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset/) and put into directory `finetune/nli`.

## External references

* https://arxiv.org/abs/2007.14062
* https://arxiv.org/abs/2011.04006
* https://huggingface.co/blog/pretraining-bert
* https://huggingface.co/blog/warm-starting-encoder-decoder
* https://huggingface.co/blog/how-to-generate
* https://discuss.huggingface.co/t/training-sentencepiece-from-scratch/3477/2
* HuggingFace `transformers`, `datasets`, `evaluate` and `hub` documentation.
