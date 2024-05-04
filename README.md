# WaveMask & WaveMix

Explanation will be given soon.

## Dataset

You can obtain all datasets under https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy. All of them are ready for training. 

```
  mkdir dataset
```
Please place all of them within the ```./dataset ``` directory.

## Quick Start

Clone the project

```bash
  git clone https://github.com/jafarbakhshaliyev/Wave-Augs.git
```

Go to the project directory

```bash
  cd Wave-Augs
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Train: 

```bash
  sh scripts/etth1.sh
  sh scripts/etth2.sh
  sh scripts/weather.sh
  sh scripts/ili.sh
```

You can change ```percentage``` to down-sample training dataset for ablation study.

## Citation

If you find the code useful, please cite our paper:

```
```

Please remember to cite all the datasets and compared methods if you use them in your experiments.

## Acknowledgements

We would like to express our gratitude to [Zheng et al.](https://arxiv.org/abs/2205.13504) for providing datasets used in this project. Additionally, we also acknowledge [Chen et al.](https://arxiv.org/abs/2302.09292) and [Zhang et al.](https://arxiv.org/abs/2303.14254)  for their code frameworks, which served as the foundation for our codebase.