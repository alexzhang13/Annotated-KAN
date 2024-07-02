# Annotated KAN
*Note: this is an unofficial resource for the [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)* paper that walks through the code.
Relevant code and notebooks to accompany the Annotated KAN blog post found here [https://alexzhang13.github.io/blog/2024/annotated-kan/](https://alexzhang13.github.io/blog/2024/annotated-kan/).


## Setup
I wanted to keep the code simple, so you only need to install the following libraries (don't worry that much about the versions except for major library versions).
```
pip install torch==2.3.1
pip install numpy==1.26.4
pip install matplotlib==3.9.0
pip install tqdm==4.66.4
pip install torchvision==0.18.1
```

The final example includes a bit about MNIST, although it's not that important. If you want to run the code, download the data following [this thread](https://github.com/pytorch/vision/issues/1938):
```
wget www.di.ens.fr/~lelarge/MNIST.tar.gz
tar -zxvf MNIST.tar.gz -C data/
```

## Notebooks
The notebooks serve as a companion source to the annotated blog if you don't want to write out the code yourself. They are taken verbatim from the
code in the blog post, and do not include any extra information.

`notebooks/MinimalKAN.ipynb` is a barebones implementation of the KAN that follows Parts I and II of the blog.

`notebooks/AnnotatedKAN.ipynb` follows the whole blog post, and also includes B-spline specific optimizations and pruning logic that was featured in the [original KAN paper](https://arxiv.org/abs/2404.19756).

## Code
The notebooks are identical to the code, which starts from the `main.py` file. Put simply,
`KAN.py` is a high-level KAN module that composes several layers defined in `KANLayer.py` and `bspline.py`.
`KANTrainer.py` and `datasets.py` includes simple code for generating synthetic data and training your models, and `main.py` runs a simple example.

For example, you can just run `python main.py`.
