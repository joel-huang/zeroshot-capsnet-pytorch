# GPU-accelerated PyTorch implementation of Zero-shot User Intent Detection via Capsule Neural Networks 

This repository implements a capsule model IntentCapsNet-ZSL on the SNIPS-NLU dataset in Python 3
with PyTorch, first introduced in the paper _Zero-shot User Intent Detection via Capsule Neural Networks_.

The code aims to follow PyTorch best practices, using `torch` instead of `numpy` where possible, and using
`.cuda()` for GPU computation. Feel free to contribute via pull requests.

# Requirements

python 3.6+

torch 1.0.1

numpy

gensim

scikit-learn

# Usage and Modification

* To run the training-validation loop: `python run.py`.
* The custom `Dataset` class is implemented in `dataset.py`.

# Acknowledgements
* Original repository (TensorFlow, Python 2): https://github.com/congyingxia/ZeroShotCapsule 
* Re-implementation (PyTorch, Python 2): https://github.com/nhhoang96/ZeroShotCapsule-PyTorch-

Please see the following paper for the details:

Congying Xia, Chenwei Zhang, Xiaohui Yan, Yi Chang, Philip S. Yu. Zero-shot User
Intent Detection via Capsule Neural Networks. In Proceedings of the 2018 Conference on
Empirical Methods in Natural Language Processing (EMNLP), 2018.

https://arxiv.org/abs/1809.00385 


```
@article{xia2018zero,
  title={Zero-shot User Intent Detection via Capsule Neural Networks},
  author={Xia, Congying and Zhang, Chenwei and Yan, Xiaohui and Chang, Yi and Yu, Philip S},
  journal={arXiv preprint arXiv:1809.00385},  
  year={2018}
}
```
# References
* https://github.com/soskek/dynamic_routing_between_capsules
* https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding 

