# CQT (Convolved Quality Transformer)

PyTorch implementation of CQT which is the learning-based full-reference image quality evaluator.



## Abstract

A hybrid architecture composed of a convolutional neural network (CNN) and a Transformer is the new trend in realizing various vision tasks while pushing the limits of learning representation. From the perspective of mechanisms of CNN and Transformer, a functional combination of them is suitable for the image quality assessment (IQA) since which requires considering both local distortion perception and global quality understanding, however, there has been scarce study employing such an approach.
This paper presents an end-to-end CNN-Transformer hybrid model for full-reference IQA named convolved quality transformer (CQT). CQT is inspired by the human's perceptual characteristics and is designed to unify the advantages of both CNN and Transformer. In QCT, convolutional layers specialize in local distortion feature extraction whereas Transformer aggregates them to estimate holistic quality via long-range interaction. Such a series of processes is repeated on a multi-scale to efficiently capture quality representation. To verify submodules in CQT performing their roles properly, we in-depth analyze the interaction between local distortions inferring global quality with attention visualization.
Finally, the perceptually pooled information from stage-wise feature embeddings derives the final quality level. The experimental results demonstrate that the proposed model achieves superior performance in comparison to previous data-driven approaches, and which is even well-generalized over standard datasets.

![cqt_architecture](https://github.com/ohheeseok/cqt_main/blob/main/images/cqt_architecture.png)


## Dependencies

- This project is constructed by using a template project, please check all dependencies [here](https://github.com/victoresque/pytorch-template).



## Download the KADID-10K dataset

- To train CQT, we utilized [KADID-10K dataset](http://database.mmsp-kn.de/kadid-10k-database.html). The downloaded data can be unzipped to any folder, but the dataset path has to be known by `./config/config.json`  file.



## Train model

- Almost all hyperparameters and settings can be controlled in `./config/config.json`. Please find it.
- If you would use your own setting, a custom config file (.json) can be employed.

```re
# train on GPU
python train.py

# train with custom config
python train.py -c <config_path>
```



Our code is released under MIT License.
