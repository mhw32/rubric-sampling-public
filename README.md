# rubric-sampling

PyTorch implementation of the paper [Zero Shot Learning for Code Education: Rubric Sampling with Deep Learning Inference](https://arxiv.org/abs/1809.01357) by Mike Wu, Milan Mosse, Noah Goodman, and Chris Piech (AAAI 2019).

## About
Massive open online courses (MOOCs) log thousands of hours of data about how students solve coding challenges. Being so rich in data, these platforms have garnered the interest of the machine learning community, where the challenge is to autonomously provide feedback to help students learn. **But annotating programs is expensive, especially since the space of possible programs is Zipfean.** This repository implements *rubric sampling*, a zero shot solution to scalable feedback prediction. Rubric sampling asks the human-in-the-loop to describe a generative process of student thinking in the form of a probabilistic context-free grammar (PCFG), which allows for infinite data generation and takes exponentially less time than labeling even a tiny dataset. Rubric sampling can be combined with either supervised networks or modern deep generative models (MVAE). We apply these methods to a Code.org curriculum, containing 8 exercises with 1,598,375 student solutions. 

### Code.org Datasets
The dataset contains a curriculum of 8 exercises involving drawing a shape in 2-dimensional space given a discrete set of actions in a block-based environment. The student's goal is to utilize concepts of geometry and looping to complete each exercise, which increases in difficulty. The dataset is composed of unlabeled programs submitted by former users of Code.org. We painstakingly labeled a small set (~300 unique programs) for problem 1 and 8,  mostly meant for baselines and computing performance. 

### Expert Rubrics
Rubrics are designed in our meta-language, which can be found in `/rubrics`. Problem 1 contains two rubrics, one made by a University professor and one made by an undergraduate teaching assistant. Problems 2 through 8 contain only one rubric made by the TA.

## Usage
TODO.

### Requirements
We use PyTorch 0.4.1 and any compatible versions of NumPy, Pandas, Scikit-Learn. We also require NLTK and tqdm.  

## Citation

If you find this useful for your research, please cite:

```
@article{wu2018zero,
  title={Zero Shot Learning for Code Education: Rubric Sampling with Deep Learning Inference},
  author={Wu, Mike and Mosse, Milan and Goodman, Noah and Piech, Chris},
  journal={arXiv preprint arXiv:1809.01357},
  year={2018}
}
```
