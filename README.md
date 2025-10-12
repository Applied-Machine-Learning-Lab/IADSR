# IADSR
Code for 《Empowering Denoising Sequential Recommendation with Large Language Model Embeddings》 accepted to CIKM 2025 Full Research track. 

Usage
Please follow the steps below to reproduce the experimental results from our paper.

1. Environment

We use the environment with torch 2.5.1, pandas 2.2.3, and numpy 1.23.1. We install [LLM2Vec](https://github.com/McGill-NLP/llm2vec/tree/main) and other dependencies by running the following commands:

```
cd IADSR
pip install info-nce-pytorch
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```
This will set up the environment for rapid reproduction of the results.

2. Data Preparation
Download Data: First, please download the raw datasets from Amazon Review Data (2014)(https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews). Place the downloaded files into a data/ directory. The primary datasets used in our work are:

Beauty

Sports and Outdoors

Toys and Games

Movielens-100k

Preprocessing: Run the Jupyter Notebook read_data.ipynb to clean and preprocess the raw data. This will generate the user-item interaction sequences required for the models.

3. Generate Semantic Embeddings
After preprocessing, run the tointerest.py script to generate semantic embeddings for each item in the dataset using the Large Language Model.

# Example for the Beauty dataset:
python tointerest.py --dataset Beauty

# Example for the Sports and Outdoors dataset:
python tointerest.py --dataset Sports_and_Outdoors

To run the baseline models:
python Gru4Rec.py

Then you can run the GRU_evaluation.py to get the evaluation!

If you find our work helpful to you, please consider citing our paper

@inproceedings{wu2025iadsr,
  author       = {Wu, Tongzhou and Wang, Yuhao and Wang, Maolin and Zhang, Chi and Zhao, Xiangyu},
  title        = {Empowering Denoising Sequential Recommendation with Large Language Model Embeddings},
  booktitle    = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)},
  pages        = {2844--2848},
  year         = {2025},
  publisher    = {ACM},
  url          = {https://github.com/Applied-Machine-Learning-Lab/IADSR}
}
