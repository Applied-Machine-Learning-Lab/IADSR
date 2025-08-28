# IADSR
Code for 《Empowering Denoising Sequential Recommendation with Large Language Model Embeddings》 accepted to CIKM 2025 Full Research track. 

Usage
Please follow the steps below to reproduce the experimental results from our paper.

1. Data Preparation
Download Data: First, please download the raw datasets from Amazon Review Data (2014). Place the downloaded files into a data/ directory. The primary datasets used in our work are:

Beauty

Sports and Outdoors

Toys and Games

Movielens-100k

Preprocessing: Run the Jupyter Notebook read_data.ipynb to clean and preprocess the raw data. This will generate the user-item interaction sequences required for the models.

2. Generate Semantic Embeddings
After preprocessing, run the tointerest.py script to generate semantic embeddings for each item in the dataset using the Large Language Model.

# Example for the Beauty dataset:
python tointerest.py --dataset Beauty

# Example for the Sports and Outdoors dataset:
python tointerest.py --dataset Sports_and_Outdoors

To run the baseline models:
python Gru4Rec.py



If you find our work helpful to you, please consider citing our paper

@inproceedings{anonymous2025iadsr,
  author       = {Anonymous Author(s)},
  title        = {Empowering Denoising Sequential Recommendation with Large Language Model Embeddings},
  booktitle    = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)},
  year         = {2025},
  publisher    = {ACM}
}
