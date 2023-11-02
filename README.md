# Link Prediction: Reddit posts Suggestion
## Dataset
The Reddit dataset is a Public graph dataset from Reddit posts made in the month of September, 2014. The node label in this case is the community, or “subreddit”, that a post belongs to. 50 large communities have been sampled to build a post-to-post graph, connecting posts if the same user comments on both. In total this dataset contains 232,965 posts with an average degree of 492. The first 20 days are used for training and the remaining days for testing (with 30% used for validation). For features, off-the-shelf 300-dimensional GloVe CommonCrawl word vectors are used.
## The Downstream Task: Link prediction
Train a Graph Neural Network model to predict whether there are links between users and subreddit

## Details about the project
- Reimplement the MLP-mixer - based Graph model called [GraphMixer](https://arxiv.org/abs/2302.11636)
- Implement the Downstream task - Link prediction on the Graph model
- Process data and feed data to model for training, validating, and testing
- Using DVC tool to build a simple pipelines for reproducibility and version dataset
- Build a CI/CD workflow with Github Actions
## Train the model
Train the model by using this command:

`
python train_link_prediction.py dataset_name reddit --model_name GraphMixer
`

Or use the reproduce the DVC pipeline that is set up in dvc.yaml

`
dvc repro
`
## References:
- [GraphMixer Model](https://arxiv.org/abs/2302.11636)
- [CI/CD with DVC, CML and Github Action](https://iterative.ai/blog/CML-runners-saving-models-2)
- [DVC tool](https://dvc.org/doc)
- Way to process graph dataset for feeding forward to model: https://github.com/twitter-research/tgn
