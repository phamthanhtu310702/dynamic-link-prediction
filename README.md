# Graphmixer: Reddit posts classification
## Dataset
The Reddit dataset is a graph dataset from Reddit posts made in the month of September, 2014. The node label in this case is the community, or “subreddit”, that a post belongs to. 50 large communities have been sampled to build a post-to-post graph, connecting posts if the same user comments on both. In total this dataset contains 232,965 posts with an average degree of 492. The first 20 days are used for training and the remaining days for testing (with 30% used for validation). For features, off-the-shelf 300-dimensional GloVe CommonCrawl word vectors are used.
```
cd preprocess_data/
python preprocess_data.py  --dataset_name reddit
```