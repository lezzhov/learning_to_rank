# Learning To Rank using the MSLR-WEB10K Dataset

In this project I evaluate a search academic dataset using common learn-to-rank features, build a ranking model using the dataset, and discuss how additional features could be used and how they would impact the performance of the model.

Author: Vladimir Lezzhov

# Dataset

The dataset consists of machine learning data, in which queries and urls are represented by IDs. The datasets consist of feature vectors extracted from query-url pairs along with relevance judgment labels. The relevance judgments are obtained from a retired labeling set of a commercial web search engine (Microsoft Bing), which take 5 values from 0 (irrelevant) to 4 (perfectly relevant).

In the data files, each row corresponds to a query-url pair. The first column is relevance label of the pair, the second column is query id, and the following columns are features. The larger value the relevance label has, the more relevant the query-url pair is. A query-url pair is represented by a 136-dimensional feature vector.

Below are two rows from MSLR-WEB10K dataset:

==============================================

0 qid:1 1:3 2:0 3:2 4:2 … 135:0 136:0

2 qid:1 1:3 2:3 3:0 4:0 … 135:0 136:0

==============================================

You can read about the features included in the dataset here: https://www.microsoft.com/en-us/research/project/mslr/

# Usage

Setup the project:

```
python3 setup.py install
```

Tuning the model hyperparameters:
```
python3 tune.py ../data
```

Training:
```
python3 train.py --num_leaves --learning_rate --reg_lambda

or, to change hyperparameters from defaults:

python3 train.py --num_leaves 66 --learning_rate 0.244 --reg_lambda 2.376
```

Testing:
```
python3 test_model.py ../data model.txt
```

Deploying:
```
python3 deploy.py [] []
```

# Model
![alt text](https://lightgbm.readthedocs.io/en/latest/_images/leaf-wise.png)

# Results

I chose nDCG as the evaluation metric because it takes graded relevance values into account. 
It does a better job in evaluating the position of ranked items compared to MAP 
(which only considers binary relevance ranks, 0 or 1) or MRR 
(which only looks at the highest ranking document and does not consider all links).

nDCG employs a factor called smooth logarithmic discounting which improves performance.
Authors of the work at http://proceedings.mlr.press/v30/Wang13.pdf show that
for every pair of substantially different ranking recommender,
the NDCG metric is consistently able to determine the better one.

All nDCG calculations are relative values on the interval 0.0 to 1.0.
A perfect nDCG score would be 1.0. Therefore,
considering the score of 0.9257691834034475, I think the model performed reasonably well.

# Future Development

Here are some possible ideas on how to improve the model's performance:

1) Use the full available dataset instead of just 1 fold, possibly incorporating 5-fold cross-validation.
2) Try to tune a higher number of hyperparameters
3) Tune hyperparamters and train the model for a longer period of time
4) Evaluate whether the removal of the features the paper found to be less useful did indeed lead to better results 
(compare mean nDCG scores for both cases)
5) Consider overfitting and underfitting, try some of the strategies suggested here:
https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html,
 e.g. Use small max_bin, Use min_data_in_leaf and min_sum_hessian_in_leaf,Use bigger training data, etc
6) Consider outliers for some features and remove them if necessary
7) Consider if it is needed to engineer new features, or to take some out.

