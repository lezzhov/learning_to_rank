# Learning To Rank using the MSLR-WEB10K dataset

In this project I evaluate a search academic dataset using common learn-to-rank features, build a ranking model using the dataset, and discuss how additional features could be used and how they would impact the performance of the model.

# The Dataset

The dataset consists of machine learning data, in which queries and urls are represented by IDs. The datasets consist of feature vectors extracted from query-url pairs along with relevance judgment labels. The relevance judgments are obtained from a retired labeling set of a commercial web search engine (Microsoft Bing), which take 5 values from 0 (irrelevant) to 4 (perfectly relevant).

In the data files, each row corresponds to a query-url pair. The first column is relevance label of the pair, the second column is query id, and the following columns are features. The larger value the relevance label has, the more relevant the query-url pair is. A query-url pair is represented by a 136-dimensional feature vector.

Below are two rows from MSLR-WEB10K dataset:

==============================================

0 qid:1 1:3 2:0 3:2 4:2 … 135:0 136:0

2 qid:1 1:3 2:3 3:0 4:0 … 135:0 136:0

==============================================

You can read about the features included in the dataset here: https://www.microsoft.com/en-us/research/project/mslr/

# Usage



# Model

# Results

# Future Development


Author: Vladimir Lezzhov

