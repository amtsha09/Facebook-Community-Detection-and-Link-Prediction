# Facebook-Community-Detection-and-Link-Prediction

## We implement community detection and link prediction algorithms using Facebook's "like" data.

- The file `edges.txt.gz` indicates like relationships between facebook users. 
- This was collected using snowball sampling: beginning with the user "Bill Gates", I crawled all the people he "likes", then, for each newly discovered user, I crawled all the people they liked.

## The Implementation is done in two phases:

### 1. Community Detection:

After the data collection process is completed we make a graph network of nodes and edges. Now to find communities in this graph we use the Girvan Newman Approach. (All the algorithms are developed from scratch without using any predefined libraries so as to understand the working of each part)

### 2. Link Prediction:

Now for the purpose of link prediction we now already have a Graph of Bill Gates "like" data. We remove 5 of the accounts liked by Bill Gates. Then we use this newly created graph and do Link Prediction, finally compute the accuracy of our prediction.


### Instructions to copy the environment to run the program without any dependency issue.

I have made a conda environment for this program and have saved that in "fbcommunitydetection.yml"

For performing and running all these files it is a better choice to activate this environment in which these have been written and executed.

Follow these steps to activate this environment:
	"conda env create -f fbcommunitydetection.yml"
	"source activate fbcommunitydetection"

Now you are in fbcommunitydetection environment and can easily run all the files in this repository

