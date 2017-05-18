# Facebook-Community-Detection-and-Link-Prediction

## We implement community detection and link prediction algorithms using Facebook's "like" data.

- The file `edges.txt.gz` indicates like relationships between facebook users. 
- This was collected using snowball sampling: beginning with the user "Bill Gates", I crawled all the people he "likes", then, for each newly discovered user, I crawled all the people they liked.

## We do the following tasks:

### Community Detection: 
- We make clusters or communities by implementing Girvan-Newman algorithms.

### Link Prediction:
- We remove 5 of the accounts that Bill Gates likes and compute our accuracy at recovering those links.


### Instructions to copy the environment to run the program without any dependency issue.

I have made a conda environment for this program and have saved that in "fbcommunitydetection.yml"

For performing and running all these files it is a better choice to activate this environment in which these have been written and executed.

Follow these steps to activate this environment:
	"conda env create -f fbcommunitydetection.yml"
	"source activate fbcommunitydetection"

Now you are in fbcommunitydetection environment and can easily run all the files in this repository

