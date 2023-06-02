# Envs
``conda env create -f requirements.txt``

# Run
``python run_DMGI_full_embed.py``
``python run_DMGI.py``

# Implementation
PyG is a popular geometric learning framework and DMGI is not implemented by it.

# Model
``DMGI`` is my implementation the original model from paper: Unsupervised Attributed Multiplex Network Embedding. 
The out put of the model is only 1 type of nodes of all nodes. For example, IMDB has three types of nodes: actor, director and movie. DMGI can only output 1 type of node embedding such as actor, as all metapath starts from actor and ends at actor.

``DMGI_FULL_EMBED`` is my implementation of DMGI. It outputs all embeddings of all types of nodes with any metapaths, which it convenient for users to conduct downstream tasks.