## **Lightweight vector searcher**

Superlight weight vector searcher, implemented using KD Trees. data is saved to json and the tree stucture is saved to the .sav structure. Examples of how to add and load data are in the notebooks. It's good for tiny applications. Uses embeddings from Sentence embedding package.

add_one()

takes the form of

query: Query to pass in as example

embedding_query: Query with mask [MASK] to create embedding out of

data: The sequel

Generates the three additional columns:

embedding: Creates embedding from BERT from the embedding_query

index: (automatically +1 last element)

hash: hash of the embedding_query

KDTree_ask.sav is for ask table indexs/KDTree

KDTree_predict.sav and KDTree_visualise.sav ^ infer
