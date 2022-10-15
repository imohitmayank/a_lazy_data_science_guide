!!! warning
    This page is still ongoing modifications. Please check back after some time or [contact me](mailto:mohitmayank1@gmail.com) if it has been a while! Sorry for the inconvinence :pray:
    
## Introduction

- Clustering is an [unsupervised task](introduction.md#unsupervised-learning) of grouping a set of items based on item’s features where the final grouping should minimize certain cost function. While finding optimum grouping is very hard, there are several algorithms that can help us find sub-optimal solutions. Also note that as the data is not labeled, the grouping could be completely different from the user’s expectation. Each clustering algorithm has its own internal similarity function and grouping strategy by which the clusters are formed.
- Clustering algorithms can be categorised based on different perspectives. Below are some examples *(not a complete list)*, 
  - Considering time of application, we can have online *(streaming data)* vs offline *(all data present)* clustering algorithms. 
  - Considering input data type, we have algorithms that consider item's features while others that considers item-item similarity matrix. 
  - Considering input parameter, there are algorithms that require no of clusters as input while others do not.

## Metrics

- Before starting with Clustering algorithms, let's understand the metrics that can be used to compare the clustering performance. 

### Silhouette Score

- Silhouette score considers the intra-cluster $a(i)$ and inter-cluster $b(i)$ distances to generate a performance metric for the clustering of a dataset. The score can range between -1 *(bad clustering)* and 1 *(good clustering)*, with higher number denoting better clustering. We can choose different distance functions *(euclidean, manhattan, cosine, etc)* based on the data. The formulation wrt each data point is shown below, where we can average the value to get dataset level scores.

$$
{\displaystyle a(i)={\frac {1}{|C_{I}|-1}}\sum _{j\in C_{I},i\neq j}d(i,j)}
$$

$$
{\displaystyle b(i)=\min _{J\neq I}{\frac {1}{|C_{J}|}}\sum _{j\in C_{J}}d(i,j)}
$$

$$
{\displaystyle s(i)={\frac {b(i)-a(i)}{\max\{a(i),b(i)\}}}}
$$

- In $a(i)$ formulation, we compare each data point against all the other points in the same cluster. We use $\frac {1}{|C_{I}|-1}$ as we ignore the $d(i, i)$ computation. In $b(i)$ formulation, we compare one data point against all data points from other clusters one at a time, and pick the value for the cluster with members having minimum average distance from the data point *(think of it like the next best cluster for the data point)*. Finally, the silhouette score for that data point is computed in $s(i)$. It is further simplified below, 

$$
s(i) = \begin{cases}
  1-a(i)/b(i), & \mbox{if } a(i) < b(i) \\
  0,  & \mbox{if } a(i) = b(i) \\
  b(i)/a(i)-1, & \mbox{if } a(i) > b(i) \\
\end{cases}
$$

- Sklearn package provides a function to compute silhouette score. The inputs are data point features, the cluster labels and the distance metric. Example call is shown below, 

```python linenums="1"
# import 
from sklearn.metrics import silhouette_score

# compute the score
score = silhouette_score(X, labels, metric='cosine')
```


## Clustering Algorithms

### K-Means

- K-means is the swiss army knife of the clustering algorithms, the forever baseline - the first clustering algorithm anyone tries :smile:. It can be easily understood by considering the steps mentioned below. The step (a) is a one time activity done during the initialization part, while steps (b) and (c) are repeated until the convergence i.e. there is no more change in cluster membership even if we continue the process or there is no more *noticable* centroid movement. 
  1. **Centroid Assignment:** Assign K centroids. There are three points to remember here, 
     1. How to decide the value of K? --> here it is an input parameter. In practice we can analyze the cluster results with different k *(2 to N)* and pick the one with best metric score like silhouette score.
     2. Are centroids choosen from data points? --> during initialization they may be selected from data points but over iterations they become their own special points that are part of the same feature space
     3. How are the centroids choosen? --> the assignment strategy can be random *(pick any random K data points)*, or follow *'furthest'* heuristic *($i^{th}$ centroid is choosen such that its minimum distance to the preceding centroids is largest)* or follow *k-mean++* heuristic *(selects a point with probability proportional to the square of its distance to the nearest preceding centroid)*.

    !!! Note
        Random initialization is not preferred, as it is possible to get all centroid assigned to data points from only one true cluster. The *'furthest'* heuristic is also not preferred as it select data points at the edges for centroid. K-means++ heuristic is more suitable as the centroid selection is more natural.

  2. **Cluster Assignment:** assign all data points to one of the centroids *(forming a cluster)* based on the closeness of the points to the centroid. The closeness is computed by a similarity function that could be equilidean distance. 
  3. **Centroid Adjustment:** adjust the centroids such that it minimises the intra-cluster distance among the cluster member. This is called inertia and its formulation is $\sum_{i=0}^n \text{min}(||x_i-\mu_{j}||^2)$, where $\mu_{j}$ is the mean of the respective cluster. The adjustment is done by moving the centroids to the mean position of the cluster members.
  
- Remember, K-means algorithm is guaranteed to converge but the final result may vary based on the centroid initialisation. This is why it is suggested to try multiple runs with different initialization and pick the one with best clustering or use smarter initialization technique like k-means++. Refer [ML Interview Questions](../machine_learning/interview_questions.md#is-k-means-clustering-algorithm-guaranteed-to-converge-with-unique-result) and [2] for more details.

- Many clustering algorithms have improved k-means over time, they are:
  - **K-Medians:**  It is a variation of k-means clustering where instead of calculating the mean for each cluster to determine its centroid, we calculates the median. As it uses Manhattan distance *(L1-norm distance)*, the algorithm becomes more reliable for discrete or binary data sets.
  - **K-Medoids:** In contrast to the k-means algorithm, k-medoids chooses actual data points as centers instead. Furthermore, k-medoids can be used with arbitrary dissimilarity measures, whereas k-means generally requires Euclidean distance. Because k-medoids minimizes a sum of pairwise dissimilarities instead of a sum of squared Euclidean distances, it is more robust to noise and outliers than k-means. Refer [3] for an intuitive solved example.
  - **K-means++:** It is the standard K-means algorithm but with a smarter initialization of the centroids. We start with choosing one center randomly from the data points. Then for each data point $x$ not chosen yet, we find the distance $D(x)$ between $x$ and the nearest center that has already been chosen. The new center is choosen again at random but this time using a weighted probability distribution where a point $x$ is chosen with probability proportional to $D(x)^2$. We repeat these steps until `k` centers have been chosen.
  - **Mini Batch K-means:** It is an optimized version of k-means for faster execution with only slighly worse results.  Here, at each iteration a set of random data points are selected to form a mini-batch and they are assigned to the nearest centroids. The centroid adjustment happens for each sample by taking streaming average of the sample and all previous samples assigned to that centroid. Mini Batch K-means converges faster than K-means.

!!! Hint
    K-Means works best in datasets with clusters that are roughly equally-sized and shaped roughly regularly. Also the data points should be in euclidean space as the K-means uses euclidean distance measure. It is not recommended for clustering neural network embeddings as they capture semantic meanings which is more suitably captured by cosine similarity. The best that can be done with K-means is to run multiple iterations on embeddings and pick the one with highest cosine silhouette score.

- Here is the code to perform Kmeans clustering and find the silhouette score [1],
  
```python linenums="1"
# import
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# load the data
X = np.load('data/embeddings.npy') # dummy

# create the model
model = KMeans(n_clusters = 2, init='k-means++', max_iter=100, n_init=1)
# get the label
labels = model.fit_predict(X)
# compute silhouette_score
sil_score = silhouette_score(X, labels, metric='cosine')
```

### Hierarchical Clustering

- It is a clustering technique that seeks to build a hierarchy of clusters i.e. we start with the dataset divided into `N` clusters and then some clusters are either merged or split based on their similarity. There are two major strategies to perform hierarchical clustering, 
  - **Agglomerative:** Bottom-up approach where we start with each sample as a separate cluster and then clusters are merged.
  - **Divisive:** Top-down approach where we start with all samples part of a single cluster and then at each level we recursively split existing clusters.
- To understand the split or merge process, we need to understand the following, 
  - **Distance metric:** this is the function that gives the distance between two data points. We can choose from a number of functions like euclidean, cosine, manhattan, etc. 
  - **Linkage criteria:** this is a function that define the distance between two clusters based on applying distance function on pairs of data from the two clusters. There are following strategies to choose from, 
    - **Complete Linkage:** Pick the most distant pair of points as the representation of cluster distance. Formulation is :  $\max \,\{\,d(a,b):a\in A,\,b\in B\,\}.$
    - **Single Linkage:** Pick the least distant pair of points as the representation of cluster distance. Formulation is : $\min \,\{\,d(a,b):a\in A,\,b\in B\,\}.$
    - **Ward:** find the pair of clusters that leads to minimum increase in total within-cluster variance after merging. This is only applicable for euclidean distance. [4]
    - **Average:** Uses the average of distances between all pairs of data points from the two clusters. Formulation is ${\displaystyle {\frac {1}{|A|\cdot |B|}}\sum _{a\in A}\sum _{b\in B}d(a,b).}$
- Now it should be easy to understand the overall process. Taking agglomerative as example, to begin with, all samples are separate clusters. At each iteration, we will compute the linkage score between all pairs of clusters and find the pair with the minimum score. We merge that pair of clusters together and go ahead with next iteration. We keep on repeating this process until we have reached a desired number of clusters (`n_clusters`) or the linkage distance threshold (`distance_threshold`) has triggered, above which, clusters will not be merged.

!!! Note
    At a time, we can only use either `n_clusters` or `distance_threshold`.

- Here is a sample code using sklearn package [1], 

```python linenums="1"
# import
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# load the data
X = np.load('data/embeddings.npy') # dummy

# create the model
model = AgglomerativeClustering(n_clusters = 2, affinity='cosine', linkage='average')
# get the label
labels = model.fit_predict(X)
# compute silhouette_score
sil_score = silhouette_score(X, labels, metric='cosine')
```

<!-- !!! Hint
    To understand the fundamentals of Linear Algebra, I would recommend [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab), a video series by [3Blue1Brown](https://www.youtube.com/c/3blue1brown). Special chapter relevant for our use case is [EigenValue and EigenVectors](https://www.youtube.com/watch?v=PFDu9oVAE-g&t=6s)
     -->
     
## References

[1] [Sklearn - Clustering](https://scikit-learn.org/stable/modules/clustering.html)

[2] [Visualizing K Means Clustering - Naftali Harris](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

[3] [K-Medoids clustering with solved example](https://www.geeksforgeeks.org/ml-k-medoids-clustering-with-example/)

[4] Wikipedia - [Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) | [Ward's method](https://en.wikipedia.org/wiki/Ward%27s_method)