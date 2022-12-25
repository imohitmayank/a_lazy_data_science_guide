
- Here are some questions and their answers to make you ready for your next interview. Best of luck :wave:

!!! Question ""
    === "Question"
        #### What is the different between Standardization and Normalization?

    === "Answer"

        Normalization is the process of scaling the data to a common scale. It is also known as Min-Max Scaling where the final range could be [0, 1] or [-1,1] or something else. $X_{new} = (X - X_{min})/(X_{max} - X_{min})$ Standardization is the process of scaling the data to have zero mean and unit variance. $X_{new} = (X - mean)/Std$

!!! Question ""
    === "Question"
        #### Is it possible that during ML training, both validation (or test) loss and accuracy, are increasing?

    === "Answer"

        Accuracy and loss are not necessarily exactly (inversely) correlated, as loss measures a difference between raw prediction (float) and class (0 or 1), while accuracy measures the difference between thresholded prediction (0 or 1) and class. So if raw predictions change, loss changes but accuracy is more "resilient" as predictions need to go over/under a threshold to actually change accuracy. [Soltius's answer on SE](https://stats.stackexchange.com/questions/282160/how-is-it-possible-that-validation-loss-is-increasing-while-validation-accuracy)


!!! Question ""
    === "Question"
        #### Is K-means clustering algorithm guaranteed to converge with unique result?

    === "Answer"

        K-means clustering algorithm is guaranteed to converge but the final result may vary based on the centroid initialisation. This is why it is suggested to try multiple initialization strategies and pick the one with best clustering. The convergence is guaranteed as the sum of squared distances between each point and its centroid strictly decreases over each iteration. Also the practical run time of k-means is basically linear. Refer [Visualizing K Means Clustering - Naftali Harris](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

!!! Question ""
    === "Question"
        #### In K-means clustering, is it possible that a centroid has no data points assigned to it?

    === "Answer"

        Yes it is possible, imagine a centroid placed in middle of ring of other centroids. Several implementations either removes that centroid or random;y replace it somewhere else in the data space. Refer [Visualizing K Means Clustering - Naftali Harris](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

!!! Question ""
    === "Question"
        #### What is entropy in information theory?

    === "Answer"

        Entropy is a measure of the amount of uncertainty or randomness in a system. It is often used in information theory and statistical mechanics to describe the unpredictability of a system or the amount of information required to describe it. It's formula is, $\mathrm {H} (X):=-\sum _{x\in {\mathcal {X}}}p(x)\log p(x)=\mathbb {E} [-\log p(X)]$

        Here is an [excellent video](https://www.youtube.com/watch?v=ErfnhcEV1O8) from Aurelien Geron, explaining the topic.

