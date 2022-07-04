
### Introduction

- Suppose you have a decision to make — like buying a house, or a car, or even a guitar. You don’t want to choose randomly or get biased by someone’s suggestion, but want to make an educated decision. For this, you gathered some information about the entity you want to buy (let’s say it’s a car). 
- So you have a list of `N` cars with their price information. As usual, we won’t want to spend more, we can just sort the cars by their price (in ascending order) and pick the top one (with the smallest price), and we are done! This was decision making with a single criterion. 
- But alas if life is so easy :smile: We would also like the car to have good mileage, better engine, faster acceleration (if you want to race), and some more. Here, you want to choose a car with the smallest price, but the highest mileage and acceleration, and so on. This problem can’t be so easily solved by simple sorting. Enter multi-criteria decision-making algorithms! :sunglasses:

### Dataset

- Let’s choose one dataset so it becomes easier to visualize the result, to understand what’s really happening behind the scenes and finally build intuition. 
- For this, I am picking cars dataset. For each car, we will focus on a subset of attributes and only pick 10 rows (unique cars) to make our life easier. Look at the selected data,

<figure markdown> 
    ![](../imgs/ra_dataset.png){ width="500" }
    <figcaption>10 rows from the cars dataset</figcaption>
</figure>

- Explaining some attributes,
  - `mpg`: a measure of how far a car can travel if you put just one gallon of petrol or diesel in its tank (mileage).
  - `displacement`: engine displacement is the measure of the cylinder volume swept by all of the pistons of a piston engine. More displacement means more power.
  - `acceleration`: a measure of how long it takes the car to reach a speed from 0. Higher the acceleration, better the car for drag racing :)

- Here please notice some points,
  - The unit and distribution of the attributes are not the same. Price plays in thousands of $, acceleration in tens of seconds and so on.

    <figure markdown> 
        ![](../imgs/ra_dataset2.png){ width="500" }
        <figcaption>describing each of the numerical columns (the attributes) of the selected data</figcaption>
    </figure>

  - The logic of best for each attribute vary as well. Here, we want to find a car with high values in mpg, displacement and acceleration. At the same time, low values in weight and price. This notion of high and low can be inferred as maximizing and minimizing the attributes, respectively.

  - There could be an additional requirement where we don’t consider each attribute equal. For example, If I want a car for racing and say I am sponsored by a billionaire, then I won’t care about mpg and price so much. I want the faster and lightest car possible. But what if I am a student (hence most probably on a strict budget) and travel a lot, then suddenly mpg and price become the most important attribute and I don’t give a damn about displacement. These notions of important of attributes can be inferred as weights assigned to each attribute. Say, price is 30% important, while displacement is only 10% and so on.

- With the requirements clear, let’s try to see how we can solve these kinds of problems.

### Generic methodology

- Most of the basic multi-criteria decision solvers have a common methodology which tries to,

  - Consider one attribute at a time and try to maximize or minimize it (as per the requirement) to generate optimized score.
  - Introduce weights to each attributes to get optimized weighted scores.
  - Combine the weighted scores (of each attribute) to create a final score for an entity (here car).

- After this, we have transformed the requirements into a single numerical attribute (final score), and as done previously we can sort on this to get the best car (this time we sort by descending as we want to pick one with maximum score). Let’s explore each step with examples.

#### Maximize and Minimize

- Remember the first point from the dataset section, attributes have very different units and distributions, which we need to handle. One possible solution is to normalize each attribute between the same range. And we also want the direction of goodness to be similar (irrespective of the logic). Hence after normalization, values near maximum of range (say 1) should mean that car is good in that attribute and lower values (say near 0) means they are bad. We do this with the following formula,

<figure markdown> 
    ![](../imgs/ra_minimax.png){ width="500" }
    <figcaption>normalization logic for maximizing and minimizing an attribute values</figcaption>
</figure>

- Look at the first equation for maximizing, one example is update mpg of each car by dividing it by sum of mpg of all cars (sum normalization). We can modify the logic by just considering the max of mpg or other formulae itself. The intention is, after applying this to each attribute, the range of each attribute will be the same as well as we can infer that value close to 1 means good.

- The formulae for minimizing is nearly the same as the maximizing one, we just inverse it (1 divided by maximize) or mirror it (by subtracting it from 1) to actually reverse the goodness direction (otherwise 1 will mean bad and 0 will mean good). Let’s see how it looks after in practice,


<figure markdown> 
    ![](../imgs/ra_sumnorm.png)
    <figcaption>Example for sum normalization heatmap of the original data. Check the ‘mpg’ value of ‘ford torino’. Originally its 17 but after sum normalization, it should be 17/156=0.109. Similarly, the ‘price’ is 20k, after inverse it will be 1/(20k/287872) = 14.4</figcaption>
</figure>

todo

!!! Note
    **Update - March 2022:** Due to code breaking changes in the latest version of scikit-criteria, it is recommended to use v0.2.11 of the package for the code discussed in the article. Code repository is here.