- Quoting [Wikipedia](https://en.wikipedia.org/wiki/Network_science), *"Network science is an academic field which studies complex networks such as telecommunication networks, computer networks, biological networks, cognitive and semantic networks, and social networks, considering distinct elements or actors represented by nodes (or vertices) and the connections between the elements or actors as links (or edges)."*
- To better understand network science, we should understand the underlying building blocks i.e. Graphs!

### Graphs 101

- Graph or Networks is used to represent relational data, where the main entities are called nodes. A relationship between nodes is represented by edges. A graph can be made complex by adding multiple types of nodes, edges, direction to edges, or even weights to edges. One example of a graph is shown below. 

<figure markdown> 
        ![](../imgs/ns_intro_graph101_karate.png)
        <figcaption>Karate dataset visualization @ [Network repository](http://networkrepository.com/graphvis.php)</figcaption>
        </figure>

- The graph above is the Karate dataset [1] which represents the social information of the members of a university karate club. Each node represents a member of the club, and each edge represents a tie between two members of the club. The left info bar states several graph properties like a number of nodes, edges, density, degree, etc. [Network repository](http://networkrepository.com/graphvis.php) contains many such networks from different fields and domains and provides visualization tools and basic stats as shown above.


### Common Concepts in Graphs

- **Nodes:** They are the basic building blocks in the graph that represent certain object, concepts or entities. Nodes *(or vertices)* can be of one type or of multiple types if the graph is homogenous or heterogenous, respecitively. 
- **Edges:** They are the ties that connect two nodes together based on some special relationship criteria. For example, for a graph where each nodes are intersection, an edge between two nodes *(or intersection)* denotes precense of direct roadways between them. Edges can also be of multiple types if the graph is heterogenous. 
- **Directed or Undirected Graph:** Edges may have a direction like `A --> B` that denotes a directional relationship. Example is Mother-Son relationship, `Kavita -- mother-of --> Mohit`is true, and not the other way around. If all the edges of a graph are directional, then it is a directed graph. Similarly, if the edges of a graph have no direction like `A -- B`, it is undirected graph. Example is a neighbor relationship, me and the guy living next door to me, are both neighbors of each other. 
- **Degrees of Node:** Degrees denote the number of direct connections of a node to other nodes in the graph. For a directed graph, we have in-degree *(edges coming to the node)* and out-degree *(edges going out from the node)*. For example in `A --> B <-- C`, A and C has `out-degree = 1 & in-degree = 0` & B has `out-degree = 0 & in-degree = 2`
- **Path and Walk**: Path or Walk are the route between two nodes that goes through multiple nodes and edges. For example, `A --> B --> C` is a path of length 2 as there are two edges in the route. The difference between a path and a walk is that walks can repeat edges and nodes. Refer [Stack Exchange Answer](https://math.stackexchange.com/questions/1890620/finding-path-lengths-by-the-power-of-adjacency-matrix-of-an-undirected-graph).
- **Connected Graph:** A graph with a possible path between any two nodes is called a connected graph. On the contrary, the graph will be disconnected and there might be multiple clusters of individual connected sub-graphs in the overall graph. 
- **Clique and Complete graph:** A clique of a graph is a set of nodes where every pair of nodes has an edge between them. It is the strongest form of cluster in the graph. Similarly if the graph itself is a clique i.e. there is an edge between all the pairs of nodes, it is called a complete graph. This also means that the graph contains all the possible edges.
- **Spanning Tree:** A tree is an undirected graph where any two nodes are connected by exaclty one path. The spanning tree of a graph is a subgraph that is a tree that contains every node in the graph. In practice, Kruskal Algorithm can be used to find the minimum spanning tree for a graph, where we have multiple possibilities of creating spanning trees but want one with minimum total edge or node weights. 
- **Adjacency matrix:** It is a square matrix of size `NxN`, where `N` is the number of unique nodes. The matrix contains 1 and 0 value that denotes the presence (1) or absence (0) of an edge between the nodes. unique graph and it's adjacency matrix is shown below, where the column and row represent nodes in alphabetic order.
  ``` mermaid
  graph LR
     A --- B --- C --- D --- E --- A
     A --- A
     E --- B
     D --- F
  ```
  $M_{Adj} = \begin{bmatrix} 1 & 1 & 0 & 0 & 1 & 0 \\ 1 & 0 & 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 & 1 & 1 \\ 1 & 1 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ \end{bmatrix}$

  !!! Hint
      $N^{th}$ power of the Adjacency Matrix ($M$) is a new square matrix ($M^N$), where $M_{ij}^N$ represents the number of walks of length $N$ between the nodes $i$ and $j$.

- **Laplacian matrix:** given a graph with $v_i$ to $v_n$ nodes, Laplacian matrix $L_{nxn}$ is

  $${\displaystyle L_{i,j}:={\begin{cases}\deg(v_{i})&{\mbox{if}}\ i=j\\-1&{\mbox{if}}\ i\neq j\ {\mbox{and}}\ v_{i}{\mbox{ is adjacent to }}v_{j}\\0&{\mbox{otherwise}},\end{cases}}}$$

  Or it can also be computed as $L = D - A$, where $D$ is the degree matrix *(could be in-degree, out-degree or both)* and $A$ is the adjacency matrix. Laplacian matrix are important objects in the field of [Spectral Graph Theory](https://web.stanford.edu/class/cs168/l/l11.pdf), and we can infer many properties of the graph like connected components by looking at its laplacian matrix. Refer this [Maths Stack Exchange Question](https://math.stackexchange.com/questions/18945/understanding-the-properties-and-use-of-the-laplacian-matrix-and-its-norm).

### References

[1] Zachary karate club — [The KONECT Project](http://konect.cc/networks/ucidata-zachary/)