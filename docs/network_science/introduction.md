- Quoting [Wikipedia](https://en.wikipedia.org/wiki/Network_science), *"Network science is an academic field which studies complex networks such as telecommunication networks, computer networks, biological networks, cognitive and semantic networks, and social networks, considering distinct elements or actors represented by nodes (or vertices) and the connections between the elements or actors as links (or edges)."*
- To better understand network science, we should understand the underlying building blocks i.e. Graphs!

### Graphs 101

- Graph or Networks is used to represent relational data, where the main entities are called nodes. A relationship between nodes is represented by edges. A graph can be made complex by adding multiple types of nodes, edges, direction to edges, or even weights to edges. One example of a graph is shown below. 

<figure markdown> 
        ![](../imgs/ns_intro_graph101_karate.png)
        <figcaption>Karate dataset visualization @ [Network repository](http://networkrepository.com/graphvis.php)</figcaption>
        </figure>

- The graph above is the Karate dataset [1] which represents the social information of the members of a university karate club. Each node represents a member of the club, and each edge represents a tie between two members of the club. The left info bar states several graph properties like a number of nodes, edges, density, degree, etc. [Network repository](http://networkrepository.com/graphvis.php) contains many such networks from different fields and domains and provides visualization tools and basic stats as shown above.

### References

[1] Zachary karate club — [The KONECT Project](http://konect.cc/networks/ucidata-zachary/)