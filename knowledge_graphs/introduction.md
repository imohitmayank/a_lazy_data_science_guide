Introduction
===========================

## What is a Knowledge graph?

- To better under Knowledge graphs, let's start by understanding its basic unit i.e. "fact".

- A fact is the most basic piece of information that can be stored in a KG. Facts can be represented in form of triplets with either of the following ways,
  - **HRT**: `<head, relation, tail>`
  - **SPO**: `<subject, predicate, object>`

- Let's follow the HRT representation for this article.

- Facts contain 3 elements which can be further represented as a graph,
  - **Head or tail**
      - entities which are real-world objects or abstract concepts
      - represented as nodes
  - **Relation**
    - the connection between entities
    - represented as edges

- A simple KG example is provided below. One example of fact could be `<BoB, is_interested_in, The_Mona_Lisa>`. You can see the KG is nothing but a collection of multiple such facts.

```{figure} /imgs/kg_1.png
---
height: 300px
---
A sample knowledge graph. {cite}`rdf_primer`
```

----------

## Why knowledge graphs?

- This is the first and a very valid question anyone could ask. We will try to go through some points wherein we compare KG with simple graph and even other ways of storing information. The aim is to highlight the major pros of using KG.

### Compared to Normal Graph

- **Heterogenous data:** supports different type of entities and relations
- **Model real-world information:** closer to our brain's mental model of the world
- **Perform logical reasoning:** traverse the graphs in a path to make logical connections

### Compared to other storage types

- **Structured representation:** far cry from unstructured representations like text data
- **Removes redundancy:** compared to tabular data, there is no need to add mostly empty columns or rows to add new data
- **Query complex information:** better than SQL for data where relationship matters more than individual data points (for example, in case you have to do lots of JOIN statements in a SQL query, which is inherently slow)

----------

## Real world examples of KG

- Knowledge graphs are used for a large number of tasks -- be it for logical reasoning, explainable recommendations to even just being a better way to store information.
- There are two very interesting examples which we can discuss briefly.

### Google Knowledge Panel

- Google search about a famous person, location or concepts return a knowledge panel on the right (as shown in the image below)
- The panel contains a wide variety of information (description, education, born, died, quotes, etc) and interestingly in different formats--(text, image, dates, numbers, etc).
- All this information can be stored in a KG and one such example is shown below. This showcase how easy it is to store information and also note how intuitive it is to just read and understand the fact from a KG.

```{figure} /imgs/kg_ex_1.png
---
height: 300px
---
Example of knowledge graph-based knowledge panel used by Google. [Right] the actual panel shown by google when you search for Einstein. [left]  recreation of how we might store similar information in a KG. Source: by Author + Google.
```
### Movie recommendation

- Classical algorithms considered user-item interactions to generate recommendations. Over time, newer algorithms are considering additional information about the user as well as items, to improve the recommendations.
- Below, we can see one KG which not only contains user-item (here movies) connections but also user-user interactions and item attributes. The idea is that provided all this additional information, we can make much more accurate and informed suggestions.
- Without going into the exact algorithm (which we will later in the article), let's rationalize what recommendations could be generated.
- "Avatar" could be recommended to,
	- Bob: as it belongs to the Sci-fi genre same as Interstellar and Inception (which is already watched by Bob)
	- Alice: as it is directed by James Cameron (Titanic)
- "Blood Diamond" recommended to,
	- Bob: as DiCaprio acted in Inception as well

```{figure} /imgs/kg_ex_2.png
---
height: 300px
---
A sample knowledge graph for movie recommendation task. {cite}`guo2020survey`
```

----------

## Open-Source Knowledge graphs

Some of the famous open-source knowledge graphs are,

- **DBpedia**: is a crowd-sourced community effort to extract structured content from the information created in various Wikimedia projects.
- **Freebase**: a massive, collaboratively edited database of cross-linked data. Touted as “an open shared database of the world's knowledge”
- **OpenCyc**: is a gateway to the full power of Cyc, one of the world's most complete general knowledge base and commonsense reasoning engine.
- **Wikidata**: is a free, collaborative, multilingual, secondary database, collecting structured data to provide support for Wikipedia
- **YAGO**:  huge semantic knowledge base, derived from Wikipedia, WordNet and GeoNames.

```{figure} /imgs/prackg_opensourcekg.png
Stats for some of the famous open source knowledge graphs {cite}`färber2018knowledge`
```

----------

## Create custom Knowledge graph

We may want to create our own KG which is specific to a domain, group or organization.

We can either do it
- manually - which is tedious, or
- automate the process - where we create a KG from existing unstructured knowledge base like text documents using NLP pipeline.

- Steps followed for automatic KG creation,

```{figure} /imgs/prackg_createkg.png
Steps involved in creating a custom knowledge graph. Source: Author + {cite}`Ji_2021`
```

----------

## Ontology of Knowledge graph

- An ontology is a model of (a relevant part of) the world, listing the types of entities, the relationships that connect them, and constraints on the ways that entities and relationships can be combined.

- Resource Description Framework (RDF) and Web Ontology Language (OWL) are some of the vocabulary framework used to model ontology.

- They provides a common framework for expressing this information so it can be exchanged between applications without loss of meaning.


```{figure} /imgs/prackg_schema.png
RDF schema triplets (informal). Source: Author + {cite}`rdf_primer`
```

- RDF provides languages for creating ontology.

- Syntax in Turtle language for creating the graph shown on the right,

```{figure} /imgs/prackg_turtle.png
Script in Turtle language to create the sample knowledge graph. Source: Author + {cite}`rdf_primer`
```

----------

## Query a Knowledge graph

### SPARQL

- Once facts are stored as RDF, we can query them to extract relevant information.

- SPARQL is a RDF query language—that is, a semantic query language for databases—able to retrieve and manipulate data stored in RDF format.

- Some interesting reads are [Walkthrough Dbpedia And Triplestore](https://mickael.kerjean.me/2016/05/20/walkthrough-dbpedia-and-triplestore/)

```{figure} /imgs/prackg_querykg_sparql.png
Query Knowledge graph using SPARQL language {cite}`wikidata_sparql_query_helper`
```

### API

- Open-source KG also provide several APIs for ready-made queries. (ex: wikidata)

```{figure} /imgs/prackg_querykg_api.png
Query Knowledge graph using available APIs {cite}`wikidata_api_services`
```

## References


```{bibliography}
:filter: docname in docnames
```