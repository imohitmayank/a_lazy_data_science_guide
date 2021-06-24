KG in Practice
=====================

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

```{bibliography}
:filter: docname in docnames
```
