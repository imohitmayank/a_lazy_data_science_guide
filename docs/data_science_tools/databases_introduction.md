# Databases

## Introduction

Databases are like a smart repository of data, where you can keep data in a structured, organized, and secure manner. While you might argue something like, "Why can't I just keep them in a single txt file?", remember things starts to get messy when you have a lot of data *(millions of entries and more)* and you still want to be fast *(finding one entry in any modern database will be much faster than from a txt file)*. In a way, databases allow for efficient retrieval, manipulation, and analysis of data. They also offer powerful querying and analysis capabilities, enabling data scientists and analysts to extract insights and make data-driven decisions. This makes them a powerful tool for data science projects. 

This article will briefly focus on the different types of databases, their features and some important generic concepts. You can also find separate articles for individual databases in this section. So without further delay, let's get started!

## Types of Databases

### Relational Databases

Relational databases are the most common type of database. They are structured using tables, with each table containing rows and columns wherein each row represents a record, and each column contains a field. This simple structure makes it easier to query the database and find the information that is needed. Additionally, relational databases can be linked together using keys, allowing data to be shared between different tables. This makes it possible to link data from different sources and use it to generate meaningful insights. Hence the name “relational” database.

#### Comparison of Relational Databases

| Feature / Database      | Oracle                | MySQL                 | Microsoft SQL Server | PostgreSQL           | MariaDB              | SQLite               | IBM Db2              | SAP HANA             |
|-------------------------|-----------------------|-----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| **License**             | Commercial            | Open Source           | Commercial           | Open Source          | Open Source          | Open Source          | Commercial           | Commercial           |
| **ACID Compliance**     | Yes                   | Yes                   | Yes                  | Yes                  | Yes                  | Yes                  | Yes                  | Yes                  |
| **Scalability**         | Vertical/Horizontal   | Vertical              | Vertical/Horizontal  | Vertical/Horizontal  | Vertical/Horizontal  | Limited              | Vertical/Horizontal  | Horizontal           |
| **Partitioning**        | Yes                   | Yes                   | Yes                  | Yes                  | Yes                  | No                   | Yes                  | Yes                  |
| **Replication**         | Yes                   | Yes                   | Yes                  | Yes                  | Yes                  | No                   | Yes                  | Yes                  |
| **JSON Support**        | Yes                   | Limited               | Yes                  | Yes                  | Yes                  | No                   | Yes                  | Yes                  |
| **Geospatial Support**  | Yes                   | Limited               | No                   | Yes (PostGIS)        | Yes                  | No                   | Yes                  | Yes                  |
| **Stored Procedures**    | Yes                   | Yes                   | Yes                  | Yes                  | Yes                  | No                   | Yes                  | Yes                  |
| **User Base**           | Large Enterprises      | Web Applications      | Enterprises          | Diverse Applications  | Web Applications     | Embedded Applications | Enterprises          | Enterprises          |
| **Popularity Rank**     | 1                     | 2                     | 3                    | 4                    | 5                    | 6                    | 7                    | 8                    |

!!! Warning
    Some details in the above table *(like Popularity Rank)* are as per the information aggregated from multiple sources at the time the article was written *(August 2024)*. This could change over time. 

This table summarizes the key features and characteristics of some of the top relational databases, highlighting their strengths and use cases. Oracle leads in enterprise environments, while MySQL and PostgreSQL are popular for web applications and diverse workloads. Microsoft SQL Server is favored in corporate settings, and MariaDB is recognized for its compatibility with MySQL. SQLite is widely used for lightweight applications, and IBM Db2 and SAP HANA cater to specific enterprise needs.

## Concepts

### ACID Compliance

ACID compliance refers to a set of principles that ensure reliable processing of database transactions. The acronym ACID stands for **Atomicity**, **Consistency**, **Isolation**, and **Durability**. Each of these properties plays a crucial role in maintaining the integrity and reliability of data within database systems. [1]

- **Atomicity:** Atomicity guarantees that a transaction is treated as a single, indivisible unit. This means that either all operations within the transaction are completed successfully, or none are applied at all. If any part of the transaction fails, the entire transaction is rolled back, ensuring that the database remains unchanged in the event of an error or failure.

- **Consistency:** Consistency ensures that a transaction brings the database from one valid state to another, by adhering to all predefined rules and constraints *(PRIMARY KEY, FOREIGN KEY, NOT NULL, CHECK, etc)*. This property prevents illegal transactions that could corrupt the database, ensuring that all data remains valid and meaningful throughout the transaction process.

- **Isolation:** Isolation ensures that concurrent transactions do not interfere with each other. Each transaction is executed in such a way that it appears to be the only transaction being processed at that time, even if multiple transactions are occurring simultaneously. This prevents issues such as dirty reads, where one transaction reads data modified by another uncommitted transaction.

- **Durability:** Durability guarantees that once a transaction has been committed, it will remain so, even in the event of a system failure. This means that the changes made by the transaction are permanently recorded in the database, ensuring data persistence and reliability.

ACID compliance is critical for applications that require high levels of data integrity, such as financial systems, healthcare databases, and any other systems where data accuracy is paramount. It helps prevent data loss, ensures consistent data states, and facilitates reliable transaction processing, which is essential for maintaining trust and accuracy in data-driven applications.

## References

[1] [PlanetScale.com - What does ACID compliance mean?](https://planetscale.com/learn/articles/what-does-acid-compliance-mean)