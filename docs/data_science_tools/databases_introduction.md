# Databases

## Introduction

Databases are an essential part of any data science project. They provide a structured, organized, and secure storage for data. They allow for efficient retrieval, manipulation, and analysis of data.  

## Relational Databases

Relational databases are structured using tables, with each table containing columns and rows. Each row represents a record, and each column contains a field. This structure makes it easier to query the database and find the information that is needed. Additionally, relational databases can be linked together using keys, allowing data to be shared between different tables. This makes it possible to link data from different sources and use it to generate meaningful insights. Hence the name “relational” database.

### Comparison of Relational Databases

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

This table summarizes the key features and characteristics of some of the top relational databases, highlighting their strengths and use cases. Oracle leads in enterprise environments, while MySQL and PostgreSQL are popular for web applications and diverse workloads, respectively. Microsoft SQL Server is favored in corporate settings, and MariaDB is recognized for its compatibility with MySQL. SQLite is widely used for lightweight applications, and IBM Db2 and SAP HANA cater to enterprise needs with robust features.