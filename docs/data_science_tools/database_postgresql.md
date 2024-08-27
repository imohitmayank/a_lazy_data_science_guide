## Introduction to PostgreSQL

PostgreSQL, often simply called Postgres, is a powerful, open-source object-relational database management system (ORDBMS). It has a strong reputation for reliability, feature robustness, and performance. PostgreSQL was first developed in 1986 at the University of California, Berkeley as part of the POSTGRES project. It has since evolved into one of the most advanced and widely-used database systems, with a strong community supporting its development. PostgreSQL supports all major operating systems, including Linux, Mac OS, and Windows.

## Key Features of PostgreSQL

PostgreSQL offers a wide range of features that make it a popular choice for many applications:

1. **Extensive data types**: PostgreSQL supports a large variety of built-in data types and allows users to define their own custom data types. It can handle complex data types such as arrays, JSON, and geometric types.

2. **ACID compliance**: PostgreSQL adheres to the ACID principles (Atomicity, Consistency, Isolation, Durability), ensuring reliable and trustworthy transactions. [More details](databases_introduction.md#acid-compliance)

3. **Concurrency control**: PostgreSQL uses multi-version concurrency control (MVCC) to provide high concurrency without conflicts, allowing multiple transactions to access the same data simultaneously.

4. **Advanced querying capabilities**: PostgreSQL supports complex SQL queries, subqueries, common table expressions (CTEs), recursive queries, and window functions. It also allows users to define their own functions, triggers, and stored procedures in various programming languages.

5. **Full-text search**: PostgreSQL provides powerful full-text search capabilities, including stemming, ranking, and phrase-searching support. It uses indexes like B-tree, hash, and GiST to optimize search performance.

6. **Replication and high availability**: PostgreSQL supports various replication strategies, such as asynchronous streaming, logical, and synchronous replication, providing data redundancy, fault tolerance, and high availability.

7. **Security and authentication**: PostgreSQL offers robust security features, including SSL encryption, username/password authentication, LDAP authentication, Kerberos authentication, role-based access control (RBAC), and row-level security (RLS).

## Setting Up PostgreSQL

To get PostgreSQL running on your local machine, you will need to have the following tools installed:

1. **PostgreSQL Server**: You can follow the step-by-step instructions provided on the [official website](https://www.postgresql.org/download/). Once the installation is complete, you can run the server by opening the application.

2. **PostgreSQL Admin/Dev Tools**: Once the PostgreSQL server is installed, you can install tools to manage and interact with PostgreSQL. There are multiple choices, each with its own set of unique features and all of them support the basic functionalities. Here are some famous ones - [PgAdmin](https://www.pgadmin.org/), [DBeaver](https://dbeaver.io/), or you can even use terminal tools like [Psql](https://www.postgresql.org/docs/current/app-psql.html).

!!! Hint
    **Installation on Mac**

    PostgreSQL can be installed on Mac by using `homebrew`. Run the command `brew install postgresql`. For more details and options, follow the [official website](https://www.postgresql.org/download/macosx/).

## Learning the Basics

Practice makes man perfect, so let's learn PostgreSQL through sample codes. Below are some sample code snippets in increasing order of complexity, designed to help you understand various aspects of PostgreSQL.

!!! Hint
    Before we begin, please note that to interact with the database, you need to use the PostgreSQL Query Language, which is a variant of the SQL language. If you are using terminal, then you can activate psql mode by running `psql`. Once inside you can connect to the database by running the following command:

    ```sql
    -- Connecting to a PostgreSQL database
    -- Use a client or terminal with appropriate access credentials
    \c my_database;
    ```
    Or you can use any of the user-interface tools like PgAdmin for better user experience. 

**1. Creating a Database**

```sql
-- Creating a database. Replace `my_database` with your database name
CREATE DATABASE my_database;
```

**2. Creating a Table**


```sql
-- Creating a simple table. Replace `employees` with your table name
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    position VARCHAR(50),
    departmentid INT,
    salary DECIMAL
);
```

!!! Hint
    [Here is a detailed list](https://www.postgresql.org/docs/current/datatype.html) of all supported data types in PostgreSQL. Note, you can also [create custom data types](https://www.postgresql.org/docs/current/sql-createtype.html).

**3. Inserting Data**

```sql
-- Inserting data into the table
INSERT INTO employees (name, position, salary)
VALUES ('John Doe', 'Software Engineer', 70000);
```

**4. Basic Data Retrieval**

```sql
-- Retrieving all data from a table
SELECT * FROM employees;

-- Limiting the number of rows returned
SELECT * FROM employees LIMIT 10;

-- Retrieving specific columns
SELECT name, position FROM employees;

-- Retrieving data in descending order
SELECT * FROM employees ORDER BY salary DESC;
```

**5. Data Retrieval with Conditions**

```sql
-- Retrieving specific data with a condition
SELECT name, position FROM employees WHERE salary > 50000;

-- Filtering on string columns
SELECT * FROM employees WHERE name LIKE '%Doe%';

-- Filtering on datetime columns
SELECT * FROM orders WHERE order_date BETWEEN '2022-01-01' AND '2022-02-01';

-- Filtering on datetime columns with interval (works same as above)
SELECT * FROM orders WHERE order_date BETWEEN '2022-01-01' AND '2022-02-01'::date + interval '1 month';

-- To filter based on multiple conditions and values
SELECT * FROM employees WHERE name LIKE '%Doe%' AND salary > 50000
    AND position in ('Software Engineer', 'Data Scientist');
```

**6. Updating Data**

```sql
-- Updating data in the table
UPDATE employees SET salary = 75000 WHERE name = 'John Doe';
```

**7. Deleting Data**

```sql
-- Deleting data from the table
DELETE FROM employees WHERE id = 1;

-- Deleting all data from the table
DELETE FROM employees;

-- Deleting the table
DROP TABLE employees;

-- Deleting multiple tables
DROP TABLE employees, departments;
```

**8. Joining Tables**

```sql
-- Creating another table
CREATE TABLE departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50)
);

-- Inserting data into the new table
INSERT INTO departments (name) VALUES ('Engineering');

-- Joining two tables
SELECT employees.name, departments.name AS department_name
FROM employees
JOIN departments ON employees.departmentid = departments.id;
```

**9. Using Aggregate Functions**

```sql
-- Using an aggregate function to get the average salary
SELECT AVG(salary) FROM employees;

-- Group by a column (ex: getting the average salary by department)
SELECT department_name, AVG(salary) AS avg_salary
FROM employees
JOIN departments ON employees.id = departments.id
GROUP BY department_name;
```

**10. Complex Query with Subquery and Grouping**

```sql
-- Finding the highest salary in each department
SELECT department_name, MAX(salary) AS max_salary
FROM (
    SELECT employees.name, employees.salary, departments.name AS department_name
    FROM employees
    JOIN departments ON employees.id = departments.id
) AS department_salaries
GROUP BY department_name;
```

These examples cover a range of basic to more complex tasks you can perform with PostgreSQL, from establishing a connection to executing advanced queries. As you become more comfortable with these operations, you'll be able to tackle more complex scenarios and optimize your database interactions.

## Python Sample Code

There are multiple python packages available for PostgreSQL like, [psycopg2](https://pypi.org/project/psycopg2/) and [asyncpg](https://pypi.org/project/asyncpg/). For this section, we will use [asyncpg](https://pypi.org/project/asyncpg/) package that provides support for asynchronous programming.

A sample code to connect to the PostgreSQL server and fetch the result is shown below, 

```python linenums="1"
# import 
import asyncio
import asyncpg

# the main function that connect to the PostgreSQL server, 
# fetch the result and print the result
async def run():
    # connect to the PostgreSQL server
    conn = await asyncpg.connect(user='postgres', password='admin',
                                 database='mydb', host='localhost')
    # fetch the result
    result = await conn.fetch(
        'SELECT * FROM mytbl LIMIT 1'
    )
    # print the result
    print(dict(result))
    # close the connection
    await conn.close()

if __name__ == '__main__':
    # run the code
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
```

Creating dynamic queries based on user input can be easily done by passing the variables to the `fetch` function. Below is the modification you need to do. If you notice, we have two variables in the query for `id` and `limit` denoted by `$1` and `$2` respectively. The respective values are passed in the `fetch` function. Rest of the code remains same.

```python linenums="1"
# fetch the result
result = await conn.fetch(
    'SELECT * FROM mytbl where id = $1 LIMIT $2',
    123, 1
)
```

You can use `conn.execute` to run the query without fetching the result. Below is the modification to the code shown above.

```python linenums="1"
# insertion example (one row)
result = await conn.execute(
    'INSERT INTO mytbl (code, name) VALUES ($1, $2) where id = $3',
    123, 'mohit', 1
)
```

If you want to execute for multiple rows, you can use `conn.executemany` instead of `conn.execute`. Below is the modification to the code shown above.

```python linenums="1"
# insertion example (multiple rows)
result = await conn.executemany(
    'INSERT INTO mytbl (code, name) VALUES ($1, $2) where id = $3',
    [(123, 'mohit', 1), (124, 'mayank', 2)]
)
```

You might want to create a generic function to execute queries and retry in case of failure. Here is how you can do it using the `tenacity` library. The below code will retry 3 times if the query fails with exponential backoff.

```python linenums="1"
# import
import asyncio
import asyncpg
import functools
from tenacity import TryAgain, retry, stop_after_attempt, wait_exponential

# custom retry logging function
def custom_retry_log(retry_state, msg):
    if retry_state.attempt_number != 1:
        print(f"Retrying {retry_state.attempt_number - 1} for {msg}")

# main function
async def execute_fetch_script(script, values=(), msg=None, retry_on_failure=True):
    # create connection
    conn = await asyncpg.connect(user='postgres', password='admin',
                                    database='mydb', host='localhost')
    try:
        # retry mechanism
        log_callback = functools.partial(custom_retry_log, msg=msg)

        # retry mechanism
        @retry(wait=wait_exponential(multiplier=2, min=2, max=16),
            stop=stop_after_attempt(4),
            after=log_callback, reraise=True)
        async def retry_wrapper():
            try:
                # execute the select SQL script
                records = await conn.fetch(script, *values)
                project_records = [dict(record) for record in records]
                print(project_records) # remove this
                return project_records
            except Exception as e:
                if retry_on_failure:
                    raise TryAgain(e)
                else:
                    print(f"Failure in {msg} - {e}")
                    return

        # db call wrapper
        return await retry_wrapper()
    except Exception as e:
        raise Exception(f"Failure in {msg} - {e}")
    finally:
        # close db connections
        await conn.close()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    script  ='SELECT * FROM mytbl where projectid = $1 LIMIT $2'
    values = (2, 1)
    loop.run_until_complete(execute_fetch_script(script, values, "Testing Run"))
```

If you notice, all of the above examples are executing the query in one transaction. In case you want to execute multiple queries in one transaction, you can do as shown below, 

```python linenums="1"
# import
import asyncio
import asyncpg
import functools
from tenacity import TryAgain, retry, stop_after_attempt, wait_exponential

# create the connection
conn = await asyncpg.connect(user='postgres', password='admin',
                                database='mydb', host='localhost')

# start the transaction
async with conn.transaction():

    try: 
        # execute the select SQL script
        records = await conn.fetch('SELECT * FROM mytbl where projectid = $1 LIMIT $2', 2, 1)

        # update the table
        await conn.execute('UPDATE mytbl SET name = $1 where projectid = $2', 'mohit', 2)

    # handle exception
    except Exception as e:
        # in case of exception rollback the transaction
        await conn.execute('ROLLBACK;')

    finally:
        # close db connections
        await conn.close()
```

## Snippets

Real world problems will require much more than what we covered in the above sections. Lets cover some important queries in this section.

**Casting a column to a different data type**

```sql
-- Casting a column to a different data type
SELECT CAST(salary AS VARCHAR) FROM employees;
```

**Using JSONB column**

```sql
-- Extracting data from JSONB column
-- Suppose data column contains {"name": "John", "address": {"city": "New York", "state": "NY"}}
SELECT name, jsonb_extract_path(data, 'address', 'city') AS city FROM employees;
```

**Extracting components from a DateTime column**

```sql
-- Extracting month from DATE column
-- Suppose in a tbl, order_date col contains info like 2022-01-01 
SELECT DATE_TRUNC('month', order_date) AS month, COUNT(*) AS order_count
FROM orders
GROUP BY month
ORDER BY month;

-- Extract year from DATE column, use: DATE_TRUNC('year', order_date)
-- Extract quarter from DATE column, use: DATE_TRUNC('quarter', order_date)
-- Extract week from DATE column, use: DATE_TRUNC('week', order_date)
-- Extract day from DATE column, use: DATE_TRUNC('day', order_date)
-- Extract hour from DATE column, use: DATE_TRUNC('hour', order_date)
-- Extract minute from DATE column, use: DATE_TRUNC('minute', order_date)
-- Extract second from DATE column, use: DATE_TRUNC('second', order_date)
```

## Conclusion

PostgreSQL's combination of features, performance, and reliability makes it a popular choice for a wide range of applications, from small projects to large-scale enterprise systems. Its open-source nature, strong community support, and continuous development ensure that PostgreSQL will remain a leading database management system for years to come. Hope this article helped you understand the basics of PostgreSQL and piqued your interest in learning more.

## References

[1] GeeksforGeeks - [What is PostgreSQL?](https://www.geeksforgeeks.org/what-is-postgresql-introduction/) | [PostgreSQL Tutorial](https://www.geeksforgeeks.org/postgresql-tutorial/)

[2] w3schools - [PostgreSQL Tutorial](https://www.w3schools.com/postgresql/postgresql_intro.php)

[3] Tutorialspoint - [PostgreSQL Tutorial](https://www.tutorialspoint.com/postgresql/postgresql_overview.htm)
