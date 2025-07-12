# Introduction to Big Data Analytics & Applications


## 🔍 What is **Big Data Analytics (BDA)?**

**Big Data Analytics** is the process of **collecting, organizing, and analyzing** large and complex data sets (known as **big data**) to discover useful information, patterns, and trends that help in decision-making.

---

### 📦 Key Characteristics of Big Data ("The 5 V’s")

| V            | Description                                                                                                     |
| ------------ | --------------------------------------------------------------------------------------------------------------- |
| **Volume**   | Massive amounts of data (terabytes, petabytes) from sources like social media, sensors, logs, etc.              |
| **Velocity** | Data is generated at high speed and must be processed quickly (real-time analytics).                            |
| **Variety**  | Data comes in many formats: structured (SQL), semi-structured (JSON, XML), unstructured (text, images, videos). |
| **Veracity** | The reliability and quality of the data.                                                                        |
| **Value**    | The actionable insights gained from analyzing the data.                                                         |

---

### 🔧 Types of Big Data Analytics

| Type                       | Purpose              | Example                                    |
| -------------------------- | -------------------- | ------------------------------------------ |
| **Descriptive Analytics**  | What happened?       | Monthly sales reports                      |
| **Diagnostic Analytics**   | Why did it happen?   | Root cause analysis of customer churn      |
| **Predictive Analytics**   | What will happen?    | Forecasting stock prices or product demand |
| **Prescriptive Analytics** | What should be done? | Suggesting optimal pricing strategies      |

---

### ⚙️ Tools & Technologies Used

| Category          | Tools                              |
| ----------------- | ---------------------------------- |
| **Storage**       | Hadoop HDFS, Amazon S3             |
| **Processing**    | Apache Spark, Hadoop MapReduce     |
| **Databases**     | MongoDB, Cassandra, HBase          |
| **Visualization** | Tableau, Power BI, Apache Superset |
| **Programming**   | Python, R, Scala, SQL              |

---

### 🧠 Applications of BDA

* **Healthcare**: Predicting disease outbreaks, personalized treatment
* **Retail**: Customer segmentation, dynamic pricing
* **Finance**: Fraud detection, algorithmic trading
* **Marketing**: Targeted ads, customer sentiment analysis
* **Manufacturing**: Predictive maintenance, supply chain optimization

---

### 📈 Benefits

* Improved decision-making
* Real-time analytics
* Enhanced customer experience
* Competitive advantage
* Cost efficiency through optimized operations

---
##### **OLTP**, **OLAP**, **ETL**, and **ELT** — are foundational in **data engineering**, **data analytics**, and **big data systems**. Here's a clear breakdown of each:

## 🔹 1. **OLTP** – *Online Transaction Processing*

### ➤ Purpose:

Handles **real-time transactional data** — frequent, short online transactions like **insert**, **update**, and **delete**.

### ➤ Characteristics:

| Feature       | Description                                   |
| ------------- | --------------------------------------------- |
| Use case      | Day-to-day operations (e.g., banking, retail) |
| Data volume   | Smaller transactions, high frequency          |
| Performance   | Fast read/write                               |
| Normalization | Highly normalized (many small tables)         |

### ➤ Example:

* ATM withdrawal
* Placing an order on an e-commerce site

---

## 🔹 2. **OLAP** – *Online Analytical Processing*

### ➤ Purpose:

Performs **complex analytical queries** on large volumes of historical data for business insights.

### ➤ Characteristics:

| Feature     | Description                          |
| ----------- | ------------------------------------ |
| Use case    | Business reporting, data mining      |
| Data volume | Large, aggregated historical data    |
| Performance | Optimized for read-heavy workloads   |
| Structure   | Denormalized (star/snowflake schema) |

### ➤ Example:

* Monthly sales reports
* Customer segmentation analytics

---

## 🔹 3. **ETL** – *Extract, Transform, Load*

### ➤ Purpose:

Used to **move data** from source systems (like OLTP databases) into a **data warehouse** after transforming it.

### ➤ Process:

1. **Extract** – Pull data from various sources
2. **Transform** – Clean, format, join, aggregate
3. **Load** – Store it in the data warehouse

### ➤ Use Case:

Traditional data warehousing (e.g., using tools like Talend, Informatica)

---

## 🔹 4. **ELT** – *Extract, Load, Transform*

### ➤ Purpose:

A modern approach where raw data is loaded first into the **data lake/warehouse**, and transformations are done **inside the target system**.

### ➤ Process:

1. **Extract** – Pull data from source
2. **Load** – Load raw data into a scalable system (e.g., BigQuery)
3. **Transform** – Use SQL/compute in-place

### ➤ Use Case:

Used in **cloud-based analytics** and **big data platforms** (e.g., Spark, Snowflake, Databricks)

---

## 🔄 Relationship Between Them

```plaintext
          +-------------+          +-------------+
          |   OLTP DB   | ──────►  |   ETL/ELT   |
          +-------------+          +-------------+
                                         │
                                         ▼
                                +----------------+
                                |    Data Lake   |
                                |   / Warehouse  |
                                +----------------+
                                         │
                                         ▼
                                   +----------+
                                   |   OLAP    |
                                   +----------+
```

* **OLTP** is the **source** (live transactions).
* **ETL/ELT** is the **pipeline** (data integration).
* **OLAP** is the **destination** for **analysis and reporting**.

---

## Data Warehouse 🏛️:  What is a **Data Warehouse**?

A **Data Warehouse** is a **centralized repository** that stores **large volumes of structured, historical data** from multiple sources. It is specifically designed for **querying and analysis**, not for daily operations.

---

### ✅ Key Features of a Data Warehouse:

* **Subject-Oriented**: Focused on business areas like sales, finance, or inventory.
* **Integrated**: Combines data from various sources (databases, files, APIs).
* **Time-Variant**: Stores historical data over time.
* **Non-Volatile**: Data doesn’t change frequently after loading.
* **Optimized for OLAP**: Fast read access, supports complex analytical queries.

---

## 🔄 How it’s linked to OLTP, OLAP, ETL, and ELT

Let's explore how everything connects logically:

---

### 📍 1. **OLTP → ETL/ELT → Data Warehouse → OLAP**

```plaintext
        +--------+          +--------+          +----------------+          +--------+
        |  OLTP  | ───────► | ETL/ELT| ───────► | Data Warehouse  | ───────► |  OLAP  |
        |System  |          |Pipeline|          | (e.g., BigQuery)|          | Tools  |
        +--------+          +--------+          +----------------+          +--------+
```

---

## 🔗 Connection Breakdown

### 🔹 1. **OLTP (Online Transaction Processing)**

* **Where data originates** (e.g., sales systems, user activity, banking transactions)
* Optimized for high-speed, real-time data entry and updates
* Examples: MySQL, PostgreSQL, Oracle (for transactions)

### 🔹 2. **ETL / ELT Pipelines**

* **ETL**: Data is extracted, transformed (cleaned, joined), then loaded into the warehouse.
* **ELT**: Data is extracted, loaded *raw* into the warehouse, and transformed *within* the warehouse.

> Tools: Apache NiFi, Airflow, Talend, dbt, AWS Glue, etc.

### 🔹 3. **Data Warehouse**

* The **central place to store** all the cleaned/aggregated data for analysis
* Schema: Denormalized (e.g., star/snowflake schemas)
* Examples: **Google BigQuery, Amazon Redshift, Snowflake, Azure Synapse, Teradata**

### 🔹 4. **OLAP (Online Analytical Processing)**

* Accesses data from the **data warehouse** for advanced **reporting, dashboards, and analysis**
* Supports multi-dimensional queries (e.g., pivot tables, drill-downs)
* Tools: **Tableau, Power BI, Looker, Superset**

---

## 🧠 Real-World Example

Let’s say you're running an e-commerce company:

| Step               | Example                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------- |
| **OLTP**           | Customer places orders, browses products (data stored in MySQL/PostgreSQL)               |
| **ETL/ELT**        | Nightly job pulls this data, cleans/transforms it                                        |
| **Data Warehouse** | Data is stored in Snowflake or BigQuery, partitioned by time                             |
| **OLAP**           | Business team uses Tableau/Power BI to analyze product sales trends or customer behavior |

---

## 📊 Summary Table

| Concept            | Purpose                       | Where It Happens    | Optimized For     |
| ------------------ | ----------------------------- | ------------------- | ----------------- |
| **OLTP**           | Transaction capture           | Source systems      | Fast reads/writes |
| **ETL/ELT**        | Data transformation/migration | Middleware or cloud | Integration       |
| **Data Warehouse** | Historical data storage       | Central repository  | Analytics         |
| **OLAP**           | Analysis/reporting            | On warehouse data   | Business insights |

---


## Fact & Dimension Table in Data Warehouse
---

## 🧱 What is a **Fact Table**?

A **Fact Table** is the **central table** in a data warehouse that contains **measurable, quantitative data** (i.e., **facts** or **metrics**).

### ✅ Characteristics:

* Contains **numeric values** like sales amount, profit, quantity, etc.
* Has **foreign keys** to dimension tables.
* Usually **very large** because it stores **transactions or events**.
* Used for **calculations** and **aggregation** (sum, avg, count, etc.)

### 🧾 Example: `Sales_Fact`

| Date\_ID | Product\_ID | Store\_ID | Units\_Sold | Revenue |
| -------- | ----------- | --------- | ----------- | ------- |
| 101      | 201         | 301       | 5           | ₹5000   |

---

## 🗂️ What is a **Dimension Table**?

A **Dimension Table** is a table that contains **descriptive attributes** (also called **dimensions**) related to the facts in the fact table.

### ✅ Characteristics:

* Contains **textual or categorical** data (names, types, categories).
* Provides **context** to the numeric data.
* Typically **smaller** than fact tables.
* Used for **filtering, grouping**, and **drilling down** in reports.

### 🧾 Example: `Product_Dim`

| Product\_ID | Product\_Name | Category   | Brand |
| ----------- | ------------- | ---------- | ----- |
| 201         | iPhone 13     | Smartphone | Apple |

---

## 🔗 How They Work Together (Schema Design)

### 🌟 **Star Schema**:

* Fact table in the **center**
* Dimension tables surrounding it
* Simple, fast for querying

```
       +-------------+
       |  Date_Dim   |
       +-------------+
             ▲
             |
+------------+-------------+
|        Sales_Fact        |
+------------+-------------+
             |
     ▲       ▲       ▲
+--------+ +--------+ +--------+
|Product | |Store   | |Customer|
|_Dim    | |_Dim    | |_Dim    |
+--------+ +--------+ +--------+
```

### ❄️ **Snowflake Schema**:

* Like star schema, but dimension tables are **normalized**
* More complex joins, but **storage-efficient**

---

## 📊 Real-World Example (E-commerce)

### 🔢 Fact Table: `Order_Fact`

| Order\_ID | Product\_ID | Customer\_ID | Date\_ID | Quantity | Total\_Amount |
| --------- | ----------- | ------------ | -------- | -------- | ------------- |
| 1001      | 101         | 501          | 20240701 | 2        | ₹40,000       |

### 📁 Dimension Tables:

* `Product_Dim`: iPhone 15, Electronics, Apple
* `Customer_Dim`: Lokesh, Bengaluru, India
* `Date_Dim`: 01-July-2024, Monday, Q3
* `Store_Dim`: Flipkart Online Store, Karnataka, South India

---

## 📌 Key Differences Summary

| Feature         | Fact Table                   | Dimension Table                    |
| --------------- | ---------------------------- | ---------------------------------- |
| **Data Type**   | Numeric, measurable          | Descriptive, categorical           |
| **Size**        | Usually large (millions+)    | Relatively small                   |
| **Primary Key** | Composite key (foreign keys) | Surrogate key (e.g., `Product_ID`) |
| **Purpose**     | Measures/events              | Context for those measures         |
| **Usage**       | Aggregation, KPIs            | Filtering, grouping, slicing       |

---

## ✅ Use in OLAP

* **Fact Table**: Used for calculations (total sales, average spend, etc.)
* **Dimension Tables**: Used for filters and drill-downs (by product, date, region, etc.)


