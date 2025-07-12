
---

# 🧪 Real-Time Data Warehouse Project: *E-commerce Sales Analytics*

### Platform: **Google BigQuery**

📁 Format: GitHub-ready, modular, and production-oriented

---

## 🎯 Objective

Build a real-time analytics system to:

* Monitor **sales**, **product performance**, and **regional trends**
* Enable BI tools for OLAP reporting
* Support both **batch ingestion (CSV)** and **streaming data (Pub/Sub or Kafka)**

---

## 🗂 GitHub-Style Project Structure

```plaintext
ecommerce_data_warehouse/
│
├── data/                        # 📦 Sample data (CSV format)
│   ├── dim_product.csv
│   ├── dim_customer.csv
│   ├── dim_date.csv
│   ├── dim_store.csv
│   └── fact_sales.csv
│
├── sql/                         # 🧾 BigQuery SQL scripts
│   ├── create_tables.sql
│   ├── load_staging.sql
│   ├── transform_and_load.sql
│   └── olap_queries.sql
│
├── ingestion/                  # 🔁 Ingestion automation
│   ├── cloud_function/
│   │   └── main.py             # Cloud Function to load to BQ
│   └── pubsub_stream/
│       └── stream_processor.py # Pub/Sub stream handler (Kafka adaptable)
│
├── dashboard/                  # 📊 Dashboard template
│   └── looker_studio_config.json
│
├── notebooks/                  # 📒 Optional: EDA in Colab/Jupyter
│   └── sales_analysis.ipynb
│
└── README.md                   # 📘 Project documentation
```

---

## ✅ Sample CSVs (`/data`)

### 🔹 `fact_sales.csv`

```csv
sale_id,product_id,customer_id,date_id,store_id,quantity,total_amount
1,101,201,20240701,301,2,40000
2,102,202,20240701,302,1,20000
```

### 🔹 `dim_product.csv`

```csv
product_id,product_name,category,brand,price
101,iPhone 13,Smartphone,Apple,20000
102,MacBook Air,Laptop,Apple,60000
```

*(Additional files for `dim_customer`, `dim_date`, `dim_store`)*

---

## ✅ SQL Scripts (`/sql`)

### 🔧 `create_tables.sql` (BigQuery)

```sql
CREATE TABLE IF NOT EXISTS ecommerce.fact_sales (
  sale_id INT64,
  product_id INT64,
  customer_id INT64,
  date_id INT64,
  store_id INT64,
  quantity INT64,
  total_amount FLOAT64
);

CREATE TABLE IF NOT EXISTS ecommerce.dim_product (
  product_id INT64,
  product_name STRING,
  category STRING,
  brand STRING,
  price FLOAT64
);
```

### 🔄 `transform_and_load.sql`

```sql
-- Example: Inserting cleaned product data
INSERT INTO ecommerce.dim_product
SELECT DISTINCT
  product_id,
  INITCAP(product_name),
  category,
  brand,
  SAFE_CAST(price AS FLOAT64)
FROM ecommerce_staging.stg_product;
```

### 📊 `olap_queries.sql`

```sql
-- Total sales by brand
SELECT brand, SUM(total_amount) AS total_revenue
FROM ecommerce.fact_sales f
JOIN ecommerce.dim_product p ON f.product_id = p.product_id
GROUP BY brand
ORDER BY total_revenue DESC;
```

---

## ✅ Cloud Function for Automated Ingestion (`/ingestion/cloud_function/main.py`)

```python
import base64
from google.cloud import bigquery

def load_to_bigquery(event, context):
    client = bigquery.Client()
    dataset = "ecommerce_staging"
    table = "stg_sales"

    data = base64.b64decode(event['data']).decode('utf-8')
    rows = [line.split(",") for line in data.strip().split("\n")]

    errors = client.insert_rows_json(f"{dataset}.{table}", [
        {
            "sale_id": int(row[0]),
            "product_id": int(row[1]),
            "customer_id": int(row[2]),
            "date_id": int(row[3]),
            "store_id": int(row[4]),
            "quantity": int(row[5]),
            "total_amount": float(row[6])
        } for row in rows[1:]  # Skip header
    ])

    if errors:
        print("Encountered errors: ", errors)
    else:
        print("Data loaded successfully.")
```

---

## ✅ Streaming Ingestion (Pub/Sub or Kafka Adaptable)

### 🔄 `ingestion/pubsub_stream/stream_processor.py`

```python
from google.cloud import pubsub_v1
from google.cloud import bigquery
import json

def callback(message):
    client = bigquery.Client()
    table_id = "ecommerce_staging.stg_sales"
    data = json.loads(message.data.decode("utf-8"))

    row = [{
        "sale_id": data["sale_id"],
        "product_id": data["product_id"],
        "customer_id": data["customer_id"],
        "date_id": data["date_id"],
        "store_id": data["store_id"],
        "quantity": data["quantity"],
        "total_amount": data["total_amount"]
    }]
    errors = client.insert_rows_json(table_id, row)
    if errors:
        print("BQ insert errors:", errors)
    message.ack()

# Subscriber setup (can be run as daemon)
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path("your-project", "sales-sub")
subscriber.subscribe(subscription_path, callback=callback)
print("Listening for messages...")
```

> ✅ Kafka users can adapt this with `confluent_kafka` and forward messages similarly.

---

## ✅ Dashboard Template (`dashboard/looker_studio_config.json`)

* Filters: Product, Region, Date Range
* KPIs: Total Revenue, Avg Order Value, Units Sold
* Graphs:

  * Monthly Revenue Trend
  * Sales by Category
  * Regional Sales Heatmap

*(Import directly into Looker Studio or Power BI)*

---

## ✅ Jupyter Notebook (`/notebooks/sales_analysis.ipynb`)

* Sample queries using `google.cloud.bigquery`
* EDA: Revenue by Brand, Store Heatmap
* Matplotlib/Seaborn for data visualization

---

## 🧩 Optional Enhancements

* Orchestrate with **Apache Airflow** (via Cloud Composer)
* Add **dbt** for data modeling
* Deploy with **Terraform/IaC**
* Implement data validation via **Great Expectations**

---

## ✅ Deployment Guide (`README.md`)

Includes:

* Setup instructions for GCP & BigQuery
* Cloud Function deployment (with `gcloud functions deploy`)
* Pub/Sub topic + subscription creation
* BI dashboard linking steps
