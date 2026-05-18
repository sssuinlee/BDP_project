# BDP_project: Sequential Product Recommendation System

An RNN deep learning architecture designed to capture sequential customer behavioral patterns and predict next-action product preferences using large-scale e-commerce datasets.

## Project Overview
### Motivation 
While many modern e-commerce platforms have a wide range of articles across diverse categories, an excessive number of options can overwhelm customers, often causing them to leave without making a purchase.
The goal of this project is to help retailers select more relevant items and make the shopping experience more effective for customers, even amid a flood of products.

### Contributions
This project handles 11GB of high-dimensional e-commerce logs, scaling data preprocessing workflows through distributed computing, and feeds clean input sequences into a RNN-based recommendation model for training. The primary prediction task is to calculate the probability distribution of a user's next behavioral action (click, cart, or order) along with the specific product ID.

## Tech Stack
* **Language:** Python
* **Big Data & Distributed Computing:** Apache Hadoop, Apache Spark (PySpark Core / RDD)
* **Deep Learning Framework:** PyTorch
* **Distributed Deep Learning Interoperability:** SparkTorch (Integration library to run PyTorch models natively within Spark pipelines)

## Dataset
* **Source:** Sourced from the Kaggle Competition: OTTO – Multi-Objective Recommender System <a href="https://www.kaggle.com/competitions/otto-recommender-system">Link</a>
* **Data Size:** 11GB of raw e-commerce transaction logs.
* **Description:** Contains high-dimensional tabular customer behavioral data, including session IDs, event types(user actions), timestamps, and aids(product IDs).

## Key Features & Engineering Implementation

### 1. Large-Scale Data Pipeline & Feature Engineering
* **Time-Series Sequence Grouping:** Grouped interaction logs chronologically by `customer_id` and converted them into time-ordered event histories.
* **Joint Multi-Objective Encoding:** Transformed both categorical features (`product_id`) and behavioral actions (`event_type`) into unified vector spaces using One-Hot Encoding (OHE). 
* **Sliding Window Sequence Batching:** Developed an RDD-level sequence-slicing logic that structures continuous histories into a uniform sliding window layout (e.g., matching 5 consecutive historical events as an `input_batch` against the immediate subsequent event as the `target_batch`).

### 2. Model Architecture & Data Flow

This sequential recommendation system bridges a distributed data engine (PySpark) with a deep learning framework (PyTorch) via `SparkTorch`. The overall end-to-end operational pipeline and model layers are structured as follows:

### End-to-End Pipeline Workflow
```text
[ 11GB Raw Weblogs ] (Hadoop HDFS)
        │
        ▼ (PySpark Core / RDD)
[ Group by Customer_ID ] & [ Chronological Alignment ]
        │
        ▼ 
[ Joint One-Hot Encoding ] ──► Map (product_id [OHE] + event_type [OHE]) 
        │                          └─► Concatenated into a unified vector space
        ▼ 
[ Sliding Window Slicing ] ──► Formats continuous rows into (Input Batch [t-5 to t-1], Target Batch [t])
        │
        ▼ (SparkTorch Interoperability)
[ PyTorch RNN Architecture ] ──► Native distributed training on cluster memory RDDs
        │
        ▼ 
[ Softmax Optimization ] ──► Outputs Probabilistic Next-Action Intent & Product ID
