# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 01: Data Ingestion & Exploratory Data Analysis (EDA)
# MAGIC 
# MAGIC **Purpose**: Load raw humanitarian aid datasets, perform comprehensive EDA, identify join keys, and write raw data to Delta Lake tables.
# MAGIC 
# MAGIC **Steps**:
# MAGIC 1. Initialize Spark session and create database
# MAGIC 2. Load all CSV datasets from `data/` using PySpark
# MAGIC 3. Run schema and basic statistics for each dataset
# MAGIC 4. Compute null counts, distinct values, min/max for numeric columns
# MAGIC 5. Identify potential join keys from column names and value analysis
# MAGIC 6. Write raw datasets to Delta tables: `humanitarian.raw_{dataset_name}`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Initialize Spark and Database

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, countDistinct, isnan, isnull, when, lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import os
from pathlib import Path

# Initialize Spark session
spark = SparkSession.builder \
    .appName("humanitarian-aid-analytics") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Create database
spark.sql("CREATE DATABASE IF NOT EXISTS humanitarian")
print("✓ Database 'humanitarian' created successfully")

# Define data directory
DATA_DIR = "/Workspace/Repos/..." # Update with your actual workspace path, or use relative path
# For this exercise, assume data files are in ./data/ relative to notebook
DATA_DIR = "./data"

print(f"✓ Data directory: {DATA_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load All CSV Datasets

# COMMAND ----------

# Define datasets to load
datasets_to_load = {
    "hno": "hpc_hno_2025.csv",
    "hrp": "humanitarian-response-plans.csv",
    "fts_requirements": "fts_requirements_funding_globalcluster_global.csv",
    "fts_incoming": "fts_incoming_funding_global.csv",
    "fts_internal": "fts_internal_funding_global.csv",
    "fts_outgoing": "fts_outgoing_funding_global.csv",
    "population": "cod_population_admin1.csv"
}

raw_datasets = {}

for dataset_name, file_name in datasets_to_load.items():
    file_path = f"{DATA_DIR}/{file_name}"
    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True, multiLine=True, escape='"')
        raw_datasets[dataset_name] = df
        print(f"✓ Loaded {dataset_name}: {file_name}")
    except Exception as e:
        print(f"✗ Failed to load {dataset_name}: {str(e)}")

print(f"\nTotal datasets loaded: {len(raw_datasets)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Schema Analysis

# COMMAND ----------

print("=" * 80)
print("DATASET SCHEMAS")
print("=" * 80)

for dataset_name, df in raw_datasets.items():
    print(f"\n{dataset_name.upper()}")
    print("-" * 80)
    df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Row Counts and Basic Statistics

# COMMAND ----------

print("=" * 80)
print("ROW COUNTS AND BASIC STATISTICS")
print("=" * 80)

dataset_stats = {}

for dataset_name, df in raw_datasets.items():
    row_count = df.count()
    col_count = len(df.columns)
    
    dataset_stats[dataset_name] = {
        "rows": row_count,
        "columns": col_count,
        "column_names": df.columns
    }
    
    print(f"\n{dataset_name.upper()}")
    print(f"  Row count: {row_count:,}")
    print(f"  Column count: {col_count}")
    print(f"  Columns: {', '.join(df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Null Count Analysis

# COMMAND ----------

print("=" * 80)
print("NULL VALUE ANALYSIS")
print("=" * 80)

for dataset_name, df in raw_datasets.items():
    print(f"\n{dataset_name.upper()}")
    print("-" * 80)
    
    null_counts = df.select([
        count(when(isnull(col(c)), 1)).alias(c) 
        for c in df.columns
    ]).collect()[0].asDict()
    
    # Calculate percentages
    row_count = df.count()
    
    for col_name, null_count in sorted(null_counts.items(), key=lambda x: x[1], reverse=True):
        null_pct = (null_count / row_count * 100) if row_count > 0 else 0
        if null_count > 0:
            print(f"  {col_name}: {null_count:,} nulls ({null_pct:.1f}%)")
    
    # Summary
    total_nulls = sum(null_counts.values())
    print(f"  Total null values: {total_nulls:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Distinct Values Analysis

# COMMAND ----------

print("=" * 80)
print("DISTINCT VALUES ANALYSIS (Top Columns by Cardinality)")
print("=" * 80)

for dataset_name, df in raw_datasets.items():
    print(f"\n{dataset_name.upper()}")
    print("-" * 80)
    
    distinct_counts = {}
    for col_name in df.columns:
        try:
            distinct_count = df.select(countDistinct(col(col_name))).collect()[0][0]
            distinct_counts[col_name] = distinct_count
        except:
            distinct_counts[col_name] = "ERROR"
    
    # Sort by distinct count
    sorted_cols = sorted(distinct_counts.items(), key=lambda x: x[1] if isinstance(x[1], int) else 0, reverse=True)
    
    for col_name, distinct_count in sorted_cols[:15]:  # Show top 15
        print(f"  {col_name}: {distinct_count} distinct values")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Sample Data Preview

# COMMAND ----------

print("=" * 80)
print("SAMPLE DATA (First 5 rows per dataset)")
print("=" * 80)

for dataset_name, df in raw_datasets.items():
    print(f"\n{dataset_name.upper()}")
    print("-" * 80)
    df.limit(5).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Identify Potential Join Keys

# COMMAND ----------

print("=" * 80)
print("POTENTIAL JOIN KEYS - COLUMN NAME ANALYSIS")
print("=" * 80)

# Collect all column names
all_columns = {}
for dataset_name, df in raw_datasets.items():
    all_columns[dataset_name] = set(col.lower() for col in df.columns)

# Look for common column patterns
join_key_candidates = {
    "country": ["country", "countrycode", "code", "iso3", "iso2"],
    "cluster": ["cluster", "sector", "cluster_name", "sector_name", "clustercode", "clusterid"],
    "year": ["year", "date_year", "year_start", "year_end"],
    "appeal_id": ["appeal", "appeal_id", "appeal_code", "plan_id", "activity_id"],
    "funding": ["funding", "amount", "value"]
}

print("\nJoin key candidates found:")
for key_type, search_terms in join_key_candidates.items():
    print(f"\n{key_type.upper()}:")
    for dataset_name, columns in all_columns.items():
        matching = [col for col in columns if any(term in col for term in search_terms)]
        if matching:
            print(f"  {dataset_name}: {', '.join(matching)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Detailed Key Column Analysis

# COMMAND ----------

# Analyze specific columns that are likely join keys
print("=" * 80)
print("DETAILED KEY COLUMN VALUE ANALYSIS")
print("=" * 80)

# Function to analyze a key column
def analyze_key_column(df, col_name, dataset_name):
    """Analyze a specific column for join key potential"""
    try:
        print(f"\n{dataset_name}.{col_name}:")
        
        # Count nulls
        null_count = df.filter(isnull(col(col_name))).count()
        total_rows = df.count()
        null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
        
        print(f"  Null count: {null_count} ({null_pct:.1f}%)")
        
        # Distinct values
        distinct_count = df.select(countDistinct(col(col_name))).collect()[0][0]
        print(f"  Distinct values: {distinct_count}")
        
        # Sample values
        sample_values = df.select(col(col_name)).dropna().distinct().limit(5).collect()
        sample_strs = [str(row[0]) for row in sample_values]
        print(f"  Sample values: {', '.join(sample_strs)}")
        
    except Exception as e:
        print(f"\n{dataset_name}.{col_name}: ERROR - {str(e)}")

# Analyze country codes
print("\n" + "="*80)
print("COUNTRY CODES")
country_cols = {
    "hno": ["countryCode", "country_code", "country"],
    "hrp": ["countryCode", "country_code", "country"],
    "fts_requirements": ["countryCode"],
    "fts_incoming": ["srcLocations", "destLocations"],
    "fts_internal": [],
    "fts_outgoing": [],
    "population": ["adm0_en"]
}

for dataset_name, col_names in country_cols.items():
    if dataset_name in raw_datasets:
        df = raw_datasets[dataset_name]
        for col_name in col_names:
            if col_name in df.columns:
                analyze_key_column(df, col_name, dataset_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Numeric Column Statistics

# COMMAND ----------

print("=" * 80)
print("NUMERIC COLUMN STATISTICS")
print("=" * 80)

from pyspark.sql.functions import min as spark_min, max as spark_max, avg, stddev

for dataset_name, df in raw_datasets.items():
    numeric_cols = [f.name for f in df.schema.fields if f.dataType.typeName() in ["int", "double", "long", "float"]]
    
    if numeric_cols:
        print(f"\n{dataset_name.upper()}")
        print("-" * 80)
        
        # Calculate stats for top numeric columns
        for col_name in numeric_cols[:10]:  # Limit to first 10
            try:
                stats = df.select(
                    spark_min(col(col_name)).alias("min"),
                    spark_max(col(col_name)).alias("max"),
                    avg(col(col_name)).alias("avg"),
                    stddev(col(col_name)).alias("stddev")
                ).collect()[0]
                
                print(f"  {col_name}:")
                print(f"    Min: {stats['min']}, Max: {stats['max']}, Avg: {stats['avg']:.2f}, StdDev: {stats['stddev']}")
            except:
                pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Write Raw Data to Delta Tables

# COMMAND ----------

print("=" * 80)
print("WRITING RAW DATASETS TO DELTA TABLES")
print("=" * 80)

for dataset_name, df in raw_datasets.items():
    delta_table_name = f"humanitarian.raw_{dataset_name}"
    
    try:
        # Write to Delta with overwrite mode
        df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(delta_table_name)
        
        row_count = df.count()
        print(f"✓ {delta_table_name}: {row_count:,} rows")
        
    except Exception as e:
        print(f"✗ Failed to write {delta_table_name}: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 12: Verify Delta Tables

# COMMAND ----------

# List all tables in humanitarian database
print("=" * 80)
print("TABLES IN HUMANITARIAN DATABASE")
print("=" * 80)

tables = spark.sql("SHOW TABLES IN humanitarian").collect()
for row in tables:
    table_name = row['tableName']
    print(f"✓ {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 13: Summary and Recommendations

# COMMAND ----------

print("=" * 80)
print("EDA SUMMARY")
print("=" * 80)

print("\nDATASETS LOADED:")
for dataset_name, stats in dataset_stats.items():
    print(f"  • {dataset_name}: {stats['rows']:,} rows, {stats['columns']} columns")

print("\nIDENTIFIED JOIN KEYS (LIKELY):")
print("  • Country code: countryCode (common across HNO, HRP, FTS datasets)")
print("  • Cluster/Sector: clusterCode / cluster (in FTS and HNO)")
print("  • Year: year or dateYear (in most datasets)")
print("  • Appeal/Plan ID: activity_appeal_id (in FTS datasets)")

print("\nNEXT STEPS:")
print("  1. Review EDA output above to confirm join keys")
print("  2. Proceed to Notebook 02 (02_pipeline.py) to join and feature engineer")
print("  3. Key join strategy: LEFT JOIN on HRP/HNO, joining with Funding and Population by country/cluster/year")

print("\nMISSING DATA NOTES:")
print("  • Many funding columns may have nulls - will be filled with 0 in pipeline")
print("  • Country/cluster codes should be non-null for joins")
print("  • Handle multi-valued fields (e.g., destLocations) in pipeline with explode()")

# COMMAND ----------

print("\n✓ Notebook 01 completed successfully!")
print("Ready for Notebook 02 (Pipeline & Feature Engineering)")
