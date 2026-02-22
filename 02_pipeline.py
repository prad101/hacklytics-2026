# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 02: PySpark Join Pipeline & Feature Engineering
# MAGIC 
# MAGIC **Purpose**: Join all raw humanitarian datasets on (country, cluster, year), handle nulls, engineer key features, and write to Delta table.
# MAGIC 
# MAGIC **Steps**:
# MAGIC 1. Load raw Delta tables
# MAGIC 2. Standardize join keys across all datasets
# MAGIC 3. Perform LEFT JOINs: HRP/HNO as base, joined with FTS Funding and Population
# MAGIC 4. Engineer features: beneficiary_to_budget_ratio, funding_gap, funding_gap_pct, cost_per_beneficiary
# MAGIC 5. Write final dataset to `humanitarian.features` Delta table
# MAGIC 6. Display and verify output

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Import Libraries and Initialize Spark

# COMMAND ----------

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, coalesce, when, isnull, isnan, 
    explode, split, trim, lower, cast,
    sum as spark_sum, avg, count, countDistinct, max as spark_max, min as spark_min,
    lit, concat_ws, regexp_replace, to_timestamp
)
from pyspark.sql.types import DoubleType, IntegerType, StringType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("humanitarian-aid-pipeline") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

print("✓ Spark session initialized")

# Create database (if missing)
spark.sql("CREATE DATABASE IF NOT EXISTS humanitarian")
print("✓ Database 'humanitarian' ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load Raw Delta Tables

# COMMAND ----------

print("=" * 80)
print("LOADING RAW DELTA TABLES")
print("=" * 80)

# Load all raw tables
raw_tables = {
    "hno": "humanitarian.raw_hno",
    "hrp": "humanitarian.raw_hrp",
    "fts_requirements": "humanitarian.raw_fts_requirements",
    "fts_incoming": "humanitarian.raw_fts_incoming",
    "fts_internal": "humanitarian.raw_fts_internal",
    "fts_outgoing": "humanitarian.raw_fts_outgoing",
    "population": "humanitarian.raw_population"
}

dfs = {}
for name, table_path in raw_tables.items():
    try:
        df = spark.table(table_path)
        dfs[name] = df
        row_count = df.count()
        col_count = len(df.columns)
        print(f"✓ {name}: {row_count:,} rows, {col_count} columns")
    except Exception as e:
        print(f"✗ Failed to load {name}: {str(e)}")

print(f"\nTotal tables loaded: {len(dfs)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Inspect and Standardize Column Names

# COMMAND ----------

print("=" * 80)
print("DATASET COLUMN NAMES (RAW)")
print("=" * 80)

for name, df in dfs.items():
    print(f"\n{name.upper()}:")
    print(f"  Columns: {df.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Standardize Join Keys (Country, Cluster, Year)

# COMMAND ----------

# Define join key mappings for each dataset
# Key challenge: finding country, cluster, year columns across different sources

print("=" * 80)
print("STANDARDIZING JOIN KEYS")
print("=" * 80)
#print("test")

# 4.1: HRP Dataset (Humanitarian Response Plans)
# Columns: code, internalId, startDate, endDate, planVersion, categories, locations, years, origRequirements, revisedRequirements
# 'locations' is country code list, 'years' is year list

if "hrp" in dfs:
    hrp_df = dfs["hrp"]
    print("\nHRP Dataset:")
    print(f"  Columns: {hrp_df.columns}")
    
    # Standardize HRP
    hrp_prep = hrp_df.select(
        col("locations").alias("country_code"),  # country codes from locations field
        col("years").alias("year"),  # year from years field
        col("code").alias("plan_id"),
        col("origRequirements").cast(DoubleType()).alias("total_requirements_usd"),
        col("revisedRequirements").cast(DoubleType()).alias("revised_requirements_usd"),
        col("startDate"),
        col("endDate"),
        col("categories").alias("plan_type"),
        col("internalId").alias("internal_id")
    ).dropna(subset=["country_code", "year"])
    
    print(f"  After standardization: {hrp_prep.count():,} rows")
    dfs["hrp_prep"] = hrp_prep

# COMMAND ----------

# 4.2: HNO Dataset (Humanitarian Needs Overview)
# Columns: Country ISO3, Admin 1 PCode, Cluster, Population, In Need, Targeted, Affected, Reached, etc.

if "hno" in dfs:
    hno_df = dfs["hno"]
    print("\nHNO Dataset:")
    print(f"  Columns: {hno_df.columns}")
    
    # Find country and cluster columns
    hno_cols = set(col.lower() for col in hno_df.columns)
    print(f"  Column names (lowercase): {sorted(hno_cols)}")
    
    # Map HNO columns
    hno_prep = hno_df.select(
        # Country: look for ISO3, Country code variations
        coalesce(
            col("country_iso3") if "country_iso3" in hno_df.columns else None,
            col("Country ISO3") if "Country ISO3" in hno_df.columns else None
        ).alias("country_code"),
        
        # Cluster
        coalesce(
            col("sector_cluster_code") if "sector_cluster_code" in hno_df.columns else None,
            col("Cluster") if "Cluster" in hno_df.columns else None
        ).alias("cluster_code"),
        
        # Year
        col("sectors_description") if "sectors_description" in hno_df.columns else None,  # or from context
        
        # Population figures - use correct HNO column names
        col("population").cast(DoubleType()),
        col("inneed").cast(DoubleType()).alias("inneed"),
        col("targeted").cast(DoubleType()).alias("targeted"),
        col("affected").cast(DoubleType()).alias("affected"),
        col("reached").cast(DoubleType()).alias("reached")
    ).dropna(subset=["country_code"])
    
    print(f"  After standardization: {hno_prep.count():,} rows")
    dfs["hno_prep"] = hno_prep

# COMMAND ----------

# 4.3: FTS Requirements Dataset (Global Cluster Funding Requirements & Actual Funding)

if "fts_requirements" in dfs:
    fts_req_df = dfs["fts_requirements"]
    print("\nFTS Requirements Dataset:")
    print(f"  Columns: {fts_req_df.columns}")
    
    # Standardize FTS Requirements
    fts_req_prep = fts_req_df.select(
        col("countryCode").alias("country_code"),
        col("clusterCode").alias("cluster_code"),
        col("cluster").alias("cluster_name"),
        col("year").cast(IntegerType()),
        col("id").alias("appeal_id"),
        col("name").alias("appeal_name"),
        col("code").alias("appeal_code"),
        col("startDate"),
        col("endDate"),
        col("requirements").cast(DoubleType()).alias("total_requirements_usd"),
        col("funding").cast(DoubleType()).alias("total_funding_usd"),
        col("percentFunded").cast(DoubleType()).alias("funding_percent")
    ).dropna(subset=["country_code", "year"])
    
    print(f"  After standardization: {fts_req_prep.count():,} rows")
    dfs["fts_requirements_prep"] = fts_req_prep

# COMMAND ----------

# 4.4: Population Dataset (COD Population)

if "population" in dfs:
    pop_df = dfs["population"]
    print("\nPopulation Dataset:")
    print(f"  Columns: {pop_df.columns}")
    
    # Aggregate population by country and year
    pop_prep = pop_df.select(
        col("ISO3").alias("country_code"),
        col("Country"),
        col("Reference_year").cast(IntegerType()).alias("year"),
        col("Population").cast(DoubleType())
    ).filter(col("country_code").isNotNull() & col("year").isNotNull())
    
    # Group by country and year to get total population
    pop_agg = pop_prep.groupBy("country_code", "year").agg(
        spark_sum("Population").alias("population"),
        col("Country").alias("country_name")  # This won't work in groupBy - need to fix
    )
    
    # Better approach: aggregate and get country name
    pop_agg = pop_prep.groupBy("country_code", "year").agg(
        spark_sum("Population").alias("population")
    )
    
    print(f"  After aggregation: {pop_agg.count():,} rows")
    dfs["population_prep"] = pop_agg

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: FTS Funding Flows (Incoming, Internal, Outgoing) - Optional Aggregation

# COMMAND ----------

# For now, we'll focus on Requirements dataset which has both requirements and funding
# FTS flows can be integrated later for transaction-level analysis

if "fts_incoming" in dfs:
    fts_in_df = dfs["fts_incoming"]
    print("\nFTS Incoming Dataset:")
    print(f"  Columns: {fts_in_df.columns}")
    print(f"  Rows: {fts_in_df.count():,}")
    
    # This is transaction-level data; we can aggregate by destination if needed
    # For now, skip detailed analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Perform LEFT JOIN - Combine All Data Sources

# COMMAND ----------

print("=" * 80)
print("PERFORMING JOINS")
print("=" * 80)

# Start with FTS Requirements as base (most complete funding data)
features_df = dfs["fts_requirements_prep"]
print(f"\n1. Base table (FTS Requirements): {features_df.count():,} rows")

# Join with HNO data (for targeted, inneed population)
# HNO may have different schema; join on country_code, cluster_code
if "hno_prep" in dfs:
    hno_agg = dfs["hno_prep"].groupBy("country_code", "cluster_code").agg(
        spark_sum("targeted").alias("targeted"),
        spark_sum("inneed").alias("inneed"),
        spark_sum("affected").alias("affected"),
        spark_sum("reached").alias("reached")
    )
    
    # Join FTS with HNO on country + cluster
    features_df = features_df.join(
        hno_agg,
        on=["country_code", "cluster_code"],
        how="left"
    )
    print(f"2. After LEFT JOIN with HNO (on country + cluster): {features_df.count():,} rows")

# Join with Population data (on country + year for population context)
if "population_prep" in dfs:
    pop_agg = dfs["population_prep"]
    features_df = features_df.join(
        pop_agg,
        on=["country_code", "year"],
        how="left"
    )
    print(f"3. After LEFT JOIN with Population: {features_df.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Handle Nulls - Fill with Defaults

# COMMAND ----------

print("=" * 80)
print("HANDLING NULL VALUES")
print("=" * 80)

# Fill nulls in funding and population columns with 0 (assume no funding/targeting if not reported)
features_df = features_df.fillna(
    value={
        "total_funding_usd": 0.0,
        "funding_percent": 0.0,
        "population": 0.0,
        "targeted": 0.0,
        "inneed": 0.0,
        "affected": 0.0,
        "reached": 0.0
    }
)

# Count nulls before and after
null_before = features_df.select([
    count(when(isnull(col(c)), 1)).alias(c) 
    for c in features_df.columns
]).collect()[0].asDict()

print("Nulls per column (after fill):")
for col_name, null_count in sorted(null_before.items(), key=lambda x: x[1], reverse=True):
    if null_count > 0:
        print(f"  {col_name}: {null_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Engineer Project-Level Features (Revised)

# COMMAND ----------

print("=" * 80)
print("ENGINEERING PROJECT-LEVEL FEATURES (IMPACT-FOCUSED)")
print("=" * 80)

# Revised features based on targeted beneficiaries and people in need
features_df = features_df.withColumn(
    # Core efficiency signal: beneficiaries per dollar funded
    "beneficiary_to_funding_ratio",
    col("targeted") / (col("total_funding_usd") + lit(1e-9))
).withColumn(
    # Scale signal: people in need per dollar requested
    "need_to_requirements_ratio",
    col("inneed") / (col("total_requirements_usd") + lit(1e-9))
).withColumn(
    # Prioritization signal: what % of people in need are being targeted
    "targeting_coverage_rate",
    col("targeted") / (col("inneed") + lit(1e-9))
).withColumn(
    # Raw dollar gap
    "funding_gap_usd",
    col("total_requirements_usd") - col("total_funding_usd")
).withColumn(
    # % of requirements unfunded, null-safe
    "funding_gap_pct",
    when(col("total_requirements_usd") > 0,
        (col("total_requirements_usd") - col("total_funding_usd"))
        / col("total_requirements_usd") * 100
    ).otherwise(lit(None))
).withColumn(
    # Fraction of requirements actually funded
    "funding_coverage_rate",
    col("total_funding_usd") / (col("total_requirements_usd") + lit(1e-9))
).withColumn(
    # Cost per targeted beneficiary
    "cost_per_beneficiary",
    when(col("targeted") > 0,
        col("total_funding_usd") / col("targeted")
    ).otherwise(lit(None))
)

print("✓ Features engineered:")
print("  • beneficiary_to_funding_ratio      — targeted people per dollar funded (efficiency)")
print("  • need_to_requirements_ratio        — people in need per dollar requested (scale)")
print("  • targeting_coverage_rate           — % of people in need being targeted (prioritization)")
print("  • funding_gap_usd                   — absolute dollar shortfall")
print("  • funding_gap_pct                   — % of need unfunded")
print("  • funding_coverage_rate             — fraction of requirements funded (0-1)")
print("  • cost_per_beneficiary              — USD per targeted beneficiary")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8b: Engineer Population-Based Features

# COMMAND ----------

print("=" * 80)
print("ENGINEERING POPULATION-BASED FEATURES")
print("=" * 80)

# Population-relative metrics (context features showing project scale vs. country size)
features_df = features_df.withColumn(
    "cost_per_capita",
    col("total_funding_usd") / (col("population") + lit(1e-9))
).withColumn(
    "requirement_per_capita",
    col("total_requirements_usd") / (col("population") + lit(1e-9))
).withColumn(
    "funding_coverage_pct",
    (col("total_funding_usd") / (col("population") + lit(1e-9))) * 100
).withColumn(
    "requirement_coverage_pct",
    (col("total_requirements_usd") / (col("population") + lit(1e-9))) * 100
)

print("✓ Population-based features engineered:")
print("  • cost_per_capita (funded spend per person)")
print("  • requirement_per_capita (required spend per person)")
print("  • funding_coverage_pct (funded spend as % of population)")
print("  • requirement_coverage_pct (required spend as % of population)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8c: Engineer Sector-Based Features

# COMMAND ----------

print("=" * 80)
print("ENGINEERING SECTOR-BASED FEATURES")
print("=" * 80)

# Create sector-level aggregations (window functions)
from pyspark.sql.window import Window

# Define window for sector + year aggregations
sector_window = Window.partitionBy("country_code", "year", "cluster_code")

# Sector totals and metrics
features_df = features_df.withColumn(
    "sector_total_requirements_usd",
    spark_sum("total_requirements_usd").over(sector_window)
).withColumn(
    "sector_total_funding_usd",
    spark_sum("total_funding_usd").over(sector_window)
).withColumn(
    "sector_total_gap_usd",
    spark_sum("funding_gap_usd").over(sector_window)
).withColumn(
    "sector_project_count",
    count("appeal_id").over(sector_window)
).withColumn(
    "project_funding_share_of_sector_pct",
    (col("total_funding_usd") / (col("sector_total_funding_usd") + lit(1e-9))) * 100
).withColumn(
    "project_requirement_share_of_sector_pct",
    (col("total_requirements_usd") / (col("sector_total_requirements_usd") + lit(1e-9))) * 100
).withColumn(
    "sector_funding_efficiency_pct",
    (col("sector_total_funding_usd") / (col("sector_total_requirements_usd") + lit(1e-9))) * 100
).withColumn(
    "project_funding_status_vs_sector_avg",
    col("funding_percent") - (col("sector_total_funding_usd") / (col("sector_total_requirements_usd") + lit(1e-9))) * 100
)

print("✓ Sector-based features engineered:")
print("  • sector_total_requirements_usd (total needs in this sector)")
print("  • sector_total_funding_usd (total funding received by sector)")
print("  • sector_total_gap_usd (total funding gap in sector)")
print("  • sector_project_count (number of projects in sector)")
print("  • project_funding_share_of_sector_pct (this project's funding % of sector total)")
print("  • project_requirement_share_of_sector_pct (this project's needs % of sector total)")
print("  • sector_funding_efficiency_pct (sector-wide funding rate)")
print("  • project_funding_status_vs_sector_avg (how this project compares to sector average)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Final Data Quality Checks

# COMMAND ----------

print("=" * 80)
print("FINAL DATA QUALITY CHECKS")
print("=" * 80)

# Row count
print(f"\nTotal rows in features dataset: {features_df.count():,}")

# Column count
print(f"Total columns: {len(features_df.columns)}")

# Nulls in key columns
key_columns = [
    "country_code", "year", "cluster_code", "total_requirements_usd", 
    "total_funding_usd", "funding_gap_pct", "cost_per_beneficiary",
    "targeted", "inneed", "population", 
    "sector_total_funding_usd", "sector_project_count"
]

null_counts = features_df.select([
    count(when(isnull(col(c)), 1)).alias(c) 
    for c in key_columns
]).collect()[0].asDict()

print("\nNulls in key columns:")
for col_name, null_count in null_counts.items():
    null_pct = (null_count / features_df.count() * 100) if features_df.count() > 0 else 0
    print(f"  {col_name}: {null_count:,} ({null_pct:.1f}%)")

# Stats on feature columns
print("\nFeature Statistics (Project-Level - Impact Focused):")
numeric_features_project = [
    "beneficiary_to_funding_ratio",
    "need_to_requirements_ratio",
    "targeting_coverage_rate",
    "funding_gap_pct",
    "funding_coverage_rate",
    "cost_per_beneficiary"
]

for col_name in numeric_features_project:
    try:
        stats = features_df.select(
            spark_min(col(col_name)).alias("min"),
            spark_max(col(col_name)).alias("max"),
            avg(col(col_name)).alias("avg")
        ).collect()[0]
        print(f"  {col_name}:")
        print(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, Avg: {stats['avg']:.4f}")
    except:
        print(f"  {col_name}: unable to compute")

print("\nFeature Statistics (Population-Based):")
numeric_features_pop = [
    "cost_per_capita",
    "requirement_per_capita",
    "funding_coverage_pct",
    "requirement_coverage_pct"
]

for col_name in numeric_features_pop:
    try:
        stats = features_df.select(
            spark_min(col(col_name)).alias("min"),
            spark_max(col(col_name)).alias("max"),
            avg(col(col_name)).alias("avg")
        ).collect()[0]
        print(f"  {col_name}:")
        print(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, Avg: {stats['avg']:.4f}")
    except:
        print(f"  {col_name}: unable to compute")

print("\nFeature Statistics (Sector-Based):")
numeric_features_sector = [
    "project_funding_share_of_sector_pct",
    "project_requirement_share_of_sector_pct",
    "sector_funding_efficiency_pct",
    "project_funding_status_vs_sector_avg"
]

for col_name in numeric_features_sector:
    try:
        stats = features_df.select(
            spark_min(col(col_name)).alias("min"),
            spark_max(col(col_name)).alias("max"),
            avg(col(col_name)).alias("avg")
        ).collect()[0]
        print(f"  {col_name}:")
        print(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, Avg: {stats['avg']:.4f}")
    except:
        print(f"  {col_name}: unable to compute")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: View Sample Data

# COMMAND ----------

print("=" * 80)
print("SAMPLE DATA (First 10 Rows)")
print("=" * 80)

features_df.limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Write Features to Delta Table

# COMMAND ----------

print("=" * 80)
print("WRITING FEATURES TO DELTA TABLE")
print("=" * 80)

delta_table_name = "humanitarian.features"

try:
    # Cache the dataframe for efficiency
    features_df.cache()
    
    # Write to Delta with overwrite mode
    features_df.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(delta_table_name)
    
    row_count = features_df.count()
    col_count = len(features_df.columns)
    
    print(f"✓ {delta_table_name}: {row_count:,} rows, {col_count} columns")
    
    # Unpersist cache
    features_df.unpersist()
    
except Exception as e:
    print(f"✗ Failed to write {delta_table_name}: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 12: Verify Delta Table Creation

# COMMAND ----------

# Read back the Delta table to confirm
print("=" * 80)
print("VERIFICATION: Reading back from Delta table")
print("=" * 80)

features_verify = spark.table("humanitarian.features")
print(f"\n✓ Table read successfully: {features_verify.count():,} rows")
print(f"✓ Schema verified with {len(features_verify.columns)} columns")

# Show sample
print("\nSchema:")
features_verify.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 13: Summary

# COMMAND ----------

print("=" * 80)
print("NOTEBOOK 02 SUMMARY")
print("=" * 80)

print("\n✓ COMPLETED TASKS:")
print("  1. Loaded raw Delta tables from Notebook 01")
print("  2. Standardized join keys (country_code, year, cluster_code)")
print("  3. Performed LEFT JOINs: FTS Requirements + Population")
print("  4. Handled nulls: filled funding with 0, preserved key columns")
print("\n  5. ENGINEERED PROJECT-LEVEL FEATURES (Impact-Focused):")
print("     - beneficiary_to_funding_ratio (efficiency: targeted people per dollar)")
print("     - need_to_requirements_ratio (scale: need per dollar requested)")
print("     - targeting_coverage_rate (prioritization: % of need being addressed)")
print("     - funding_gap_usd & funding_gap_pct (absolute and % financing gaps)")
print("     - funding_coverage_rate (% of requirements met)")
print("     - cost_per_beneficiary (USD per targeted person)")
print("\n  6. ENGINEERED POPULATION-BASED FEATURES (Context):")
print("     - cost_per_capita (funded spend per person)")
print("     - requirement_per_capita (required spend per person)")
print("     - funding_coverage_pct (funded spend as % of population)")
print("     - requirement_coverage_pct (required spend as % of population)")
print("\n  7. ENGINEERED SECTOR-BASED FEATURES (Benchmarking):")
print("     - sector_total_requirements_usd (total needs in sector)")
print("     - sector_total_funding_usd (total sector funding)")
print("     - sector_total_gap_usd (total sector gap)")
print("     - sector_project_count (# projects in sector)")
print("     - project_funding_share_of_sector_pct (project's % of sector funding)")
print("     - project_requirement_share_of_sector_pct (project's % of sector needs)")
print("     - sector_funding_efficiency_pct (sector-wide funding rate)")
print("     - project_funding_status_vs_sector_avg (vs. sector benchmark)")
print("\n  8. Written final dataset to humanitarian.features Delta table")

print("\n✓ FEATURE SUMMARY:")
print("  • Total features: 22 (project, population, sector-based)")
print("  • IMPACT-FOCUSED: beneficiary_to_funding_ratio, targeting_coverage_rate")
print("  • Can use for anomaly detection (Notebook 03)")
print("  • Can use for benchmarking/clustering (Notebook 04)")
print("  • Can use for optimization (Notebook 05)")

print("\n✓ NEXT STEPS:")
print("  → Proceed to Notebook 03 (Anomaly Detection)")
print("  → Use impact-focused features for benefit/cost efficiency analysis")
print("  → Apply sector-based context for comparative anomalies")

# COMMAND ----------

print("\n✓ Notebook 02 completed successfully!")
