#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, lower, when, from_json

# ---------- params ----------
KAFKA_BOOTSTRAP = "localhost:19092"
KAFKA_TOPIC     = "bsky_posts"       # <-- reading real posts
CHECKPOINT_DIR  = "/home/vmoralesv/k/spark/checkpoints/bsky_os"
OS_HOST         = "10.142.0.3"
OS_PORT         = "9200"
OS_INDEX        = "posts_scored"     # keep same index
# ----------------------------

spark = (SparkSession.builder
         .appName("bsky-to-opensearch")
         .getOrCreate())

spark.conf.set("spark.sql.streaming.schemaInference", True)

raw = (spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .option("failOnDataLoss", "false")
    .load())

schema = """
  id string,
  timestamp string,
  text string,
  author string,
  followers int,
  retweets int,
  language string,
  source string
"""

df = (raw.selectExpr("CAST(value AS STRING) AS json")
          .select(from_json(col("json"), schema).alias("r"))
          .select("r.*")
          .where(col("text").isNotNull()))

# ---- toy heuristic to keep index compatible ----
txt = lower(col("text"))
prob = (lit(0.15)
        + when(txt.rlike(r"(vacuna|secreto|cura|milagro|free money|click here)"), 0.5).otherwise(0)
        + when(txt.rlike(r"!{2,}"), 0.2).otherwise(0))
scored = df.withColumn("prob_fake", when(prob > 0.99, 0.99).otherwise(prob))

# ---- write to OpenSearch (using elasticsearch-hadoop) ----
query = (scored.writeStream
    .format("org.elasticsearch.spark.sql")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_DIR)
    .option("es.nodes", OS_HOST)
    .option("es.port", OS_PORT)
    .option("es.nodes.wan.only", "true")
    .start(OS_INDEX))

query.awaitTermination()
