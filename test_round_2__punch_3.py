import re
import urllib.request
import timeit
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from graphframes import GraphFrame
from pyspark.sql import Row

import os

# Point JAVA_HOME to your Semeru/OpenJ9 JDK (please update this to your own local path)
#os.environ["JAVA_HOME"] = os.path.expanduser(
#    "~/Library/Java/JavaVirtualMachines/semeru-17.0.9/Contents/Home"
#)

# Enable IBM OpenJ9 tracing (e.g., trace calls to TreeNode.generateTreeString)
#os.environ["JAVA_TOOL_OPTIONS"] = "-Xtrace:methods={org/apache/spark/util/collection/unsafe/sort/UnsafeExternalSorter.spill},print=mt"

# ------------------------------------------------------------------------------
# 0. Utils
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# 1. Create Spark session
# ------------------------------------------------------------------------------
spark = (
    SparkSession.builder
    .master("local[*]")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)

# Cambiar el nivel de logging a DEBUG
#spark.sparkContext.setLogLevel("DEBUG")



# ------------------------------------------------------------------------------
# 4. Run equivalent SQL-style query
# ------------------------------------------------------------------------------
"""
sql_query = (
    vertices.alias("a")
    .join(edges.alias("follows"), col("a.id") == col("follows.src"))
    .join(vertices.alias("b"), col("follows.dst") == col("b.id"))
)
sql_query.explain()
"""

# columnas del dataframe vertices
vcols = vertices.columns

# struct a con todos los campos NULL
a_struct = F.lit(None).alias("a")

# struct b con todos los campos NULL
b_struct = F.lit(None).alias("b")

# añadirlos a edges
edges_updated = edges.select("*", F.lit(None).alias("a"), F.lit(None).alias("b"))


def process_batch(n, v, e):
    """
    Procesa un batch de vertices sobre e usando mapPartitions y broadcast.

    Args:
        v (dict): Diccionario {id: Row} del batch actual.
        e (RDD): RDD de edges a procesar.

    Returns:
        RDD: e actualizado después de aplicar el batch.
    """
    spark.sparkContext.setJobDescription(f"process_batch #{num_batch}")

    # Crear broadcast del batch
    bc = spark.sparkContext.broadcast(v)

    # Aplicar mapPartitions usando la función pasada
    e_result = e.mapPartitions(lambda it: process_edges_partition(it, bc))

    # Persistimos para fijar el RDD y mantenerlo estable
    e_result = e_result.persist()

    # Necesitamos materializar el rdd antes de liberar el broadcast
    not_null_counts = e_result.map(lambda r: (
        0 if r.a is None else 1,
        0 if r.b is None else 1
    )).reduce(lambda r1, r2: (r1[0] + r2[0], r1[1] + r2[1]))

    # Liberamos memoria de broadcast y RDD antiguo
    bc.unpersist()
    e.unpersist()

    print(f"process_batch #{n}: Processed {len(vertices_batch):,} vertices. not_null_counts: {not_null_counts}")

    return e_result


def process_edges_partition(iterator, bc_map):
    m = bc_map.value

    for r in iterator:
        src_id = r.src
        a = r.a
        if not a:
            a = m.get(src_id)

        dst_id = r.dst
        b = r.b
        if not b:
            b = m.get(dst_id)

        yield Row(src=src_id, dst=dst_id, a=a, b=b)

BATCH_SIZE = 50_000
print(f"BATCH_SIZE: {BATCH_SIZE:,}")

edges_updated_rdd = edges_updated.rdd

num_batch = 1
vertices_batch = {}
for row in vertices.toLocalIterator():
    vertices_batch[row.id] = row

    if len(vertices_batch) == BATCH_SIZE:
        edges_updated_rdd = process_batch(num_batch, vertices_batch, edges_updated_rdd)
        num_batch += 1
        vertices_batch = {}

# último batch si quedó algo
if vertices_batch:
    edges_updated_rdd = process_batch(num_batch, vertices_batch, edges_updated_rdd)

sql_query = spark.createDataFrame(edges_updated_rdd)

print("Running equivalent SQL-style query ...")
elapsed = timeit.timeit(sql_query.write.format("noop").mode("overwrite").save, number=1)
print(f"SQL-style query Query took {elapsed:.2f} seconds")

# ------------------------------------------------------------------------------
# 6. Stop session
# ------------------------------------------------------------------------------
input("Press any key to stop")
spark.stop()
