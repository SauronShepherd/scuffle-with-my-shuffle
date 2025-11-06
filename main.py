import re
import urllib.request
import timeit
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, struct
from graphframes import GraphFrame

# ------------------------------------------------------------------------------
# 0. Utils
# ------------------------------------------------------------------------------

def download_and_read_gz_csv(url, schema):
    """
    Downloads a compressed CSV (.gz) from a URL and reads it into a PySpark DataFrame.
    """

    local_path = "/tmp/" + re.sub(r'[^\w\-]+', '_', url).replace("_gz", ".gz")
    if not Path(local_path).exists():
        print(f"Downloading from {url} ...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Download complete. Saved to {local_path}")

    print("Reading into Spark DataFrame...")
    df = spark.read \
        .option("sep", "\t") \
        .schema(schema) \
        .csv(f"file://{local_path}")

    print(f"Loaded DataFrame with {df.count():,} rows.")
    return df


# ------------------------------------------------------------------------------
# 1. Create Spark session
# ------------------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("ScuffleWithMyShuffle")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.jars.packages", "io.graphframes:graphframes-spark4_2.13:0.10.0")
    .getOrCreate()
)

# ------------------------------------------------------------------------------
# 2. Load vertices and edges
# ------------------------------------------------------------------------------
vertices_schema =  """id INT, public INT, completion_percentage INT, gender STRING, region STRING, last_login STRING, 
registration STRING, AGE STRING, body STRING, I_am_working_in_field STRING, spoken_languages STRING, hobbies STRING, 
I_most_enjoy_good_food STRING,  pets STRING, body_type STRING, my_eyesight STRING, eye_color STRING, hair_color STRING, 
hair_type STRING,  completed_level_of_education STRING, favourite_color STRING, relation_to_smoking STRING, 
relation_to_alcohol STRING, sign_in_zodiac STRING, on_pokec_i_am_looking_for STRING, love_is_for_me STRING, 
relation_to_casual_sex STRING, my_partner_should_be STRING, marital_status STRING, children STRING, 
relation_to_children STRING, I_like_movies STRING, I_like_watching_movie STRING, I_like_music STRING, 
I_mostly_like_listening_to_music STRING,  the_idea_of_good_evening STRING, I_like_specialties_from_kitchen STRING, 
fun STRING,  I_am_going_to_concerts STRING, my_active_sports STRING, my_passive_sports STRING, profession STRING, 
I_like_books STRING, life_style STRING, music STRING, cars STRING, politics STRING, relationships STRING, 
art_culture STRING, hobbies_interests STRING, science_technologies STRING, computers_internet STRING, education STRING, 
sport STRING, movies STRING, travelling STRING, health STRING, companies_brands STRING, more STRING"""
vertices = download_and_read_gz_csv("https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz", vertices_schema)

print("Vertices:")
vertices.show(truncate=False)

edges_schema = "src INT, dst INT"
edges = download_and_read_gz_csv("https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz", edges_schema)

print("Edges:")
edges.show(truncate=False)

# ------------------------------------------------------------------------------
# 3. Add 'relationship' attribute to edges and graph cretion
# ------------------------------------------------------------------------------
edges = edges.withColumn("relationship", lit("follows"))

friends = GraphFrame(vertices, edges)

# ------------------------------------------------------------------------------
# 4. Run motif query
# ------------------------------------------------------------------------------
spark.sparkContext.setJobDescription("Motif query")
motif_query = (
    friends.find("(a)-[follows]->(b)")
    .filter("follows.relationship = 'follows'")
)

print("Running motif query ...")
elapsed = timeit.timeit(lambda: motif_query.write.format("noop").mode("overwrite").save())
print(f"Motif query took {elapsed:.2f} seconds")


# ------------------------------------------------------------------------------
# 5. Run equivalent SQL-style query
# ------------------------------------------------------------------------------
spark.sparkContext.setJobDescription("SQL-style query")
sql_query = (
    friends.vertices.alias("a")
    .join(friends.edges.alias("follows"), col("a.id") == col("follows.src"))
    .join(friends.vertices.alias("b"), col("follows.dst") == col("b.id"))
    .filter("follows.relationship = 'follows'")
    .select(
        struct("a.*").alias("a"),
        struct("follows.*").alias("follows"),
        struct("b.*").alias("b")
    )
)

print("Running equivalent SQL-style query ...")
elapsed = timeit.timeit(lambda: sql_query.write.format("noop").mode("overwrite").save())
print(f"SQL-style query Query took {elapsed:.2f} seconds")

# ------------------------------------------------------------------------------
# 6. Stop session
# ------------------------------------------------------------------------------
input("Press any key to stop")
spark.stop()
