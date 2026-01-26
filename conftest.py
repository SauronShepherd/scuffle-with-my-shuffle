import re
import urllib.request
import ssl
from pathlib import Path
import pytest

from pyspark.sql import SparkSession, DataFrame


def download_and_read_gz_csv(name: str, url: str, schema: str):
    """
    Downloads a compressed CSV (.gz) from a URL and reads it into a PySpark DataFrame.
    """
    # Temporarily disable SSL certificate verification to allow downloading files from HTTPS sources
    # without certificate errors (useful for testing; not recommended for production)
    ssl._create_default_https_context = ssl._create_unverified_context

    local_path = re.sub(r'[^\w\-]+', '_', url).replace("_gz", ".gz")
    if not Path(local_path).exists():
        print(f"Downloading from {url} ...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Download complete. Saved to {local_path}")

    spark = SparkSession.getActiveSession()
    df = spark.read.option("sep", "\t").schema(schema).csv(local_path)
    df.createOrReplaceTempView(name)

    count = df.count()

    partitions = spark.sparkContext.getConf().get(name + ".partitions")
    if partitions:
        df = df.repartition(int(partitions))
    else:
        partitions = df.rdd.getNumPartitions()

    print(f"Read '{name}' with {len(df.schema.fields):,} columns and {count:,} rows into {partitions} partitions.")
    return df


def get_vertices():
    schema = """id INT, public INT, completion_percentage INT, gender STRING, region STRING, last_login STRING, 
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
    return download_and_read_gz_csv(
        "vertices",
        "https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz",
        schema
    )


def get_edges():
    schema = "src INT, dst INT"
    return download_and_read_gz_csv(
        "edges",
        "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz",
        schema
    )



# -------- Spark fixture --------

@pytest.fixture
def input_data(request):

    config = getattr(request, "param", {})

    builder = SparkSession.builder.appName(request.node.name)
    # Aplicar configuración personalizada
    for key, value in config.items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()

    # Set log level to DEBUG
    spark.sparkContext.setLogLevel("DEBUG")

    vertices = get_vertices()
    edges = get_edges()

    yield vertices, edges
    print("Press a key to continue...")
    input()
    spark.stop()
