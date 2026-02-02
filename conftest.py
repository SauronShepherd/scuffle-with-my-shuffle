import re
import urllib.request
import ssl
from pathlib import Path
import pytest

from pyspark.sql import SparkSession, DataFrame


def download_and_read_gz_csv(name: str, url: str, schema: str):
    """
    Downloads a compressed CSV (.gz) from a URL and loads it into a PySpark DataFrame.

    Parameters
    ----------
    name : str
        Temporary table name the DataFrame will be registered as.
    url : str
        HTTP(S) URL pointing to a .gz-compressed CSV.
    schema : str
        Spark schema string describing the file layout.

    Returns
    -------
    DataFrame
        PySpark DataFrame containing the downloaded data.
    """

    # Disable SSL certificate verification to avoid failures on HTTPS sources
    # (convenient for tests; should NOT be used in production).
    ssl._create_default_https_context = ssl._create_unverified_context

    # Create a local file path based on the URL (safe for filesystem)
    # Replace non-alphanumeric characters with underscores.
    local_path = re.sub(r'[^\w\-]+', '_', url).replace("_gz", ".gz")

    # Download file only if not present locally
    if not Path(local_path).exists():
        print(f"Downloading from {url} ...")
        urllib.request.urlretrieve(url, local_path)
        print(f"Download complete. Saved to {local_path}")

    # Get active Spark session (must already exist)
    spark = SparkSession.getActiveSession()

    # Load the .gz CSV using tab separation and the provided schema
    df = spark.read.option("sep", "\t").schema(schema).csv(local_path)

    # Register as a temporary SQL table
    df.createOrReplaceTempView(name)

    # Trigger read and count rows
    count = df.count()

    # Check for a custom number of partitions defined in Spark config
    partitions = spark.sparkContext.getConf().get(name + ".partitions")
    if partitions:
        df = df.repartition(int(partitions))
    else:
        partitions = df.rdd.getNumPartitions()

    print(
        f"Read '{name}' with {len(df.schema.fields):,} columns "
        f"and {count:,} rows into {partitions} partitions."
    )

    return df


def get_vertices():
    """
    Loads the Pokec user profile table.

    Returns
    -------
    DataFrame
        DataFrame containing the vertices (user profiles).
    """

    # Schema definition for the vertices dataset
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
    """
    Loads the Pokec edge list (user relationships).

    Returns
    -------
    DataFrame
        DataFrame containing the edges between users.
    """

    schema = "src INT, dst INT"

    return download_and_read_gz_csv(
        "edges",
        "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz",
        schema
    )


# -------------------------------------------------------------------
# Pytest input_data fixture
# -------------------------------------------------------------------

@pytest.fixture
def input_data(request):
    """
    Creates a Spark session for the test, applies optional test-specific
    configuration, loads the vertices and edges datasets, and yields them.


    Parameters
    ----------
    request : pytest.FixtureRequest
        Allows parametrization of the fixture. Any dictionary passed via
        @pytest.mark.parametrize(..., indirect=True) will be interpreted
        as Spark configuration options.

    Yields
    ------
    (DataFrame, DataFrame)
        Tuple containing (vertices_df, edges_df).

    Notes
    -----
    - The Spark session is stopped after the test completes.
    - Pauses execution waiting for user input after the test run.
    """

    # Retrieve configuration passed via parametrize(..., indirect=True)
    config = getattr(request, "param", {})

    # Build a Spark session with an application name derived from the test name
    builder = SparkSession.builder

    # Apply any custom Spark configuration options
    for key, value in config.items():
        builder = builder.config(key, value)

    # Create or reuse existing Spark session
    spark = builder.getOrCreate()

    # Increase log verbosity to DEBUG for detailed trace output
    #spark.sparkContext.setLogLevel("DEBUG")

    # Load datasets
    vertices = get_vertices()
    edges = get_edges()

    # Yield the data for the test
    yield vertices, edges

    # Pause to allow inspection before Spark shutdown
    print("Press a key to continue...")
    input()

    # Cleanly stop Spark after the test
    spark.stop()


# -------------------------------------------------------------------
# Pytest spark fixture
# -------------------------------------------------------------------

@pytest.fixture
def spark(request):
    """
    Creates a Spark session for the test, applies optional test-specific
    configuration, loads the vertices and edges datasets, and yields them.


    Parameters
    ----------
    request : pytest.FixtureRequest
        Allows parametrization of the fixture. Any dictionary passed via
        @pytest.mark.parametrize(..., indirect=True) will be interpreted
        as Spark configuration options.

    Yields
    ------
    SparkSession

    Notes
    -----
    - The Spark session is stopped after the test completes.
    - Pauses execution waiting for user input after the test run.
    """

    # Create or reuse existing Spark session
    spark = SparkSession.builder.getOrCreate()

    # Yield the data for the test
    yield spark

    # Pause to allow inspection before Spark shutdown
    print("Press a key to continue...")
    input()

    # Cleanly stop Spark after the test
    spark.stop()
