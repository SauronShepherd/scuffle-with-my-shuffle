import pytest
from pyspark.sql.functions import col


@pytest.mark.parametrize(
    "input_data",
    [
        # Pass custom Spark configuration to the fixture.
        # This will be applied inside the input_data() fixture via builder.config(...)
        {"spark.driver.memory": "8g"}
    ],
    indirect=True  # Tells pytest to pass this dict as request.param to the fixture
)
def test(input_data):
    """
    Simple integration test that:
    1. Loads the vertices and edges DataFrames from the fixture.
    2. Performs a self-join on the graph edges to link user A → user B.
    3. Writes the resulting DataFrame using the 'noop' output format.
       ('noop' is often used in Spark tests to avoid file I/O.)
    """

    # Unpack DataFrames provided by input_data fixture
    vertices, edges = input_data

    # Join:
    #   vertices AS a  -- source user
    #   edges AS follows  -- relationships (src → dst)
    #   vertices AS b  -- target user
    df = (
        vertices.alias("a")
                .join(edges.alias("follows"), col("a.id") == col("follows.src"))
                .join(vertices.alias("b"), col("follows.dst") == col("b.id"))
    )

    # Write results to a no-op sink to trigger execution without output
    df.write.format("noop").mode("overwrite").save()
