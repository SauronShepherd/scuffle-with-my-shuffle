import pytest
from pyspark.sql.functions import col


@pytest.mark.parametrize(
    "input_data",
    [
        # Pass custom Spark configuration to the fixture.
        # This will be applied inside the input_data() fixture via builder.config(...)
        {"spark.driver.memory": "8g", "spark.sql.analyzer.failAmbiguousSelfJoin": "false"}
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

    # 1. Manually trigger a shuffle on 'id' to create a single reusable Exchange stage
    # Spark's AQE should ideally store this in its stageCache for recycling.
    rep_vertices = vertices.repartition("id")

    cond = rep_vertices["id"] == col("src")

    df = (
        rep_vertices.alias("a").join(edges, cond)
                    .drop("src")
                    .withColumnRenamed("dst", "src")
                    .join(rep_vertices.alias("b"), cond)
                    .drop("src")
    )

    # Write results to a no-op sink to trigger execution without output
    df.write.format("noop").mode("overwrite").save()
    df.explain()
    #print(f"Count: {df.count()}")