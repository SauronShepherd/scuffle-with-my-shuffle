import pytest


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

    # 1. Shared object for Global Recycling
    rep_v = vertices.repartition("id")

    # 2. First Join: Vertex A -> Edges
    # We select ONLY what we need and rename the ID immediately
    step_1 = rep_v.join(edges, rep_v["id"] == edges["src"]) \
        .select(rep_v["id"].alias("id_a"), "dst")

    # 3. Second Join: Result -> Vertex B
    # Now we join Vertex B to the 'dst' column
    # Because we renamed the first ID to 'id_a', there is no overlap
    df = step_1.join(rep_v, step_1["dst"] == rep_v["id"])

    # Write results to a no-op sink to trigger execution without output
    df.write.format("noop").mode("overwrite").save()
    df.explain()
    #print(f"Count: {df.count()}")