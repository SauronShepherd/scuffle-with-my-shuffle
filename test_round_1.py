import pytest

from pyspark.sql.functions import col


@pytest.mark.parametrize(
    "input_data",
    [
        {"spark.driver.memory": "8g"}
    ],
    indirect=True
)
def test(input_data):
    vertices, edges = input_data
    df = (
        vertices.alias("a")
                .join(edges.alias("follows"), col("a.id") == col("follows.src"))
                .join(vertices.alias("b"), col("follows.dst") == col("b.id"))
    )
    df.write.format("noop").mode("overwrite").save()
