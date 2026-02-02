import pytest
import time
import pandas as pd
import plotly.graph_objects as go
from pyspark.sql.functions import col, expr, rand, sha2
from py4j.java_gateway import java_import


def test(spark):
    # ThreadMXBean for counting threads
    java_import(spark._jvm, "java.lang.management.ManagementFactory")
    thread_mx = spark._jvm.ManagementFactory.getThreadMXBean()

    # Streaming source: generates rows continuously
    df = spark.readStream.format("rate").option("rowsPerSecond", 50_000).load()

    # Transformations
    df2 = df.withColumn("key", expr("value % 5000000")) \
        .withColumn("noise", sha2(rand().cast("string"), 512)) \
        .withColumn("junk", expr("explode(array_repeat(noise, 3))"))

    # Aggregate to trigger shuffle
    df3 = df2.groupBy("key").count()

    # Write to noop sink
    query = df3.writeStream.format("noop").outputMode("complete").start()

    # List to store data for plotting
    plot_data = []

    start_time = time.time()
    print("Monitoring threads for 60 seconds...")

    while time.time() - start_time < 60:
        elapsed = int(time.time() - start_time)
        all_threads = thread_mx.getThreadInfo(thread_mx.getAllThreadIds(), 0)

        counts = {
            "Time": elapsed,
            "ChecksumCheckpointFileManager": 0,
            "block-manager-ask": 0,
            "block-manager-storage": 0,
            "shuffle-exchange": 0,
            "other": 0
        }

        for t in all_threads:
            if t is None: continue
            name = t.getThreadName()
            matched = False

            for key in ["ChecksumCheckpointFileManager", "block-manager-ask", "block-manager-storage",
                        "shuffle-exchange"]:
                if key in name:
                    counts[key] += 1
                    matched = True
                    break

            if not matched:
                counts["other"] += 1

        plot_data.append(counts)
        time.sleep(5)

    query.stop()
    print("Streaming stress test finished. Generating plot...")

    # --- Plotly Generation ---
    df_plot = pd.DataFrame(plot_data)

    fig = go.Figure()

    # Define the categories to stack (excluding 'Time')
    categories = ["shuffle-exchange", "block-manager-storage", "block-manager-ask", "ChecksumCheckpointFileManager",
                  "other"]

    for cat in categories:
        fig.add_trace(go.Scatter(
            x=df_plot["Time"],
            y=df_plot[cat],
            mode='lines',
            name=cat,
            stackgroup='one',  # This is what creates the "mounting" / stacked effect
            groupnorm=None  # Use 'percent' here if you want to see relative share
        ))

    fig.update_layout(
        title="Spark JVM Thread Pool Allocation Over Time",
        xaxis_title="Seconds Elapsed",
        yaxis_title="Total Thread Count",
        hovermode="x unified",
        template="plotly_white"
    )

    fig.show()