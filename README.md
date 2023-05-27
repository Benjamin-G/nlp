# nlp

```console
λ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Setup Neo4j in Docker Terminal

```console
$SPARK_HOME/bin/spark-shell --packages org.neo4j:neo4j-connector-apache-spark_2.12:5.0.1_for_spark_3
