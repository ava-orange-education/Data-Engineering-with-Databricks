# Customer Churn Prediction Pipeline
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark

# Initialize Spark session
spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()

# Load and prepare data
df = spark.read.csv("dbfs:/FileStore/customer_data.csv", header=True, inferSchema=True)
df = df.dropna()  # Handle missing values
df = df.repartition(8)  # Optimize partitioning

# Feature engineering
categorical_cols = ["gender", "contract_type"]
numeric_cols = ["age", "tenure", "monthly_charges"]

stages = []

for cat_col in categorical_cols:
    stringIndexer = StringIndexer(inputCol=cat_col, outputCol=cat_col + "_index")
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[cat_col + "_onehot"])
    stages += [stringIndexer, encoder]

assembler_inputs = [c + "_onehot" for c in categorical_cols] + numeric_cols
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="assembled_features")
scaler = StandardScaler(inputCol="assembled_features", outputCol="features")

stages += [assembler, scaler]

# Prepare label
label_indexer = StringIndexer(inputCol="churn", outputCol="label")
stages += [label_indexer]

# Create pipeline
rf = RandomForestClassifier(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=stages + [rf])

# Hyperparameter tuning
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 100]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol="label")

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

# Split data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# MLflow experiment tracking
mlflow.spark.autolog()

with mlflow.start_run(run_name="RF_Churn_Prediction"):
    # Fit model
    model = crossval.fit(train_data)
    
    # Make predictions
    predictions = model.transform(test_data)
    
    # Evaluate model
    auc = evaluator.evaluate(predictions)
    print(f"AUC: {auc}")
    
    # Log custom metric
    mlflow.log_metric("AUC", auc)
    
    # Save model
    best_model = model.bestModel
    mlflow.spark.log_model(best_model, "random_forest_model")

# Register model in MLflow Model Registry
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_version = client.create_model_version(
    name="churn_prediction_model",
    source=f"runs:/{mlflow.active_run().info.run_id}/random_forest_model",
    run_id=mlflow.active_run().info.run_id
)

print(f"Model version {model_version.version} created")
