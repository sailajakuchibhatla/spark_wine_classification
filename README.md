# spark_wine_classification
**Description:** Built a wine quality prediction ML model in Spark over AWS/Docker.

### Goal:
The goal of this project was to learn how to develop parallel machine learning (ML) applications in Amazon
AWS cloud platform. I learned: <br />
(1) how to use Apache Spark to train an ML model in parallel on multiple EC2 instances; <br />
(2) how to use Spark’s MLlib to develop and use an ML model in the cloud; <br />
(3) How to use Docker to create a container for your ML model to simplify model deployment.<br />
<br />
### Files:
`LogisticRegression.java`: Uses Apache Spark and Spark MLlib to train a LinearRegressionWithLBFGS Model on 
wine data to predict wine quality. Then uses a validation dataset to evaluate the model using accuracy and F1 score.
<br />
`LogisticRegressionPrediction.java`: Takes the TestDataset.csv and LinearRegressionModel as parameters to predict the
wine quality on a single instance in Spark. Outputs the Accuracy and F1 score.
<br />
`Dockerfile`: Used to build, run and publish image in Docker Hub. This file only runs the LogisticRegressionPrediction
algorithm.
