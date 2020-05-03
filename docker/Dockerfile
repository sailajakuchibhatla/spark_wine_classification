FROM bde2020/spark-submit:2.4.5-hadoop2.7

COPY . /

RUN apk add --no-cache openjdk8 maven

EXPOSE 80

CMD ["java","-jar","LogisticRegressionPrediction.jar","/ValidationDataset.csv","/LogisticRegressionModel"]