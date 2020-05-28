"""
    Required information extraction, preliminary analysis and data preprocessing.

    /opt/spark-2.2.0/bin/spark-submit --master spark://hadoop-master:7077 --total-executor-cores 5 --executor-memory 1g analysis.py

    @Author: Mar Alguacil
"""

import sys

from pyspark.ml.feature import VectorAssembler

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import LogisticRegression

from utils.utils import *
from utils.preprocessing import *
from utils.training_test import *


if __name__ == "__main__":
    sc = initializeSparkContext()
    sc.setLogLevel("ERROR") # Remove INFO/WARN messages
    sqlContext = initializeSQLContext(sc)

    column_names = ['PSSM_central_-1_Y', 'PredSA_central_-2', 'PSSM_r1_-3_T',
                    'PSSM_r1_3_N', 'PSSM_central_1_S', 'PSSM_r2_3_H']

    try:
        data = loadCSV(sqlContext)
    except:
        data = selectCSVcolumns(sc, sqlContext,
                                "/user/datasets/ecbdl14/ECBDL14_IR2.header",
                                "/user/datasets/ecbdl14/ECBDL14_IR2.data",
                                column_names)

    # Summary
    print('\033[93m \033[1m \033[4m SUMMARY \033[0m')
    print('\033[93m')
    data.printSchema()
    print('')
    print(data.describe().show())

    print(data.groupby('label').count().show())


    # Balance classes - Random Undersample (RUS)
    try:
        data = loadCSV(sqlContext, 'undersampled.training')
    except:
        data = underSample(data)
        # Export data to CSV
        data.write.csv('./undersampled.training', header=True)

    print('Balanced classes')
    print(data.groupby('label').count().show())
    print('\033[0m')


    # Create training and test data sets
    training_data, test_data = data.randomSplit([0.8, 0.2], seed=7)
    print('Test data')
    print(test_data.groupby('label').count().show())

    print('Training data')
    print(training_data.groupby('label').count().show())
    print(training_data.show())

    # New data set with the following columns:
    #   - 'label' - class.
    #   - 'features' - a vector containing the particular attributes.
    assembler = VectorAssembler(inputCols = column_names,
                                outputCol ='features')

    training_data = assembler.transform(training_data)
    training_data = training_data.select('label', 'features')
    # training_data = training_data.drop(*column_names)
    test_data = assembler.transform(test_data)
    test_data = test_data.select('label', 'features')
    # test_data = test_data.drop(*column_names)
    print(training_data.take(1))

    # Scale
    training_scale, _ = standardScale(training_data)
    print('\nScaled training data (Standard)')
    print(training_scale.take(1))
    # training_scale.write.csv('db/training_scale', header=True)
    # training_scale.rdd.saveAsPickleFile('db/training_scale')

    training_scale, _ = minMaxScale(training_data)
    print('\nScaled training data (Min-Max)')
    print(training_scale.take(1))

    print('\033[94m')
    # Fit the model
    print('Training...')
    lr = LogisticRegression()
    pipeline = Pipeline(stages=[lr])
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    paramGrid = ParamGridBuilder().addGrid('maxIter', [10]). \
                                   addGrid('regParam', [0.3]). \
                                   addGrid('elasticNetParam', [0.8]). \
                                   build()
    cvModelLR =  trainCV(pipeline, paramGrid, evaluator, training_data)
    prettyPrintCV(cvModelLR)

    print('\nScaled data')
    print('  - Standard')
    cvModelLR_scale =  trainCV(pipeline, paramGrid, evaluator, training_scale)
    prettyPrintCV(cvModelLR_scale)
    print('\n  - Min-Max')
    cvModelLR_scale =  trainCV(pipeline, paramGrid, evaluator, training_scale)
    prettyPrintCV(cvModelLR_scale)





