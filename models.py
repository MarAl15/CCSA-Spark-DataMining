"""
    Data mining classification models:
        - lr/LR - Logistic Regression
        - rf/RF - Random Forest
        - gbt/GBT - Gradient-Boosted Tree
        - svm/SVM - Support Vector Machine

    RUN: /opt/spark-2.2.0/bin/spark-submit --master spark://hadoop-master:7077 --total-executor-cores 5 --executor-memory 1g models.py (<model>)
    By default: svm/SVM

    @Author: Mar Alguacil
"""
import sys

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import LinearSVC

from utils.utils import *
from utils.preprocessing import *
from utils.training_test import *


if __name__ == "__main__":
    method = ''
    if len(sys.argv) > 1:
        method = sys.argv.pop()

    sc = initializeSparkContext()
    sc.setLogLevel("ERROR") # Remove INFO/WARN messages
    sqlContext = initializeSQLContext(sc)

    column_names = ['PSSM_central_-1_Y', 'PredSA_central_-2', 'PSSM_r1_-3_T',
                    'PSSM_r1_3_N', 'PSSM_central_1_S', 'PSSM_r2_3_H']
    # Balance classes - Random Undersample (RUS)
    try:
        data = loadCSV(sqlContext, 'undersampled.training')
    except:
        try:
            data = loadCSV(sqlContext)
        except:
            data = selectCSVcolumns(sc, sqlContext,
                                    "/user/datasets/ecbdl14/ECBDL14_IR2.header",
                                    "/user/datasets/ecbdl14/ECBDL14_IR2.data",
                                    column_names)
        # Balance classes
        data = underSample(data)
        # Export data to CSV
        data.write.csv('./undersampled.training', header=True)

    # Create training and test data sets
    training_data, test_data = data.randomSplit([0.8, 0.2], seed=7)

    assembler = VectorAssembler(inputCols = column_names,
                                outputCol ='features')

    training_data = assembler.transform(training_data)
    training_data = training_data.select('label', 'features')
    test_data = assembler.transform(test_data)
    test_data = test_data.select('label', 'features')

    # Scale
    training_data, scaler = standardScale(training_data)
    test_data, _ = standardScale(test_data, scaler)

    print('Training...')

    print('\033[93m')
    # Fit the model
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    if method in ['lr', 'LR']:
        print("---------------------------------- Binomial Logistic Regression ----------------------------------")
        lr = LogisticRegression()
        paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [10, 20]). \
                                       addGrid(lr.regParam, [0, 0.3, 0.7]). \
                                       addGrid(lr.elasticNetParam, [0.2, 0.8]). \
                                       build()
        model =  trainTVS(lr, paramGrid, evaluator, training_data)
    elif method in ['rf', 'RF']:
        print("---------------------------------- Random Forest ----------------------------------")
        rf = RandomForestClassifier()
        paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [10, 20]). \
                                       addGrid(rf.maxDepth, [4, 5]). \
                                       build()
        model =  trainTVS(rf, paramGrid, evaluator, training_data)
    elif method in ['gbt', 'GBT']:
        print("---------------------------------- Gradient-Boosted Tree ----------------------------------")
        gbt = GBTClassifier()
        paramGrid = ParamGridBuilder().addGrid(gbt.maxIter, [10, 15]). \
                                       addGrid(gbt.maxDepth, [4, 5]). \
                                       build()
        model =  trainTVS(gbt, paramGrid, evaluator, training_data)
    else:
        print("---------------------------------- Linear Support Vector Machine ----------------------------------")
        svm = LinearSVC()
        paramGrid = ParamGridBuilder().addGrid(svm.maxIter, [5, 10]). \
                                       addGrid(svm.regParam, [0.01, 0.1]). \
                                       build()
        model =  trainTVS(svm, paramGrid, evaluator, training_data)

    print('\033[94m')
    prettyPrintTVS(model)

    print('\033[92m')
    print("Testing...")
    print("AUC: " + str(validate(test_data, evaluator, model)))
    print('\033[0m')





