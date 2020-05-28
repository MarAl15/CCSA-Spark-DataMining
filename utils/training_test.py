from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit


def trainCV(estimator, paramGrid, evaluator, data, k=3):
    """
        k-fold cross-validation.
    """
    cv = CrossValidator(estimator=estimator,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=k,
                        seed=7)
    return cv.fit(data) # model

def trainTVS(estimator, paramGrid, evaluator, data):
    """
        Train validation split.
    """
    tvs = TrainValidationSplit(estimator=estimator,
                               estimatorParamMaps=paramGrid,
                               evaluator=evaluator,
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8,
                               seed=7)
    return tvs.fit(data) # model

def prettyPrintCV(model):
    for combination, auc in zip(model.getEstimatorParamMaps(), model.avgMetrics):
        print((combination, "AUC: " + str(auc)))

def prettyPrintTVS(model):
    for combination, auc in zip(model.getEstimatorParamMaps(), model.validationMetrics):
        d = {} #Empty dictionary to add values into
        for key in combination.keys():
            d[key.name] = combination[key]
        print((d, "AUC: " + str(auc)))


def validate(data, evaluator, cvModel):
    return evaluator.evaluate(cvModel.transform(data))





