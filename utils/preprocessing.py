from py4j.protocol import Py4JJavaError
import pyspark.sql.functions as f

from pyspark.ml.feature import Normalizer, StandardScaler, MinMaxScaler


def underSample(data):
    """
        Randomly subset all the classes in the set so that their class frequencies match
        the least prevalent class (the class with label 1 in this case).
    """
    c0 = data.filter(f.col('label')==0).count()
    c1 = data.filter(f.col('label')==1).count()*1.0
    return data.sampleBy('label', fractions={0: c1/c0, 1: 1.0}, seed=7).cache()

def standardScale(data, scalerModel=None):
    """
        Standardizes features by removing the mean and scaling to unit variance.
    """
    if scalerModel == None:
        scaler = StandardScaler(inputCol='features',
                                outputCol='scalFeatures',
                                withStd=True, withMean=False)

        # Compute summary statistics by fitting the StandardScaler
        scalerModel = scaler.fit(data)

    # Normalize each feature to have unit standard deviation.
    scaled_data = scalerModel.transform(data)
    scaled_data = scaled_data.drop('features')
    scaled_data = scaled_data.withColumnRenamed('scalFeatures', 'features')
    return scaled_data, scalerModel

def minMaxScale(data, scalerModel=None):
    """
        Rescales each feature individually to a common range [0, 1].
    """
    if scalerModel == None:
        scaler = MinMaxScaler(inputCol='features',
                              outputCol='scalFeatures')

        # Compute summary statistics and generate MinMaxScalerModel
        scalerModel = scaler.fit(data)

    # Rescale each feature to range [min, max].
    scaled_data = scalerModel.transform(data)
    scaled_data = scaled_data.drop('features')
    scaled_data = scaled_data.withColumnRenamed('scalFeatures', 'features')
    return scaled_data, scalerModel



