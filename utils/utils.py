from pyspark import sql, SparkContext, SparkConf


def initializeSparkContext():
    """
        Creates and returns a Spark context with Spark configuration.
    """
    conf = SparkConf().setAppName("Practical assignment 4 - Mar Alguacil")

    return SparkContext(conf=conf)

def initializeSQLContext(sc):
    """
        Creates and returns a SQL context from the Spark context.
    """
    return sql.SQLContext(sc)

def selectCSVcolumns(sc, sqlc, header_file, data_file, column_names):
    """
        Extracts particular columns from a CSV file in a Spark and SQL Context,
        creates 'filteredC.small.training', a CSV file with the selected columns, and
        returns the new dataset.
    """
    # Get column names
    headers = sc.textFile(header_file).filter(lambda line: '@inputs' in line).map(lambda line: line.split(", ")).collect()[0]

    # Remove '@inputs' from the first item
    headers[0] = headers[0].replace('@inputs ', '')

    # Find the position of the required columns
    indices = map(lambda col: headers.index(col), column_names)

    # Load CSV file
    data = sqlc.read.csv(data_file, header=False, sep=",", inferSchema=True)

    # Extract specific selected columns
    data = data.select(data.columns[indices[0]], # 'PSSM_central_-1_Y'
                       data.columns[indices[1]], # 'PredSA_central_-2'
                       data.columns[indices[2]], # 'PSSM_r1_-3_T'
                       data.columns[indices[3]], # 'PSSM_r1_3_N'
                       data.columns[indices[4]], # 'PSSM_central_1_S'
                       data.columns[indices[5]], # 'PSSM_r2_3_H'
                       data.columns[len(headers)] # 'label'
                      )

    # Set column names
    column_names.append('label')
    for old_col, new_col in zip(data.columns, column_names):
        data = data.withColumnRenamed(old_col, new_col)

    # Export data to CSV
    data.write.csv('./filteredC.small.training', header=True)

    return data

def loadCSV(sqlc, csv_file='filteredC.small.training'):
    """
        Read csv_file.
    """
    return sqlc.read.csv(csv_file, header=True, inferSchema=True)


