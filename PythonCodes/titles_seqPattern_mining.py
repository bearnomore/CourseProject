from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pyspark.ml.fpm import FPGrowth, PrefixSpan
from pyspark import SparkContext, Row, SparkConf, sql
import os
import csv


def create_sequence(line):
    arr = line.strip().split(' ')
    result = []
    for i in arr:
        result.append([i])
    return Row(sequence=result)


os.environ['HADOOP_HOME'] = 'E:/Spark/spark-3.0.1-bin-hadoop2.7/'

path = 'E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/'

conf = SparkConf().setAppName("fpm").setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = sql.SQLContext(sc)
sc.setLogLevel("DEBUG")

data = sc.textFile(path + "DBLP2000_preprocessed_titles.txt")
transactions = data.map(lambda line: create_sequence(line)).collect()
df = sc.parallelize(transactions).cache().toDF()
prefixSpan = PrefixSpan()
prefixSpan.setMinSupport(0.000999)
prefixSpan.setMaxPatternLength(20)
result = prefixSpan.findFrequentSequentialPatterns(df).rdd.coalesce(1).saveAsTextFile(path+"output")
