#!/usr/bin/python
import os
import re


import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import *
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType
from pyspark.sql.functions import col
import pyspark.sql.functions as fn


from pyspark.mllib.util import MLUtils
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, Tokenizer, RegexTokenizer, StopWordsRemover
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors, SparseVector

sc = pyspark.SparkContext()


dataSchema = StructType([
        StructField("textID",StringType(),True),
        StructField("ID(seq)",StringType(),True),
        StructField("word",StringType(),True),
        StructField("lemma",StringType(),True),
        StructField("PoS",StringType(),True)])

spark = SparkSession.builder.getOrCreate()


#Load the data set and drop null rows
data = spark.read.option('delimiter', '\t').csv(path='hdfs:///user/camli/16-10-us.txt', schema=dataSchema)
data_all = data['textID','lemma'].na.drop()

#Group by article/news ID, make a word list
data_all_g = data_all.groupby("textID").agg(fn.collect_list("lemma"))

#Create word frequency matrix
cv = CountVectorizer(inputCol="collect_list(lemma)", outputCol="vectors")
cv_model = cv.fit(data_all_g)

#Create stopwords criterias
top20 = list(cv_model.vocabulary[0:20])
more_then_3_charachters = [word for word in cv_model.vocabulary if len(word) <= 3]
contains_digits = [word for word in cv_model.vocabulary if any(char.isdigit() for char in word)]

#Create an extended stopword list
stopwords = ['time','year','make','take','first','like','also','would','share','people','']  #Add additional stopwords in this list

default_stop = StopWordsRemover.loadDefaultStopWords('english')
#Combine the four stopwords
stopwords = stopwords + top20  + more_then_3_charachters + contains_digits + default_stop

#Remove stopwords from the tokenized list
remover = StopWordsRemover(inputCol="collect_list(lemma)", outputCol="filtered", stopWords = stopwords)
data_all_filtered = remover.transform(data_all_g)

#Create a new CountVectorizer model without the stopwords
cv = CountVectorizer(inputCol="filtered", outputCol="vectors")
cvmodel = cv.fit(data_all_filtered)
df_vect = cvmodel.transform(data_all_filtered)
"""
#Create proper input format for LDA model
parseData = df_vect.select('textID','vectors').rdd.map(lambda x: [int(x[0]), Vectors.dense(x[1])] )

#Train LDA model
ldaModel = LDA.train(parseData, k=7)


# Save and load model
ldaModel.save(sc, "LDAModel")
"""


ldaModel = LDAModel.load(sc, "LDAModel")


"""
with open ('topic_result.txt', 'w') as f:
    #Print the topics in the model
    topics = ldaModel.describeTopics(maxTermsPerTopic = 10)
    for x, topic in enumerate(topics):
        f.write('topic nr: ' + str(x)+ '\n')
        words = topic[0]
        weights = topic[1]
        for n in range(len(words)):
            f.write(cvmodel.vocabulary[words[n]] + ' ' + str(weights[n])+ '\n')
"""

#Print the topics in the model
res=[]
topics = ldaModel.describeTopics(maxTermsPerTopic = 10)
for x, topic in enumerate(topics):
    res.append('topic nr: ' + str(x))
    words = topic[0]
    weights = topic[1]
    for n in range(len(words)):
        res.append(cvmodel.vocabulary[words[n]] + ' ' + str(weights[n]))

#resSchema = StructType([
#        StructField("topic_result", StringType())])

topic_res = spark.createDataFrame(res, StringType())
topic_res.coalesce(1).write.csv("hdfs:///user/camli/topic_result.txt")


spark.stop()
