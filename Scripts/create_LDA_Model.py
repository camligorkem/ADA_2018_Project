#!/usr/bin/python
import os
import re
import sys

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import *
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType
from pyspark.sql.functions import col, udf
import pyspark.sql.functions as fn


from pyspark.mllib.util import MLUtils
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, Tokenizer, RegexTokenizer, StopWordsRemover
from pyspark.mllib.linalg import Vectors, SparseVector
from pyspark.ml.clustering import LDA
import numpy as np

sc = pyspark.SparkContext()

dataSchema = StructType([
        StructField("ID",StringType(),True),
        StructField("ID(seq)",StringType(),True),
        StructField("word",StringType(),True),
        StructField("lemma",StringType(),True),
        StructField("PoS",StringType(),True)])

spark = SparkSession.builder.getOrCreate()

#Load the data set and drop null rows
fileName= sys.argv[1]
k =sys.argv[2]
k = int(k)
user= sys.argv[3]
sig = sys.argv[4]
sigma = float(sig)
country =sys.argv[5]

folderName = country+'_results_k'+str(k)+'_sig'+str(sig)

#Load the data set and drop null rows
data = spark.read.option('delimiter', '\t').csv(path='hdfs:///user/'+user+'/'+fileName, schema=dataSchema)
data_all = data['ID','lemma'].na.drop()

exception_list =['art','job','joy','gym','law','flu','boy','guy','gay','rob','bio','war','ill',
                    'eat','us','uk','rap','phd','man','win','die','act','ceo','cto','gen','ski','hit'
                    'eu','pay','fan','car','web','mom','act','tax','ad','fat','cop','dna','oil','sea','ban'
                    'sex','ice','son','jew','ram','pc','mba','bar','gas','god','net','aid','usa','spy'
                    'spa','fee','fun','pop','dad','fbi','cia','gdp']

#Create an extended stopword list
stopwords_extra = ['time','year','make','take','first','like','also','would','share','people','come','find','last','tell'
                'many','need','look','know','give','want','back','show','continue','without','much','more','describe','part'
                'high','according']  #Add additional stopwords in this list

#Group by article/news ID, make a word list
data_all_g = data_all.groupby("ID").agg(fn.collect_list("lemma"))
data_sample = data_all_g.sample(False, 0.1, seed=0).limit(100000)

#Create word frequency matrix
cv = CountVectorizer(inputCol="collect_list(lemma)", outputCol="features")
cv_model = cv.fit(data_sample)

#Truncate word-list from upper and lower tail
limit = int(len(cv_model.vocabulary)*((100-sigma)/200))

#Create stopwords
top30 = list(cv_model.vocabulary[0:limit])
more_then_3_charachters = [word for word in cv_model.vocabulary if len(word) <= 3 and word not in exception_list]
contains_digits = [word for word in cv_model.vocabulary if any(char.isdigit() for char in word)]
last30 = list(cv_model.vocabulary[-limit:])

default_stop = StopWordsRemover.loadDefaultStopWords('english')
#Combine the five stopwords
stopwords = stopwords_extra + top30  + more_then_3_charachters + contains_digits + default_stop + last30

#Remove stopwords from the tokenized list
remover = StopWordsRemover(inputCol="collect_list(lemma)", outputCol="filtered", stopWords = stopwords)
data_all_filtered = remover.transform(data_sample)

#Create a new CountVectorizer model without the stopwords
cv = CountVectorizer(inputCol="filtered", outputCol="features")
cvmodel = cv.fit(data_all_filtered)
df_vect = cvmodel.transform(data_all_filtered)
countVectors =  df_vect.select( "ID","features" ).cache()

# Trains a LDA model.
lda = LDA(k=k,maxIter=50, optimizer='em')
ldaModel = lda.fit(countVectors)

ll = ldaModel.logLikelihood(countVectors)
lp = ldaModel.logPerplexity(countVectors)

#Print the topics in the model
res=[]
topics = ldaModel.describeTopics(maxTermsPerTopic = 10)
res.append(fileName)
res.append('k='+str(k))
res.append("Log Likelihood=" + str(ll))
res.append("Perplexity=" + str(lp))

x = topics.collect()

for topic in (x):
    res.append('topic nr: ' + str(topic.topic))
    words = topic.termIndices
    weights = topic.termWeights
    for n in range(len(words)):
         res.append(cvmodel.vocabulary[words[n]] + ' ' + str(weights[n]))

topic_res = spark.createDataFrame(res, StringType())
topic_res.coalesce(1).write.csv("hdfs:///user/"+user+"/"+country+"/"+folderName+"/topics_"+folderName)

############################## ASSIGN ALL DOCS ###################################################
#Same for full data to do topic assignment ##
data_all_filteredx = remover.transform(data_all_g)

#Create a new CountVectorizer model without the stopwords
cv2 = CountVectorizer(inputCol="filtered", outputCol="features")
cvmodel2 = cv2.fit(data_all_filtered)
df_vect2 = cvmodel2.transform(data_all_filteredx)
countVectors2 =  df_vect2.select( "ID","features" ).cache()

getMainTopicIdx = udf(lambda l: int(np.argmax([float(x) for x in l])), IntegerType())

# assign topics to full data
transformed = ldaModel.transform(countVectors2)
assignDocs = transformed.select("ID",getMainTopicIdx("topicDistribution").alias("idxMainTopic"))
assignDocs.coalesce(1).write.csv("hdfs:///user/"+user+"/"+country+"/"+folderName+"/docs_assignment_"+folderName)

# Save model
ldaModel.save(country+"/"+folderName+"/LDAModel_"+folderName)

spark.stop()
