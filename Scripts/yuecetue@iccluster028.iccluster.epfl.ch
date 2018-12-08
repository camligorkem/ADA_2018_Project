#!/usr/bin/python
import os
import re
import sys

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
fileName= sys.argv[1]
k =sys.argv[2]
k = int(k)
user= sys.argv[3]


data = spark.read.option('delimiter', '\t').csv(path='hdfs:///user/'+user+'/'+fileName, schema=dataSchema)
data_all = data['textID','lemma'].na.drop()

#Group by article/news ID, make a word list
data_all_g = data_all.groupby("textID").agg(fn.collect_list("lemma"))

data_sample = data_all_g.sample(False, 0.1, seed=0).limit(100000)

#Create word frequency matrix
cv = CountVectorizer(inputCol="collect_list(lemma)", outputCol="vectors")
cv_model = cv.fit(data_sample)

exception_list =['art','job','joy','gym','law','flu','boy','guy','gay','rob','bio','war','ill',
                    'eat','us','uk','rap','phd','man','win','die','act','ceo','cto','gen','ski','hit'
                    'eu','pay','fan','car','web','mom','act','tax','ad','fat','cop','dna','oil','sea','ban'
                    'sex','ice','son','jew','ram','pc','mba','bar','gas','god','net','aid','usa','spy'
                    'spa','fee','fun','pop','dad','fbi','cia','gdp']

#Create stopwords
top30 = list(cv_model.vocabulary[0:30])
more_then_3_charachters = [word for word in cv_model.vocabulary if len(word) <= 3 and word not in exception_list]
contains_digits = [word for word in cv_model.vocabulary if any(char.isdigit() for char in word)]

#Create an extended stopword list
stopwords = ['time','year','make','take','first','like','also','would','share','people','come','find','last','tell'
            'many','need','look','know','give','want','back','show','continue','without','much','more','describe','part'
            'high','according']  #Add additional stopwords in this list

default_stop = StopWordsRemover.loadDefaultStopWords('english')
#Combine the four stopwords
stopwords = stopwords + top30  + more_then_3_charachters + contains_digits + default_stop

#Remove stopwords from the tokenized list
remover = StopWordsRemover(inputCol="collect_list(lemma)", outputCol="filtered", stopWords = stopwords)
data_all_filtered = remover.transform(data_sample)

#Create a new CountVectorizer model without the stopwords
cv = CountVectorizer(inputCol="filtered", outputCol="vectors")
cvmodel = cv.fit(data_all_filtered)
df_vect = cvmodel.transform(data_all_filtered)

#Create proper input format for LDA model
parseData = df_vect.select('textID','vectors').rdd.map(lambda x: [int(x[0]), Vectors.dense(x[1])] )

#Train LDA model
ldaModel = LDA.train(parseData, k=k)

# Save and load model
ldaModel.save(sc, "LDAModel_"+fileName+"_k"+str(k))

#ldaModel = LDAModel.load(sc, "LDAModel_"+fileName+"_k"+str(k))

#Print the topics in the model
res=[]
topics = ldaModel.describeTopics(maxTermsPerTopic = 10)
res.append(fileName)
res.append('k='+str(k))

for x, topic in enumerate(topics):
    res.append('topic nr: ' + str(x))
    words = topic[0]
    weights = topic[1]
    for n in range(len(words)):
        res.append(cvmodel.vocabulary[words[n]] + ' ' + str(weights[n]))

topic_res = spark.createDataFrame(res, StringType())
topic_res.coalesce(1).write.csv("hdfs:///user/"+user+"/topics_"+fileName+"_k"+str(k))

spark.stop()
