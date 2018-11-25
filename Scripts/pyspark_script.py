#!/usr/bin/python
import os
import re


from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import *
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType
from pyspark.sql.functions import col
import pyspark.sql.functions as fn


dataSchema = StructType([
        StructField("textID",IntegerType(),True),
        StructField("word_count",IntegerType(),True),
        StructField("date",DateType(),True),
        StructField("country",StringType(),True),
        StructField("website",StringType(),True),
        StructField("url",StringType(),True),
        StructField("title",StringType(),True)])


spark = SparkSession.builder.getOrCreate()

dataAll  = spark.read.option('delimiter', '\t').csv(path='hdfs:///user/yuecetue/now_sources_full.txt', schema=dataSchema)


filter_urls = dataAll.groupby(['country']).count().alias('articles_per_country')
filter_words = dataAll.groupby(['country']).agg(fn.sum('word_count').alias('words_per_country'))
filter_sources = dataAll.groupby(['country','website']).count().alias('sources_per_country')
filter_websites = dataAll.groupby(['country']).agg(fn.countDistinct('website').alias('websites_per_country'))

dataAll.write.mode('overwrite').parquet("hdfs:///user/yuecetue/data.parquet")
filter_urls.write.mode('overwrite').parquet("hdfs:///user/yuecetue/filter_urls.parquet")
filter_words.write.mode('overwrite').parquet("hdfs:///user/yuecetue/filter_words.parquet")
filter_sources.write.mode('overwrite').parquet("hdfs:///user/yuecetue/filter_sources.parquet")
filter_websites.write.mode('overwrite').parquet("hdfs:///user/yuecetue/filter_websites.parquet")

spark.stop()
