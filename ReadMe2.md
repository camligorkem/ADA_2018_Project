# Country Overviews Through Online Newspapers and Magazines

# Abstract

> “A good newspaper is a nation talking to itself.” \
>  Arthur Miller

In the newspaper and magazine articles we cover more than events, we share what we are doing, what we are interested in and what matters to us as a society. Our aim in this project is to see how the news content changes for countries over time and is there a connection between country profiles and the news published. Our motivation behind this project is to understand, the underlying profile of the published news content: the topics of the news, its tone, most commonly used words and whether these information change between countries or time. Do the news become more global or they manage to conserve some trends from the country they are published? We aim to see if country specific information and published news are somehow correlated or a good newspaper/article is now a **_world_** talking to itself? We will use the News on the Web dataset more specifically, Now corpus data and we will gather the country profiles from The World Factbook data [[1]](https://www.cia.gov/library/publications/the-world-factbook/).     


# Research questions
## Contents of the News
- What are the main topics of the published news? (tech, politics, sports, etc.)
- What are the distributions of these topics over country and time?
XXXXTODOXXXXX- What are some mostly used words in the countries/topic? 

## News and Country Profiles
- Is there some trends or patterns between news articles and country profiles?
  - Are countries having X attribute/fact, have more news on topic Y?
XXXXTODOXXXX  - Are countries having X attribute/fact, use Z common specific words more often?
XXXXTODOXXXX  - If there is some patterns or correlation, do they change over time?

# Dataset
## News on the Web: Now Corpus

Now Corpus has online magazine and newspaper text data from 20 different English speaking countries collected each day. The dataset we have in the cluster is for 6 years from 2010-2016. It has lexicon, source, text and wlp (word, lemma, PoS tag) data.  
The data is in txt format but th data size is too large, it is around 5.9 million words, in terms of computation time we might need to limit and use part of the dataset, we can do this either by choosing some specific countries or specific time frames. We might limit the time by just doing the analysis for 1 year, or some selected months for each year or some selected days for each month. 

We can enrich the dataset by doing following:
- By using the source data we can group the news by country and time. 
- By using the title and url information of the news we can infer the topic of the newspaper (most cases url contains sports, healthcare, tech words in the urls).
- By using the wlp data we can find the frequency of the words in the text.

## World Factbook

World Factbook contains the daa about each country profiles collected by CIA. The data is open source and can be obtained from [this link](https://github.com/factbook/factbook.json) as a json format. We are only interested in 20 countries therefore, we will use only the countries in which we have news article available. Also, the country profiles are in-depth, we will choose key information such as population, age profile, sex ratio, and other socio-economic informations. 

# Data Analysis
We did two data exploration for each dataset separately.

## FactBook Dataset Analysis

The World Factbook data set provides information on the main topics of geography, history, people, government, economy, communications, transportation, military, and transnational issues for 267 world entities. Among them, we selected 20 countries whose internet media coverage data exists in the now corpus data. In our initial analysis in FactBook dataset, we observed that the features of the countries mostly belongs to 2015 to 2017. Since our aim is to see the correlation between news and factbook data, we decided to use the news belonging to last few years. 

Since the dataset is rather small compared to the news data, we downloaded the dataset as json to our local computer. We filtered the data by getting only 20 countries that exist in the News on the web dataset (NowCorpus). These countries are:0      United States, Ireland, Australia, United Kingdom, Canada, India, New Zealand, South Africa, Sri Lanka, Singapore, Philippines,  Ghana, Nigeria, Kenya, Hong Kong, Jamaica, Pakistan, Bangladesh, Malaysia and Tanzania.
 
For each country, Factbook provides us more than 100 facts under the different main topics listed above.
First we read all the given facts under the each topic on the website (https://www.cia.gov/library/publications/the-world-factbook/). We decided on the useful features of the data that we might possibly find some correlation between the news topics that are extracted from news data. 

The following facts are selected to compare with the now corpus data: 
1) People and Society: Population, Age structure, Median age, Population growth rate, Birth rate, Death rate, Net migration rate, Sex ratio, Life expectancy at birth, Religions, Ethnic groups 
2) Economy: GDP - per capita (PPP), Unemployment rate, Inflation rate (consumer prices), Population below poverty line 
3) Energy: Electricity - from other renewable source, Carbon dioxide emissions from consumption of energy
4) Communications: Internet users
5) Government: Country name
6) Geography: Geographic coordinates, Natural hazards, Environment - current issues
        
We extracted the selected features for our 20 specific countries and did some preprocessing to clean the data. These preprocessing includes text data cleaning, splitting text and extracting the only usefull information needed. We also checked which facts include up to date data and which data have clear comparible usefull information. During this process, we decided not to use Natural Hazards feature since the effect, frequency and size of the hazards are not comparible between countries. Our second concern was that distribution of the different hazards are very varied and for 20 countries does not seem possible to do correlation. Similarly, Environmental Issues data also does not provide the degree of the problem and creates similar analysis issues. We also decided to exclude Religious, Population below poverty line and Ethnic groups data since the latest data belongs to the years of 2000-2016 range changing in each countries. For the rest of the facts, we selected the latest date information mostly belongs to 2016. Some of the facts includes values per population, per gender or per age group. In these facts we mostly selected values for overall population.  

Then we created a dataframe including the country names as columns and each fact added as columns. All rate, percentage and numbers in texts converted to float. We plotted all the facts for each countries in /Notebooks/WorldFact.ipynb.

## Now Corpus Dataset Analysis

In Now Corpus dataset have 5 different data file type which are Database, WordLemPoS, Text, Sources, Lexicon. We first downloaded the samples for each of these datafiles and examined which ones are relevant for our research questions. We decided to use Source Data file including each article's source website, source link, word counts etc. 

In the Source Data, we use all the 7 attributes which are textId, #words, date, country, website, url and title.
Since the 2 source data files together has managable size we ran them on the local with the Source_data_exploration notebook. We answered some crucial question about data to see how the news article distributed around country, websites etc. We also checked if we can find the article topics based on URL. We tried this approach to check if the urls already including the topic of the articles or not. Unfortunately, the percentage of found topics were really low around 30%. 
Therefore, we decided to go with a different approach.

In our main approach, we decided to use WordLemPos Data since it includes each article's word list and run LDA to find the news topics for each country data. We selected LDA since it was one to newest state of the art algorithm in topic finding in text data. In the WordLemPos data we selected textId and lemma columns to use for analysis instead of using the raw text data from each article. Lemma columns were already lemmatized and stemmed.

Our further now corpus data set analysis composed of two parts: 
1) Source data anaylsis to see source distribitions and word counts
2) Topic Findng with LDA


### Source Data Analysis
In source data analysis we did analysis for full data of all years, then we also did the analysis for articles and sources we used in the selected year interval. Details of these results can be seen in the website of our project. 
In general we checked from how many unique website resources, the news articles are collected. How many articles are provided for each country. We observed that number of articles are not equally distributed, US has the more articles compared to other countrie and the least number of articles belong to Tanzania with 15848 articles.
In average countries has around 306608 articles. We also explored the total number of words in articles per country. US again has the most word count which makes sense since they have more articles collected. Overall, for all years for each country we have at least 8 million of words collected.
We also explore the article counts per website to understand if the articles collected evenly. Howerver, some websites such as Times of India, Telegraph.co.uk has more articles collected ompared to other resources. Therefore, one needs to consider this fact while interpreting the results.

### Topic Finding with LDA
In natural language processing, latent Dirichlet allocation (LDA) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. For example, if observations are words collected into documents, it posits that each document is a mixture of a small number of topics and that each word's presence is attributable to one of the document's topics. LDA is an example of a topic model.(check https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation for the description).

In order to use this methodology in our project, we  first had to provide a clean wordlist of articles by doing data preparation and noise elimination. Then we contructed LDA model, by determining right parameters for a logical clustering of the data resulting with meaningfull news topics. During this process, several parameters are tested and tuned iteratively by running the LDA on clusters with different parametric combinations. In addition to these steps we also had to discover which lda algorithm in spark is useful and matches our needs, and with which optimizer we can get the best results. 
Each of these steps are explained in detail in the following subsections.

#### Data Preparation
Data preparation step includes time interval selection, sampling and noise elimination for the news data.
##### Time Interval Selection 
In time interval selection process, we had to take into account the up to date provided country facts from factbook data, computational time of LDA and size of the now corpus data. 
 
After analysing the data of factbook we observed that most of the values belong to 2016, in order to correlate fact with news  we decided to use the recent news data for further analysis. 
First we tried 3 and 2 years' news data but computational time was so high (around 40 hours) in LDA. Then we tried 1 year data.
The last year data of Now Corpus belongs to 2016 but does not include last two months. In order to eliminate month bias in the data results we used last 12 months of the data set corresponds to the data published between November 2015 and October 2016.
This data even was very log to process within the given time interval of our project (each 24 hour). Therefore we sampled our data by selecting random articles through the selected year interval and time decreased to 8 hours.
We also paralelized the process to dicrease the run time.

##### Noise Elimination Steps

We cleaned the data from numbers and specific characters then deleted stopwords by using two different already existing lists.
However, in the results there were several words that are not contributing to determine a specific topic per cluster. Afterwards, we applied following steps to make words list data cleaner. Some of the steps are applied by doing several LDA iterations and checking most frequent word lists of each clusters. 

1) Elimination of Stopwords Using already existing spark ml library default stopwordlist.
2) Iterative elimination of unrelated words that are not contributing to the topic selection (such as :also, share, many, like)
We did several LDA clustering iterations and check the resulting most frequent/common words for each topic and iteratively eliminated unrelated words. 
3) Eliminating digits
4) Eliminating less than 3 letter words: this step is used in several previous studies on natural language processing. 
5) Adding back important 2/3 letter word list: we were concerned about eliminating useful important words potantially contributes to the clustering such as art, man, gun, war, eu, us, win, car. Therefore, we decided to create a 2 -3 letter word list to put back to the data. 
6) Iteratively selecting sigma value for tail cutting to create most freq and least freq extra words: In the previous studies on nlp, in order to eliminate noise, most frequent and least frequent words are deleted from the data sets. Since we did not want to eliminate some X percent of the data randomly we used a more scientific methodology. In nlp, the word distribution is assumed as gaussian, in order to cut the tails of this distribution, we decided to use sigma value. We cutted the tails and keep the data within the borders of 1 sigma, 1.5 sigma, and 2 sigma from each side (each side of the graph: to the left x sigma and to the right x sigma),that corresponds to %95.4, %86.6, and %68.3 of the data respectively. In order to see the visual version of this check the following link:
https://thecuriousastronomer.wordpress.com/2014/06/26/what-does-a-1-sigma-3-sigma-or-5-sigma-detection-mean/
Then we did LDA with the combination of these 3 sigma values and different cluster(k) numbers. At the end we got the best results from using 2 sigma.
 

#### LDA Model Construction 
In order to get relevant most frequent words for each clusters and having meaningfull topics we had to decide 
1)how many clusters should be determined in the model
2)which sample size is best to represent the data
3)which sigma value should be selected for most/least frequent word election for cleaning

In order to determine these 3 values to create our final model we run the LDA with combinations of different values of each number. Then we checked the cluster results and thought about possible topics that can be determined from each cluster. We selected the values resulting with the most relevant, meaningful and distinguishable clusters of frequent words. 

We run LDA for each country seperately. The reason behind this approach is following: Each country speaks differently about on topic even the consept is the same. For instance if the Topic is Sports, Canada talks about Hokey while India talks about Cricket. Correspondingly the name of the sports celebrities changes as well. Similarly, if the topic is politics in India we can see the religion related words while in US we see Trump. However, if we see Trump word in another country the word belongs to International topic. Therefore, running the model on a mixed country data may result in either very undistinguishable clusters with very general words or wrong topic assignments.

##### Selecting number of cluster (k) to select  proper number of meaningful topics
We first tried 3-4-5 to have similar topics for each country but the most frequent words in the clusters were very close. 
Then we tried 7-10-13  number of cluster and selected 7 since 10-13 were not distinguishable while 7 gives the logical news topics. 

##### Sample size selection 
As we discussed in the Time Interval Selection section, we decided to sample the data. In order to determine which percentage of the data will be sampled we iteratively sampled and tested %10, %20 and %25 of the data. The best result is btained from %25 sampling.

##### Selecting sigma value for tail cutting
In order to find most logical topics and better clustering by cutting most frequenct and least frequent words we tried different sigma values to cut the tails of our word data distribution for each country. Best result is obtained with the k=7, sampling: %25 and using sigma: 2 which corresponds to %95.4 of our sampled word data.

#### LDA Model Selection 
We tried two different lda library: mllib clustering lda and ml.clustering lda.
mllib clustering lda was used and we reognized that after clustering it doe not provide a function for cluster assignment for each article. It was also problematic in paralelization.
ml.clustering lda improved our computational time by paralelization and it provides easier topic assignment after clustering using transform function.
Second model selected according to lower perplexity score ve meaningfull topic dagilimina baktik
Log likehoods are also taken into account.

##### LDA optimizer selection

Each optimizer in lda model provides different list of most frequent word list we selected most meaningfull. **EM optimizer.
Online optimizer

Also em has better perplexity score.


## Correlation of Two Data
-grouping
-pearson spearman


## Contributions
Gorkem Camli: choice of datasets, creating the plan for each milestone, exploratory data analysis and attribute description on Now Corpus Data, generating interactive graphs,generating interactive maps, analysis of final results, creating a website that also serves as a platform for the data story, development of project topic, commenting the code, writing the explanations in the notebook, writing the data story, LDA model construction, LDA code implementation and iterative run on clusters,topic selection for each country, correlation analysis. 

Arzu Guneysu Ozgur: creating the plan for each milestone,exploratory data analysis and attribute description on Factbook data, aggregating data and plotting, analysis of final results,data preparation,sigma value use in data cleaning,generating interactive maps, creating content for the website, commenting the code, writing the explanations in the notebook,writing the data story,LDA model construction, development of project topic,topic selection for each country, correlation analysis.

Ezgi Yuceturk: choice of datasets,creating the plan for each milestone,LDA model construction, LDA code implementation and iterative run on clusters,code implementation for managing the big data on spark,analysis of final results, developing host website for the final presentation,development of project topic, commenting the code, writing the explanations in the notebook, writing the data story, topic selection for each country, topic assignmentcorrelation analysis.

# Planning for each Milestone
## Prepare and Explore Data: Until Nov 11.
- Understand how to manage the data. 
- Decide on how to filter the now corpus to have a managable data size.
- Decide on which attributes will be taken from Factbook data.
- Collect and filter both data.
- Clean the datasets.
- Do descriptive statistics and exploration of the data.

## Enrich Data and Start Analysis: Until Nov 18.
- Find the most frequent X words in each article (excluding stop words).
- Start doing preliminary analysis of news content.
- Start doing preliminary analysis of country profiles vs news content.

## Finalize and Revise the Work done for Milestone 1: Until Nov 25.
- Finalize preliminary analyses.
- Revise and comment the code.
- Decide the next steps.

## Prepare and Explore Data: Until Dec 2.
- Precise the topics per each country:
  Run LDA for each country (we do it separately otherwise the words selected might be more biased per some country has more words or articles such as US).  
- For each article do the topic assignment.
- For each country find topic percentages.

## Find Correlations between Now Corpus and FactBook: Until Dec 9.
- Find the country-wise correlations between Now Corpus and Factbook.
- Gather analysis and correlation results.
- Do data visualization.

## Finalize and Revise the Work done for Project: Until Dec 16.
- Finalize the project,
- Clean, comment and revise the notebooks.
- Create data story.

# Folder Structure

This project's github repository folder structure explained below:

We have 3 folders:
- Data:
    In this folder, we only included the data we have extracted from FactBook dataset for 20 countries and the parquet data that we created for the analysis.   
    - Factbook.json : Factbook data from CIA for 20 countries
    - data.parquet folder: Query results saved from pyspark_script.py results
- Scripts:
    -  pyspark_script.py: Source data filtered on cluster with pyspark.
- Notebooks:
    - WorldFact.ipynb : Data Exploration and Analysis for FactBook data
    - Source_Data_Exploration.ipynb : Now Corpus data: Source data: Data Exploration and Analysis 
    - WordLemPos_Topic_Modelling.ipynb : Now Corpus data: WordLemPos data: Data Exploration, Analysis and LDA Topic Modelling
    - Spark_Notebook.ipynb : Initial steps to understand managing big data in spark and cluster
    
__Order of notebooks__ are same as above. You can start checking the notebooks with WorldFact notebook for FactBook data analysis then proceed with Source data Exploration for Now Corpus Source analysis and then continue with WordLemPos_Topic_Modelling Notebook to see how we find the topics for news. The last notebook called Spark_Notebook
this notebook contains our initial steps to understand how we can run and manage the big data on cluster with pyspark. We initally tried this with Source data of Now Corpus, and compare with our previous result, for the next steps we will further use this for WordLemPos data.
