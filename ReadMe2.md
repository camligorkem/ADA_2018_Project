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

In Now Corpus dataset have 5 different data file type which are Database, WordLemPoS, Text, Sources, Lexicon. We first downloaded the samples for each of these datafiles and decided to use Sources and WordLemPos since we are interested in finding topics related to each news article.      

In the Source file we use all the 7 attributes which are textId, #words, date, country, website, url and title.
Since the 2 source data files together has managable size we ran them on the local with the Source_data_exploration notebook. We answered some crucial question about data to see how the news article distributed around country, websites etc. Also, we checked if we can find the article topics based on URL, however the % of found topics were really low around 30%, therefore we decided to go with a different approach and run LDA to find the topics on article text in order to do that we use WordLemPos data.     

In the WordLemPos data we selecte textId and lemma columns to use for analysis. Instead of using the raw text data from each article, what we did is to use the lemma column from the WordLemPos data, they were already lemmatized and stemmed so it eased the preprocessing part for us. What we did next clean the data numbers and specific characters, delete some stopwords and unnecessary words, create dictionary and doc2bow that has for each article words and their corresponding count. Then, we ran the LDA data model to find 7 topics (we did a few trials with less and more topics numbers but thought for now 7 returns good results nor too geneal neither too specific.). For now, we ran the results on 1 country:US, 1 month data to see the results are meaningful. As next step we will apply this approach to every 20 country for the last few years data.

### Source Data Analysis

### Topic Finding with LDA

#### Data Preparation

##### Time Interval Selection 
First we tried 3 years data then 2 and 1 year data.
Computational time was so high like 40 hours in LDA.
We sampled the data of 1 year as well.
* 24 saat surdu 
sample yaptik 8 e dustu. 

Based on our RQs we extracted last year.

##### Noise elimination steps
1) Elimination of Stopwords Using already existing spark ml library default stopwordlist.
2) Iteractive elimination of unrelated words that are not contributing to the topic selection (also, share, many, like etc.)
We did several clustering iterations and check the resulting most frequent/common words for each topic and iteratively eliminated unrelated words. 
3) Eliminating digits
4) Eliminated less than 3 letter words
5) Iteratively selecting sigma value for tail cutting to create most freq and least freq extra words.
6) Adding back important 2/3 letter word list


##### Extra words elimination through creating extralist with iterations

#### LDA Model Construction 


##### Selecting number of cluster (k) to select  proper number of meaningful topics- 
3-4-5 cok yakin cikti
then we tried 7-10-13 
10-13 were not distinguishable

##### Sample size selection - %10 %20 **%25 denendi

##### Selecting sigma value for tail cutting
data noise orani belirledik 
finding most logical topics and better clustering by using sigma

#### LDA Model Selection 
mllib clustering lda , no topic assignment function , problematic paralalization
*ml.clustering lda,  paralalized, faster topic assignment after clustering is available using transform function

Final model selected according to lower perplexity score ve meaningfull topic dagilimina baktik
Log likehoods are also taken into account.
##### LDA optimizer selection

Each optimizer in lda model provides different list of most frequent word list we selected most meaningfull. **EM optimizer.
Online optimizer

Also em has better perplexity score.


## Correlation of Two Data
-grouping
-pearson spearman




# Planning
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
