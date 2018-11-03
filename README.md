# Country Overviews Through Online Newspapers and Magazines

# Abstract

> “A good newspaper is a nation talking to itself.” \
>  Arthur Miller

In the newspaper and magazine articles we cover more than events, we share what we are doing, what we are interested in and what matters to us as a society. Our aim in this project is to see how the news content changes for countries over time and is there a connection between country profiles and the news published. Our motivation behind this project is to understand, the underlying profile of the published news content: the topics of the news, its tone, most commonly used words and whether these information change between countries or time. Do the news become more global or they manage to conserve some trends from the country they are published? We aim to see if country specific information and published news are somehow correlated or a good newspaper/article is now a **_world_** talking to itself? We will use the News on the Web dataset more specifically, Now corpus data and we will gather the country profiles from The World Factbook data [[1]](https://www.cia.gov/library/publications/the-world-factbook/).     


# Research questions
## Contents of the News
- What are the main topics of the published news? (tech, politics, sports, etc.)
- What are the distributions of these topics over country and time?
- Is there a dominant tone  in the articles based on topic/country/time?
- What are some mostly used words in the articles? 

## News and Country Profiles
- Is there some trends or patterns between news articles and country profiles?
  - Are countries having X attribute, publish more news on topic Y?
  - Are countries having X similar attribute, uses similar tones in the news?
  - Are countries having X attribute, use Z common specific words more often?
- If there is some patterns or correlation, do they change over time?

# Dataset
## News on the Web: Now Corpus

Now Corpus has online magazine and newspaper text data from 20 different English speaking countries collected each day. The dataset we have in the cluster is for 6 years from 2010-2016. It has lexicon, source, text and wlp (word, lemma, PoS tag) data.  
The data is in txt format but th data size is too large, it is around 5.9 million words, in terms of computation time we might need to limit and use part of the dataset, we can do this either by choosing some specific countries or specific time frames. We might limit the time by just doing the analysis for 1 year, or some selected months for each year or some selected days for each month. 

We can enrich the dataset by doing following:
- By using the source data we can group the news by country and time. 
- By using the title and url information of the news we can infer the topic of the newspaper (most cases url contains sports, healthcare, tech words in the urls).
- By using the original text and wlp data we can find the tone (positive, negative) of the article with the help of nltk or textblob library.
- By using the wlp data we can find the frequency of the words in the text.


## World Factbook

World Factbook contains the daa about each country profiles collected by CIA. The data is open source and can be obtained from [this link](https://github.com/factbook/factbook.json) as a json format. We are only interested in 20 countries therefore, we will use only the countries in which we have news article available. Also, the country profiles are in-depth, we will choose key information such as population, age profile, sex ratio, and other socio-economic informations. 


# A list of internal milestones up until project milestone 2

## Prepare and Explore Data: Until Nov 11.
- Understand how to manage the data. 
- Decide on how to filter the now corpus to have a managable data size.
- Decide on which attributes will be taken from Factbook data.
- Collect and filter both data.
- Clean the datasets.
- Do descriptive statistics and exploration of the data.

## Enrich Data and Start Analysis: Until Nov 18.
- Find the topics for each article.
- Find the tone/sentiment of each article.
- Find the most frequent X words in each article (excluding stop words).
- Start doing preliminary analysis of news content.
- Start doing preliminary analysis of country profiles vs news content.

## Finalize and Revise the Work done for Milestone 1: Until Nov 25.
- Finalize preliminary analyses.
- Revise and comment the code.
- Decide the next steps.

# Questions for TAa
- Can we pick a specific time interval instead of using the whole now corpus dataset? If so, which approach do you think would be more interesting to pursue: limit the time by just doing the analysis for 1 year, or some selected months for each year or some selected days for each month of every year?
- Where can we store the newly created variables for the now corpus data?

