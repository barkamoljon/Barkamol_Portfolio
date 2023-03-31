# Barkamol_Portfolio
Barkamol Urinboev

## [Project 1: Credit Risk Modeling](https://github.com/barkamoljon/PortfolioProjects/tree/main/Credit_Risk_Modeling)

## Overview
This project aims to measure the credit risk of LendingClub, (an American peer-to-peer lending company), by calculating the expected loss of their outstanding loans. Credit risk is the likelihood that a borrower would not repay their loan to the lender. By continually evaluating the risk and adjusting their credit policies, the lender could minimize its credit losses while it reaches the fullest potential to maximize revenues on loan borrowing. It is also crucial for the lender to abide by regulations that require them to conduct their business with sufficient capital adequacy, which, if in low, will risk the stability of the economic system.

The key metric of credit risk is Expected Loss (EL), calculated by multiplying the results across three models: PD (Probability of Default), LGD (Loss Given Default), and EAD (Exposure at Default). The project includes all three models to help reach the final goal of credit risk measurement.

 
## Requirements
* __Python Version__: 3.10.0
* __Packages__: pandas, numpy, sklearn, scipy, matplotlib, seaborn, pickle
* __Algorithms__: regression (multiple linear), classification (logistic regression)
* __Dataset Source__: https://www.kaggle.com/datasets/barkamolurinboev/credit-risk-default

## [Project 2: Customer Analytics](https://github.com/barkamoljon/PortfolioProjects/tree/main/Customer%20Analytics)

## Overview
This project aims to support a retail or FMCG (fast-moving consumer goods) company to formulate marketing and pricing strategies that could maximize revenues on each brand of candy bars. To reach the fullest potential of bringing up revenues, a company should find the 'sweet spot' for price to maximize three customer behaviours: purchase probability, brand choice probability, and purchase quantity. 

Data from customer purchase history were used for training the regression models to predict those three customer behaviours in a preconceived price range. The results were then converted into price elasticities so that we can examine the effects of changing price on each of the behaviours. Hence, we will be able to find the suitable marketing and pricing strategies.

To better position our products, we will firstly perform segmentation on our customers to support our analysis on customer behaviours, allowing us to customize marketing strategies for customers with different backgrounds.


## Requirements
* __Python Version__: 3.9.6
* __Packages__: pandas, numpy, sklearn, scipy, matplotlib, seaborn, pickle, tensorflow 
* __Algorithms__: clustering(K-means, PCA), regression(logistic, linear), neural network
* __Dataset Source__: https://www.kaggle.com/datasets/barkamolurinboev/audiobooks-data/settings?select=New_Audiobooks_Data.csv

## [Project 3: Absenteeism](https://github.com/barkamoljon/PortfolioProjects/tree/main/Absenteeism)

## Overview
This project is a part of the PortfolioProjects repository by barkamoljon and focuses on predicting absenteeism at work using machine learning algorithms. The project uses the "Absenteeism at work" dataset and the goal is to build a model that can accurately predict the probability of an employee being absent from work.

## Requirements
* __Python Version__: 3.9.6
* __Packages__: pandas, numpy, and sklearn
* __Algorithms__: regression(logistic, linear), BaseEstimator, TransformerMixin, metrics
* __Dataset Source__: https://www.kaggle.com/datasets/barkamolurinboev/absenteeism

## [Project 4: Tableau Portfolio Project SQL Queries](https://github.com/barkamoljon/PortfolioProjects/blob/main/Tableau%20Portfolio%20Project%20SQL%20Queries.sql)

## Overview
The Tableau Portfolio Project is a collection of SQL queries and Tableau dashboards that explore a dataset of customer orders for a fictitious company. The dataset includes information on customer orders, products, and sales, and the goal of this project is to provide insights into the company's sales and customer behavior.

The SQL queries are used to prepare the data for analysis, and the results are then visualized in Tableau dashboards. The dashboards provide an interactive way to explore the data and to discover insights into the company's sales trends, customer behavior, and product performance.

The project demonstrates skills and knowledge in SQL, data cleaning and preparation, data analysis, and data visualization using Tableau. By exploring the data and creating interactive dashboards, this project provides valuable insights into the company's sales and customer behavior, which could be used to inform business decisions and strategies.

## Requirements
* __Microsoft SQL Server Version__: 18.12.1
* __Packages__: tableau, microsoft sql server, sql queries
* __Dataset Source__: https://www.kaggle.com/datasets/barkamolurinboev/airbnb-listings-2016

## [Project 5: Data Cleaning in SQL Queries](https://github.com/barkamoljon/PortfolioProjects/blob/main/Data%20Cleaning%20with%20SQL%20Queries.sql)

## Overview
Data cleaning is a crucial step in the data analysis process, and it involves identifying and correcting errors, inconsistencies, and inaccuracies in data. The SQL script in this project provides a set of queries that can be used to perform various data cleaning tasks.

The script is well-organized and divided into sections that correspond to specific data cleaning tasks. For example, the section on handling missing data provides queries for identifying and handling missing values in a dataset. Similarly, the section on data type conversions provides queries for converting data from one data type to another.

One of the strengths of this project is that it covers a wide range of data cleaning tasks. The queries provided can be used to remove duplicates, standardize data, and address inconsistencies in data. The queries are also customizable, allowing users to modify them to suit their specific needs.

Overall, the "Data Cleaning SQL Queries" project is a valuable resource for anyone working with data in SQL. The queries provided can save time and effort by automating many data cleaning tasks and ensuring that data is accurate and ready for analysis. The project is a testament to the importance of data cleaning and the power of SQL as a tool for managing and manipulating data.

## Requirements
* __Microsoft SQL Server Version__: 18.12.1
* __Packages__: tableau, microsoft sql server, sql queries
* __Dataset Source__: https://www.kaggle.com/datasets/barkamolurinboev/nashville-dataset



## [Project 6: COVID Portfolio Project](https://github.com/barkamoljon/PortfolioProjects/blob/main/COVID%20Portfolio%20Project%20-%20Data%20Exploration.sql)

## Overview
The project is titled "COVID Portfolio Project - Data Exploration" and the code is written in SQL language. The main objective of the project is to explore and analyze COVID-19 data using various SQL queries and visualizations.

The project begins with importing COVID-19 data into a SQL database, followed by data cleaning and pre-processing. Then, the author uses various SQL queries to answer questions related to COVID-19, such as:

- What is the total number of confirmed cases and deaths in each country?
- How has the number of cases and deaths changed over time?
- What are the top 10 countries with the highest number of cases and deaths?
- How has the spread of COVID-19 varied by continent?

The author also uses SQL to create visualizations such as bar charts, line graphs, and heatmaps to help better understand the data.

Overall, the project provides a good example of how SQL can be used to analyze and visualize COVID-19 data, and may be useful for anyone interested in data exploration or epidemiology.

## Requirements
* __Microsoft SQL Server Version__: 18.12.1
* __Packages__: tableau, microsoft sql server, sql queries
* __Dataset Source__: https://www.kaggle.com/datasets/barkamolurinboev/covid19

## [Project 7: Resume Parser with NLP](https://github.com/barkamoljon/Resume_Parser)

## Overview
The Resume Parser with NLP is a program that uses natural language processing (NLP) to extract essential information from resumes. The program receives resumes in PDF format, and it processes them using several NLP techniques to extract personal and professional details such as name, email, phone number, work experience, education, skills, and other relevant information.

The program uses several Python libraries such as spaCy, PyPDF2, and Regex to perform NLP tasks such as tokenization, parsing, named-entity recognition, and information extraction. It also uses machine learning models to classify the information and extract specific details. 

The program is designed to automate the resume screening process, making it more efficient and accurate. It can be used by HR departments or recruitment agencies to quickly identify the most promising candidates and filter out irrelevant applications. 

Overall, the Resume Parser with NLP is a powerful tool that streamlines the recruitment process and helps organizations find and hire the best candidates for their open positions.

## Requirements
* __Python Version__: 3.9.6
* __Packages__: pandas, numpy, spacy, gradio, json, pypdf2, fitz, and sklearn
* __Algorithms__: neaural network, nlp
* __Dataset Source__:  https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner


## [Project 8: ChatBot](https://github.com/barkamoljon/PortfolioProjects/tree/0dd26d1e938c06592f8a1289d04a0a7c7beb7f3b/Developer%20ChatBot%20with%20ChatGPT%20API)

## Overview
The "Developer ChatBot with ChatGPT API" is a project that aims to develop a chatbot using the ChatGPT API to assist developers with their programming-related queries. The project is developed in a Jupyter Notebook, which allows for easy implementation and execution of the code.

The chatbot is implemented using a while loop, which keeps running until the user types "bye" to end the conversation. The ChatGPT API is used to generate the responses of the chatbot. To use the API, you need to send a POST request to the API endpoint with the user's input as the query parameter. The API returns a JSON object containing the response of the chatbot, which is then displayed to the user.

The project is a valuable tool for any programming community as it provides instant solutions to users' queries. It can be customized to suit different requirements, making it an excellent tool for developers who want to create a chatbot to help other developers. Overall, the "Developer ChatBot with ChatGPT API" project is an excellent example of how machine learning can be used to develop intelligent chatbots to assist users with their queries.

## Requirements
* __Python Version__: 3.9.6
* __Packages__: gradio, and openai
* __Algorithms__: OpenAI's GPT-3

## [Project 9: Time Series Forecasting](https://github.com/barkamoljon/PortfolioProjects/tree/main/Time%20Series%20Forecasting)

## Overview
Overview of Time Series Forecasting using FBProphet.ipynb:

The goal of this project is to demonstrate time series forecasting using the FBProphet library in Python. The dataset used in the project is the daily number of passengers for an airline company, spanning from 1949 to 1960. The project involves data preprocessing, visualization, modeling, forecasting, and evaluation of the model.

In data preprocessing, the "Month" column is parsed, the columns are renamed, and the "Passengers" column is converted to a numeric type. In data visualization, the time series data is plotted to visualize the trend and seasonality.

In modeling, a Prophet model is created and fit to the data. The model is trained on the first 80% of the data and tested on the remaining 20%. In forecasting, the model is used to forecast the number of passengers for the next few years.

In evaluation, the performance of the model is evaluated using various metrics such as mean absolute error, mean squared error, and root mean squared error. In visualization, the forecasted data is plotted along with the historical data to visualize the accuracy of the model.

The project demonstrates that FBProphet can be used to effectively forecast time series data. The model accurately captures the trend and seasonality of the data and provides valuable insights for decision-making. The notebook can be used as a guide for those interested in learning how to use FBProphet for time series forecasting.


## Requirements
* __Python Version__: 3.9.6
* __Packages__: pandas, numpy, matplotlib, seaborn, sklearn, fbprophet, and pystan
* __Algorithms__: metrics(mean_absolute_error)
* __Dataset Source__: https://www.kaggle.com/code/caglarhekimci/time-series-forecasting-fb-prophet



## [Project 10: Data Cleaning Portfolio Project Queries](https://github.com/barkamoljon/PortfolioProjects/blob/main/Data%20Cleaning%20Portfolio%20Project%20Queries.sql)

## Overview
The project titled "Data Cleaning Portfolio Project Queries" is a SQL-based project that focuses on data cleaning and data manipulation. The project begins by importing data into a SQL database, followed by data cleaning and pre-processing using various SQL queries.

The author employs a wide range of SQL queries to clean and manipulate the data, such as:

- Removing duplicate rows from the data.
- Handling missing values using various techniques such as imputation and deletion.
- Renaming columns and changing data types to improve data quality.
- Normalizing and standardizing data to ensure consistency.
The project also includes examples of how to use SQL queries to merge and join data from multiple tables, as well as how to filter and select specific rows based on certain conditions.

Overall, the project provides a good example of how to use SQL queries to clean and manipulate data effectively. The techniques used in the project may be useful for anyone interested in data cleaning, data manipulation, or SQL programming.

## Requirements
* __Microsoft SQL Server Version__: 18.12.1
* __Packages__: microsoft sql server, sql queries
* __Dataset Source__: https://www.kaggle.com/datasets/barkamolurinboev/nashville-dataset

## [Project 11: House Price Prediction](https://github.com/barkamoljon/PortfolioProjects/blob/main/House_Price_Prediction.ipynb)

## Overview
The project titled "House Price Prediction" is a Jupyter Notebook-based project that focuses on predicting house prices using various machine learning algorithms. It begins with importing and exploring a dataset of house prices, followed by data cleaning and pre-processing.

The author employs a wide range of machine learning algorithms to predict house prices, such as:

- Linear regression
- Decision trees
- Random forests
- Gradient boosting
The author uses various techniques to evaluate the performance of the models, such as mean squared error (MSE) and root mean squared error (RMSE). Additionally, the author uses various data visualization techniques to help better understand the data and the model's predictions.

The project also includes examples of how to use feature selection techniques to select the most important features in the dataset, which can help improve the accuracy of the models.

Overall, the project provides a good example of how to use machine learning algorithms to predict house prices and may be useful for anyone interested in machine learning, data analysis, or real estate.

## Requirements
* __Python Version__: 3.9.6
* __Packages__: pandas, numpy, matplotlib, seaborn, sklearn, lazypredict, and joblib
* __Algorithms__: regression(logistic, linear), Decision trees, TransformerMixin, Gradient boosting, metrics(RMSE, MSE)
* __Dataset Source__: https://github.com/anvarnarz/praktikum_datasets/blob/main/housing_data_08-02-2021.csv

## [Project 12: Amazon Web Scraper Project](https://github.com/barkamoljon/PortfolioProjects/blob/main/Amazon%20Web%20Scraper%20Project.ipynb)

## Overview
The Amazon Web Scraper Project is a Python-based project that focuses on scraping data from Amazon's website using the BeautifulSoup library. 
It can automatically extract product data from Amazon 86,400 times a day through Python libraries and generate product Dataset
The project involves collecting information on product names, prices, ratings, and descriptions from Amazon's Best Sellers page. The data collected is then stored in a CSV file for future analysis. The code is heavily commented, making it easy for beginners to understand, and provides ample opportunities for customization or further development. The project is an excellent way to learn about web scraping, data handling, and Python programming as a whole. By the end of the project, developers will have gained a valuable skillset that they can apply to numerous other projects.

## Requirements
* __Python Version__: 3.9.6
* __Packages__: pandas, BeautifulSoup, requests, time, datetime, and smtplib
* __Dataset Source__: https://www.amazon.com/Mathematics-Machine-Learning-Peter-Deisenroth/dp/110845514X/ref=sr_1_1?crid=10QFPWPXWB3D9&keywords=math+for+machine+learning&qid=1677503609&sprefix=MAth+for+machi%2Caps%2C701&sr=8-1


## [Project 13: Movie Portfolio Project](https://github.com/barkamoljon/PortfolioProjects/blob/main/Movie%20Portfolio%20Project.ipynb)

## Overview
The project focuses on analyzing a dataset of movie ratings and reviews using Python. The project begins by importing necessary libraries such as pandas, matplotlib, and seaborn.

The author then proceeds to clean and pre-process the dataset to ensure that it is in a format suitable for analysis. The pre-processing techniques used include handling missing values, removing duplicate data, and transforming data types.

The author uses various data visualization techniques to help better understand the data, such as creating histograms, scatter plots, and heatmaps. The project also includes examples of how to use statistical analysis to gain insights from the data, such as calculating mean, median, and mode.


Overall, the project provides a good example of how to use data analysis gain insights from a dataset of movie ratings and reviews. The techniques used in the project may be useful for anyone interested in data analysis,  or the movie industry.

## Requirements
* __Python Version__: 3.9.6
* __Packages__: pandas, numpy, matplotlib, and seaborn
* __Dataset Source__: https://www.kaggle.com/datasets/danielgrijalvas/movies
