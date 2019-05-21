# advertisement_recommendation

## Introduction
This repository contains the capstone project of my Data Scientist Udacity Nanodegree. The goal is to analyse the datasets in 
the example_data folder and recommend offers to customers that are profitable for the company. 

## Data Acquisition
Exemplary data is provided in the example_data folder. The data can be extended and/or changed as desired. 
However, the format must stay the same to ensure proper functionality of the provided module.

## Files
The main file in this project is recommender.py. It contains a Recommender class with all required algorithms, 
which can be used for the data analysis and recommendations. The example.ipynb notebook contains some examples
on how to use the Recommender class properly. The analysis.ipynb notebook contains the analysis that was performed 
for the Medium article [here](https://medium.com/@edizherkert/optimizing-the-profitability-of-customer-advertisements-b652c1e56bdb). In the optimization.ipynb notebook the grid search optimization for fine-tuning the matrix factorization is performed.
The example_data folder contains the datasets to analyse and the exports folder contains exported images from the analysis.ipynb notebook.

## Problem Formulation
The goal of this project is to find a way to measure the profitability of advertisments and make recommendations to increase
the profit of promotional measures of a company. Here, profitability is measured as the difference between the average customer transactions
and the investment that a customer has to do to complete an offer.

## Results
A detailed discussion of the analysis and the outcomes can be found in the Medium article 
[here](https://medium.com/@edizherkert/optimizing-the-profitability-of-customer-advertisements-b652c1e56bdb).
