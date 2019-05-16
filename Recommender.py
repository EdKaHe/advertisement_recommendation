from scipy.stats import norm
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

class Recommender:
    # def __init__(self):
        
    def fit(self, profile, portfolio, transcript):
        # add the dataframes to the dataframe
        self.profile = profile
        self.portfolio = portfolio
        self.transcript = transcript
    
        # clean the dataframes
        self.__clean_data()
        
    def predict(self, user_id):
        pass
    
    def __clean_data(self):
        # get the dataframes from the dataframe
        profile = self.profile 
        portfolio = self.portfolio
        transcript = self.transcript
    
        # rename the ids to user_id and offer_id
        profile = profile.rename(columns={"id": "user_id"})
        portfolio = portfolio.rename(columns={"id": "offer_id"})
        transcript = transcript.rename(columns={"person": "user_id"})
        
        # one hot encode the channels in the advertisment portfolio
        mlb = MultiLabelBinarizer()
        # binarize the channels column
        values = mlb.fit_transform(portfolio.pop("channels"))
        # get the names of the binarized values
        columns = mlb.classes_
        # create a dataframe out of the binarized values
        channels = pd.DataFrame(values, columns=columns)
        # join the one hot encoded column
        portfolio = portfolio.join(channels)
        
        # one hot encode the gender in the user profiles
        genders = pd.get_dummies(profile.pop("gender"))
        profile = profile.join(genders)
        
        # remove the date when the user got a member
        profile = profile.drop(columns=["became_member_on"])
        
        # perform the mapping to the new columns
        transcript["key"] = transcript["value"].apply(lambda x: "offer_id" if list(x.keys())[0] == "offer id" else list(x.keys())[0])
        transcript["value"] = transcript["value"].apply(lambda x: list(x.values())[0])
        
        # categorize the numerical age column in the user profiles
        bins = [0, 2, 10, 20, 30, 40, 50, 60, 70, 80, 110, np.inf]
        labels = ['<2y', '2-10y', '10-20y', '20-30y', '30-40y', '40-50y', '50-60y', '60-70y', '70-80y', '80-110y', 'false_age']
        binned_age = pd.get_dummies(pd.cut(profile.pop("age"), bins, labels=labels))
        profile = profile.join(binned_age)
        
        # convert numerical data into bins of income ranges
        bins = [0, 20_000, 40_000, 60_000, 80_000, 100_000, np.inf]
        labels = ['<20,000USD', '20,000-40,000USD', '40,000-60,000USD', '60,000-80,000USD', '80,000-100,000USD', '>100,000USD']
        binned_income = pd.get_dummies(pd.cut(profile.pop("income"), bins, labels=labels))
        profile = profile.join(binned_income)
            
        # get the transactions of each user
        transactions = transcript.loc[transcript['event']=="transaction", ["user_id", "value"]]
        transactions["value"] = transactions["value"].astype("int")
        # calculate the average spendings of each user and the standard deviation of the spendings
        average_transaction = transactions.groupby('user_id')[['value']].mean()
        stddev_transaction = transactions.groupby('user_id')[['value']].std()
        # rename the value column
        average_transaction = average_transaction.rename(columns={'value': 'average_transaction'})
        stddev_transaction = stddev_transaction.rename(columns={'value': 'stddev_transaction'})
        # join the new dataframe with the profile dataframe
        profile = profile.join(average_transaction, on='user_id')
        profile = profile.join(stddev_transaction, on='user_id')
        # replace missing values in the average and stddev transaction column
        profile.loc[profile['average_transaction'].isnull(), 'average_transaction'] = 0
        profile.loc[profile['stddev_transaction'].isnull(), 'stddev_transaction'] = 0
        
        # add the dataframes to the instance
        self.profile = profile
        self.portfolio = portfolio
        self.transcript = transcript  
    
    @staticmethod
    def __collaborative_recommendation():
        pass
    @staticmethod
    def __popular_recommendation():
        pass