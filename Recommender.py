from scipy.stats import norm
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

class Recommender:
    # def __init__(self):
        
    def fit(self, profile, portfolio, transcript):
        # add the dataframes to the dataframe
        self.profile = profile
        self.portfolio = portfolio
        self.transcript = transcript
    
        # clean the dataframes
        profile, portfolio, transcript = self.__clean_data(profile, portfolio, transcript)
        
        # get the sparse user item matrix
        user_item = self.__user_item(profile, portfolio, transcript)
        
        # fill the missing values using matrix factorization
        user_item = Recommender.__matrix_factorization(user_item)
        
        # add the full user item dataframe to the instance
        self.user_item = user_item
        
    def predict(self, user_id):
        pass
    
    def __clean_data(self, profile, portfolio, transcript):
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
        
        return profile, portfolio, transcript 
        
    def __user_item(self, profile, portfolio, transcript):
        
        # merge the average transactions with stddev to the transcript dataframe
        transcript = transcript.merge(profile[["user_id", "average_transaction"]], on="user_id", how="left")
        # get the received offers
        received = transcript[transcript["event"]=="offer received"]
        # get the completed offers
        completed = transcript[transcript["event"]=="offer completed"]

        # get the offers that were promoted and completed
        completed = completed.loc[completed["user_id"].isin(received["user_id"]) & completed["value"].isin(received["value"]), \
            ["user_id", "value", "average_transaction"]]
        completed = completed.rename(columns=dict(value="offer_id"))

        # add the difficulty of an offer to the dataframe
        completed = completed.merge(portfolio[["offer_id", "difficulty"]], on="offer_id")
        
        # calculate the yield
        completed["yield"] = completed["difficulty"] - completed["average_transaction"]
        
        # construct the user item dataframe with the yield as basis of assesment
        user_item = completed[["user_id", "offer_id", "yield"]].groupby(["user_id", "offer_id"]).mean().unstack()
        user_item.columns = user_item.columns.droplevel()
        
        return user_item
    
    @staticmethod
    def __matrix_factorization(user_item, latent_features=20, learning_rate=0.0001, iters=100):
        # get the user item matrix
        user_item_mat = user_item.values
        
        # get the number of users, offers and ratings
        n_users = user_item_mat.shape[0]
        n_offers = user_item_mat.shape[1]
        n_ratings = np.sum(~np.isnan(user_item_mat))
        
        # initialize the user and offer matrices with random values
        user_mat = np.random.rand(n_users, latent_features)
        offer_mat = np.random.rand(latent_features, n_offers)

        # initialize sse at 0 for first iteration
        sse_accum = 0

        # keep track of iteration and MSE
        print("Optimizaiton Statistics")
        print("Iterations | Mean Squared Error ")

        # for each iteration
        for iteration in range(iters):

            # update our sse
            old_sse = sse_accum
            sse_accum = 0

            # For each user-offer pair
            for i in range(n_users):
                for j in range(n_offers):

                    # if the rating exists
                    if ~np.isnan(user_item_mat[i, j]):

                        # compute the error as the actual minus the dot product of the user and offer latent features
                        diff = user_item_mat[i, j] - np.dot(user_mat[i, :], offer_mat[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff ** 2

                        # update the values in each matrix in the direction of the gradient
                        for k in range(latent_features):
                            user_mat[i, k] += learning_rate * (2 * diff * offer_mat[k, j])
                            offer_mat[k, j] += learning_rate * (2 * diff * user_mat[i, k])

            # print results
            print("%d \t\t %f" % (iteration+1, sse_accum / n_ratings))
        
        return user_mat @ offer_mat
        
    @staticmethod
    def __popular_recommendation():
        pass