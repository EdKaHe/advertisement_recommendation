import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import MultiLabelBinarizer

class Recommender:        
    def fit(self, profile, portfolio, transcript):
        """
        cleans the profile, portfolio and transcript dataframes and stores
        them in the clean_profile, clean_transcript and clean_portfolio attributes. From
        this the user item matrix is computed on the basis of yield per advertisment for each user.
        The sparse user item matrix is filled using matrix factorization and stored as user_item
        in the instance attributes.

        Args:
            profile (DataFrame): informations about the users in the same format
                as supplied in the example folder
            portfolio (DataFrame): informations about the advertisments in the same 
                format as supplied in the example folder
            transcript (DataFrame): temporal informations about user actions in the same
                format as supplied in the example folder

        Returns:
        """
    
        # add the dataframes to the dataframe
        self.profile = profile
        self.portfolio = portfolio
        self.transcript = transcript
    
        # clean the dataframes
        clean_profile, clean_portfolio, clean_transcript = Recommender._clean_data(profile, portfolio, transcript)
        
        # add the cleaned data to the instance
        self.clean_profile = clean_profile
        self.clean_portfolio = clean_portfolio
        self.clean_transcript = clean_transcript
        
        # get the sparse user item matrix
        user_item = Recommender._user_item(clean_profile, clean_portfolio, clean_transcript)
        
        # fill the missing values using matrix factorization
        full_user_item = Recommender._matrix_factorization(user_item)
        # add the full user item dataframe to the instance
        self.full_user_item = full_user_item
        
    def predict(self, user_id, confidence=0.9):
        """
        gets a sorted list of offers with the predicted yield. If the user is in the database the 
        prediction is done on the basis of collaborative filtering. Otherwise the most popular ads 
        are recommended. Only offers that will show profit with a certain confidence are returned.

        Args:
           user_id (str): id of the user for which an array of promising offers shall be returned
           confidence (float): confidence that the returned offers will show a profit

        Returns:
            offer_id (array): array of ids of offers that will show a profit for the specified user
               with a certain confidence
        """
        # get the user item dataframe
        user_item = self.full_user_item
        # get the cleaned profile dataframe
        clean_profile = self.clean_profile 
        
        # check whether the user has completed any offers
        if np.any(user_item.index == user_id):
            # make recommendation based on collaborative recommendation
            offer_id, predicted_yield = Recommender._collaborative_recommendation(user_item, user_id)
            # get the standard deviation of the user spendings
            sigma = clean_profile.loc[clean_profile["user_id"]==user_id, "stddev_transaction"]
        else:
            # make recommendation based on the most popular ads 
            offer_id, predicted_yield = Recommender._popular_recommendation(user_item) 
            # get the median standard deviation of all user spendings
            sigma = clean_profile["stddev_transaction"].median()
            
        
        # get the probability for a spending at least returning the investment of the ad
        probability = norm.cdf(predicted_yield, 0, sigma)
        
        # only return offers that return yield with a certain confidence
        offer_id = offer_id[probability > confidence]
            
        return offer_id
    
    @staticmethod
    def _clean_data(profile, portfolio, transcript):
        """
        takes the loaded profile, portfolio, and transcript dataframes and cleans them. This includes the 
        removal of unnecessary columns, one hot encoding of categorical columns, unpacking of nested columns 
        and categorization of numeric data. Furthermore some additional features as the average user spendings and
        the standard deviation are calculated and added to the user profile dataframe.

        Args:
            profile (DataFrame): informations about the users in the same format
                as supplied in the example folder
            portfolio (DataFrame): informations about the advertisments in the same 
                format as supplied in the example folder
            transcript (DataFrame): temporal informations about user actions in the same
                format as supplied in the example folder

        Returns:
            profile (DataFrame): profile data after cleaning and data engineering
            portfolio (DataFrame): portfolio data after cleaning and data engineering
            transcript (DataFrame): transcript data after cleaning and data engineering
        """
        
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
        
    @staticmethod
    def _user_item(clean_profile, clean_portfolio, clean_transcript):
        """
        calculates the user item matrix based on the average yield that each advertisement
        returns for each user        
        
        Args:
            clean_profile (DataFrame): informations about the users in the same format
                as supplied in the example folder
            clean_portfolio (DataFrame): informations about the advertisments in the same 
                format as supplied in the example folder
            clean_transcript (DataFrame): temporal informations about user actions in the same
                format as supplied in the example folder

        Returns:
            user_item (DataFrame): user item matrix as dataframe with the user ids as indices
                and offer ids as columns
        """
        
        # rename the cleaned dataframes
        profile = clean_profile
        portfolio = clean_portfolio
        transcript = clean_transcript
        
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
    def _matrix_factorization(user_item, latent_features=20, learning_rate=0.0001, iters=100):
        """
        performs matrix factorization on the sparse user-item matrix in order to fill the missing
        values and returns the user-item matrix after factorization

        Args:
            user_item (DataFrame): user item matrix as dataframe with the user ids as indices
                and offer ids as columns
            latent_features (int): number of latent features to use for the matrix factorization
            learning_rate (float): learning rate that scales the update step
            iters (int): number of iterations to update the user item matrix

        Returns:
            user_item (DataFrame): updated user item matrix after the matrix factorization
        """
        
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
            
        # get the user item matrix
        user_item_mat = user_mat @ offer_mat
        
        # get the user item dataframe
        user_item = pd.DataFrame(user_item_mat, index=user_item.index, columns=user_item.columns)
        
        return user_item
        
    @staticmethod
    def _collaborative_recommendation(user_item, user_id):
        """
        make a advertisement recommendation for a user on the basis of the user-item matrix after
        matrix factorization

        Args:
            user_item (DataFrame): user item matrix after matrix factorization as dataframe 
                with the user ids as indices and offer ids as columns
            user_id (str): id of the user to make a recommendation for

        Returns:
            offer_ids (array): sorted array with ids of the advertisement offers that are most
                promising to return yield
            predicted_yield (array): sorted array with the predicted yield for each offer in offer_ids
        """
        # get the offer ids
        offer_ids = user_item.columns.values
        
        # get the predicted yield for the user for each offer
        predicted_yield = user_item[user_item.index==user_id].values.flatten()
        
        # sort by the highest predicted yield
        idx = np.argsort(predicted_yield)[::-1]
        offer_ids = offer_ids[idx]
        predicted_yield = predicted_yield[idx]

        return offer_ids, predicted_yield
        
    @staticmethod
    def _popular_recommendation(user_item):
        """
        make a advertisement recommendation on the basis of the most popular advertisements

        Args:
            user_item (DataFrame): user item matrix after matrix factorization as dataframe 
                with the user ids as indices and offer ids as columns

        Returns:
            offer_ids (array): sorted array with ids of the advertisement offers that are most
                promising to return yield    
            median_yield (array): sorted array with the median yield of all users for each offer 
                in offer_ids
        """
        # get the offer ids
        offer_ids = user_item.columns.values
        # get the median yield of each offer
        median_yield = user_item.median().values
        
        # sort the offers and yield
        idx = np.argsort(median_yield)[::-1]
        offer_ids = offer_ids[idx]
        median_yield = median_yield[idx]
        
        return offer_ids, median_yield
        