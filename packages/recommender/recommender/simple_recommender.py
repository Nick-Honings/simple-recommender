import threading
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from helpers import DataPreprocessors

import pandas as pd

# Returns a new dataframe containing average rating_column per product.
from video_game import VideoGame


def calculate_average_product_rating(dataframe, id_column_name, rating_column):
    if id_column_name in dataframe.columns and rating_column in dataframe.columns:
        averages = dataframe.groupby(id_column_name)[rating_column].mean().reset_index().sort_values('overall',
                                                                                                     ascending=False)
        averages.sort_index(inplace=True)
        return averages
    raise Exception("Dataframe does not contain specified columns")


class SimpleRecommender:

    def __init__(self, config):
        self.__config = config
        self.__metadata = None
        self.__reviews = None
        self.__averages = None
        self.__is_trained = False

    def train(self, completion_event):
        # Try to read data from disk
        try:
            self.__averages = pd.read_pickle(self.__config['review_model_path'])
            print("Found existing averages file, using that...")
        except FileNotFoundError:
            pass

        try:
            self.__metadata = pd.read_pickle(self.__config['metadata_model_path'])
            print("Found existing metadata file, using that...")
        except FileNotFoundError:
            pass

        if self.__reviews is None:
            print("Processing review file...")
            self.__reviews = self.__load_data('review_data_path')
        if self.__metadata is None:
            print("Processing metadata file...")

            self.__metadata = self.__load_data('metadata_data_path')[0:1000]  # Whole dataset is way too large.
            self.__metadata.to_pickle(self.__config['metadata_model_path'])

        if self.__averages is None:
            print("Calculating average product ratings...")
            self.__averages = calculate_average_product_rating(self.__reviews, 'asin', 'overall')
            self.__averages.to_pickle(self.__config['review_model_path'])

        # Todo: Fix this
        self.__is_trained = True
        # Notify that the training is completed
        completion_event.set()
        print("Training completed...")

    def __load_data(self, config_index):
        return DataPreprocessors.get_dataframe(
            DataPreprocessors.read_data_from_gzip(self.__config[config_index])
        )

    # Get average ratings. If model is already trained, return from disk or memory. Throws error if model is not trained
    def get_averages(self):
        if self.__is_trained:
            if self.__averages is not None:
                return self.__averages
            else:
                return pd.read_pickle(self.__config[2])
        else:
            raise Exception("Please train a model first")

    # Get product metadata. If model is already trained, return from disk or memory. Throws error if model is not
    # trained
    def get_metadata(self):
        if self.__is_trained:
            if self.__metadata is not None:
                return self.__metadata
            else:
                return pd.read_pickle(self.__config[3])
        else:
            raise Exception("Please train a model first")

    def find_similar_product_descriptions(self):
        tfidf = TfidfVectorizer(stop_words='english')
        # Construct the required TF-IDF matrix by fitting and transforming the data
        tfidf_matrix = tfidf.fit_transform(self.__metadata['description'])
        return tfidf_matrix

    # Todo: Save recommendations to file, so a next question with same input is faster.
    def get_recommendations(self, title):
        title = title.lower()
        if len(self.__metadata['description']) == 0:
            raise Exception("No description found in dataset")

        self.__metadata['description'] = self.__metadata['description'].astype(str)
        similarity_matrix = self.find_similar_product_descriptions()
        cosine_sim = linear_kernel(similarity_matrix, similarity_matrix)
        # Create a reverse lookup list
        indices = pd.Series(self.__metadata.index, index=self.__metadata['title'].str.lower()).drop_duplicates()

        # Find index in list
        idx = None
        for index in indices.index:
            if index.find(title) != -1:
                idx = indices[index]
                break

        # Get the pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get the 10 most similar, excluding the first, which is a reference to itself
        sim_scores = sim_scores[1:11]
        video_indices = [i[0] for i in sim_scores]
        return self.__metadata['title'].iloc[video_indices]

    def construct_recommended_video_games(self, recommendations):
        video_recommendations = []
        for r in recommendations:
            product = self.__metadata.loc[self.__metadata['title'] == r]
            v = VideoGame()
            if len(product['title']) != 0:
                v.title = product['title'].iloc[0]
            else:
                v.title = "Not Found"
            if len(product['asin']) != 0:
                v.asin_id = product['asin'].iloc[0]
            else:
                v.asin_id = "Not Found"
            video_recommendations.append(v)

        for i in video_recommendations:
            product = self.__averages.loc[self.__averages['asin'] == i.asin_id]
            if len(product) != 0:
                i.average_rating = round(product['overall'].item(), 2)

        return video_recommendations
