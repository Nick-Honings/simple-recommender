import gzip
import json
import threading

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from video_game import VideoGame



# https://www.rainforestapi.com/docs/product-data-api/parameters/product
# https://www.datacamp.com/tutorial/recommender-systems-python
# https://towardsdatascience.com/whats-in-a-word-da7373a8ccb

def read_data(file_path):
    output = []
    with gzip.open(file_path) as file:
        for line in file:
            output.append(json.loads(line.strip()))
    return output


def get_dataframe(data):
    output = pd.DataFrame.from_dict(data)
    index = pd.Index(range(0, len(output), 1))
    output = output.set_index(index)
    return output.fillna('')


# Returns a new dataframe containing average rating_column per product. C1=product, rating_column=rating_column
def get_average_product_rating(dataframe, product_id, rating):
    if product_id in dataframe.columns and rating in dataframe.columns:
        return dataframe.groupby(product_id)[rating].mean().reset_index().sort_values('overall', ascending=False)
    raise Exception("Dataframe does not contain specified columns")


# Find product by id in specified dataset
# Todo: Error handling
def find_product_by(dataset, column, value):
    return dataset.loc[dataset[column] == value]


# Expects description column to be present
# Matrix does contain weird & non-english words, this does need a filter.
def find_similar_product_descriptions(dataset):
    # Define a TF-IDF Vectorizer object. Remove all english stop words
    tfidf = TfidfVectorizer(stop_words='english')
    # Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(dataset['description'])
    return tfidf_matrix


def get_recommendations(title, dataset):
    # convert all objects in column to string to prep them for processing
    title = title.lower()
    dataset['description'] = dataset['description'].astype(str)
    similarity_matrix = find_similar_product_descriptions(dataset)
    cosine_sim = linear_kernel(similarity_matrix, similarity_matrix)
    # Create a reverse lookup list
    indices = pd.Series(dataset.index, index=dataset['title'].str.lower()).drop_duplicates()
    # print(indices.index)
    # Get the index of the game that matches the title

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
    return dataset['title'].iloc[video_indices]


review_data_path = "../dataset/Video_Games_k5_cluster.json.gz"
metadata_data_path = "../dataset/meta_Video_Games.json.gz"
review_model_path = "../dataset/processed/average_product_rating_k5.pkl"
metadata_model_path = "../dataset/processed/metadata.pkl"

training_completed = threading.Event()


# Train dataset and save it to a file to be used as a model
def train(config):
    # Get dataframe containing __reviews
    reviews = get_dataframe(read_data(review_data_path))
    # Get average rating_column per product. If no rating_column is found for specific product, this means the dataset
    # is not large enough or a game does not have 5 __reviews, in which the average review will statistically be invalid
    averages = get_average_product_rating(reviews, 'asin', 'overall')
    averages.sort_index(inplace=True)

    averages.to_pickle(review_model_path)

    # Get review __metadata, slice it otherwise it is way too much to process.
    meta = get_dataframe(read_data(metadata_data_path)[0:20000])
    meta.to_pickle(metadata_model_path)
    training_completed.set()

train_yourself = False

if __name__ == '__main__':
    if train_yourself:
        print("Please wait a moment as I'm training myself...")
        thread = threading.Thread(target=train)
        thread.start()
        training_completed.wait()
        print("Training done...")

    average_ratings = pd.read_pickle(review_model_path)
    metadata = pd.read_pickle(metadata_model_path)

    videogame_title = input("What is your favourite video game: ")
    recommendations = get_recommendations(videogame_title, metadata).tolist()

    # Todo: sort recommendation on average review level 1. Extract asin from __metadata list -> find by title value 2.
    #  Get review score from list -> find by asin 3. Sort recommendation output from high to low. 4. Display output
    #  5. Done. 6. SAve model to file to be used later. 7. Create interface or rest api to allow variable input 6. If
    #  we'd want to make it fancy, caluclate similarity scores of every title and base search results on that instead
    #  of string matching 7. Superfancy -> multithreaded

    recommendation_ids = []
    video_recommendations = []
    for r in recommendations:
        # Could be optimised by looking for id
        product = find_product_by(metadata, 'title', r)
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

    recommendation_ratings = []
    # Might not find every id since there is not an average rating_column for every id
    for i in video_recommendations:
        # Find the average rating_column in the dataset generated before
        product = find_product_by(average_ratings, 'asin', i.asin_id)

        if len(product) != 0:
            i.average_rating = round(product['overall'].item(), 2)
            recommendation_ratings.append(product['overall'])

    for v in video_recommendations:
        print(v.__str__())
