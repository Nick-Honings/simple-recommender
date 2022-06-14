import sys
import threading
from simple_recommender import SimpleRecommender
from helpers import UIHelper

data_config = {
    'review_data_path': '../dataset/Video_Games_k5_cluster.json.gz',
    'metadata_data_path': '../dataset/meta_Video_Games.json.gz',
    'review_model_path': '../dataset/processed/average_product_rating_k5.pkl',
    'metadata_model_path': '../dataset/processed/__metadata.pkl'
}

# always_train= True|False -> whether to always train dataset on startup
if __name__ == '__main__':
    recommender = SimpleRecommender(data_config)
    completion_event = threading.Event()
    print("Please wait a moment as I'm training myself")
    thread = threading.Thread(target=recommender.train, args=[completion_event])
    thread.start()
    completion_event.wait()
    print("Training completed.")
    UIHelper.clear_console()

    average_ratings = recommender.get_averages()
    metadata = recommender.get_metadata()
    video_game_title = input("What is your favourite video game: ")

    # Todo: get_recommendations
    raw_recommendations = recommender.get_recommendations(video_game_title).tolist()
    video_recommendation = recommender.construct_recommended_video_games(raw_recommendations)
    print("I found these products in my collection:")
    for video in video_recommendation:
        print(video)







