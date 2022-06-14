class VideoGame:

    def __init__(self):
        self.asin_id = None
        self.title = None
        self.average_rating = "No rating_column found"

    def __str__(self) -> str:
        return "Title: {0} \n Average Rating: {1} \n".format(self.title, self.average_rating)
