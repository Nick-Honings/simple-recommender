import gzip
import json
import pandas as pd
import os


class DataPreprocessors:

    # Read .gz file into memory
    @staticmethod
    def read_data_from_gzip(file_path):
        output = []
        with gzip.open(file_path) as file:
            for line in file:
                output.append(json.loads(line.strip()))
        return output

    # Get dataframe, add index and fill NaN fields
    @staticmethod
    def get_dataframe(data):
        output = pd.DataFrame.from_dict(data)
        index = pd.Index(range(0, len(output), 1))
        output = output.set_index(index)
        return output.fillna('')


class UIHelper:

    @staticmethod
    def clear_console():
        command = 'clear'
        if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
            command = 'cls'
        os.system(command)
