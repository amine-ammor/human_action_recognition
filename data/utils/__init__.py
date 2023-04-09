"""
    module containing the main routines to fetch the dataset used for the project
"""
from . import remote_data_fetcher_mod

def get_data_fetcher(dataset="kth_dataset"):
    list_datasets = ["kth_dataset"]

    if dataset == "kth_dataset":
        website_url="https://www.csc.kth.se/cvap/actions"
        expression=".zip"
        folder_dst="data/inputs/kth_dataset"
    else:
        ValueError(f"dataset must be among this list {*list_datasets,}")

    remote_data_folder = remote_data_fetcher_mod.RemoteDataFetcher(website_url,expression,
                                                                   folder_dst)
    remote_data_folder.download()
    remote_data_folder.unzip()
    remote_data_folder.clean()
