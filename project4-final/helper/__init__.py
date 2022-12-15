
from .features import generate_features
from .preprocess import merge_data, process_data, read_dataframes

def run_pipeline(
    df_beer_addr: str,
    df_brewery_addr: str,
    df_state_addr: str
):
    """
    Run pipeline for generating 
    """

    # Read all datasets and preprocess them
    data_dict = read_dataframes(df_beer_addr, df_brewery_addr, df_state_addr)

    df_bb = merge_data(data_dict["beer"], data_dict["brewery"], data_dict["state"])

    df_bb = process_data(df_bb)

    features_dict, encoder_dict, y = generate_features(df_bb)

    