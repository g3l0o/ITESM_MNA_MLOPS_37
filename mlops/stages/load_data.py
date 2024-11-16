import pandas as pd
import yaml
import argparse

def load_data(config_path):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    df = pd.read_csv(config['load_data']['raw'])

    
    df.columns = df.columns.str.strip()

    cluster_ids = df['Category ID'].unique()
    label_mapper = {cluster_ids[i]: i for i in range(len(cluster_ids))}
    df['Category ID'] = df['Category ID'].map(label_mapper)

    df.to_csv(config['load_data']['loaded_data_dir'], index=False)

if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    load_data(config_path=args.config)