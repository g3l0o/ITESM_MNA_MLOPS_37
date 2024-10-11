import pandas as pd
import sys


def load_standarize_data(filepath):
    df = pd.read_csv(filepath)
    rename_mapper = {column: column.strip().lower().replace(" ", "_") for column in df.columns}
    new_df = df.rename(rename_mapper, axis=1)

    cluster_ids = new_df['category_id'].unique()
    label_mapper = {cluster_ids[i]: i for i in range(len(cluster_ids))}
    new_df['category_id'] = new_df['category_id'].map(label_mapper)
    return new_df

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_file = sys.argv[2]
    data = load_standarize_data(data_path)
    data.to_csv(output_file, index=False)
