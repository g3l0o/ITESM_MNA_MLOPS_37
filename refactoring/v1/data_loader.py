import pandas as pd
def load_data(path):
    df = pd.read_csv(path)

    rename_mapper = {column:column.strip() for column in df.columns}
    new_df = df.rename(rename_mapper,axis=1)

    cluster_ids = new_df['Category ID'].unique()
    label_mapper = {cluster_ids[i]:i for i in range(len(cluster_ids))}
    new_df['Category ID'] = new_df['Category ID'].map(label_mapper)
    return df