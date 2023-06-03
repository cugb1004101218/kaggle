import pandas as pd
import config

def gen_unique_id(df):
    d = {}
    for col in df.columns:
        df[col+"_unique_id"] = pd.factorize(df[col])[0]
    for col in df.columns:
        print(col, df[col].dtype, len(df[col].unique()))
        d[col] = len(df[col].unique())
    print(d)
    return df

def gen_train_data(articles_df, user_df, train_df):
    train_df = train_df.merge(articles_df, on="article_id")
    train_df = train_df.merge(user_df, on="customer_id")
    return train_df

if __name__ == '__main__':
    articles_df = pd.read_csv(config.article_path)
    articles_df = gen_unique_id(articles_df)
    user_df = pd.read_csv(config.user_path)
    user_df = gen_unique_id(user_df)
    #train_df = pd.read_csv(config.train_path)
    #train_df = gen_train_data(articles_df, user_df, train_df)
    #train_df.to_csv(config.sample_path)