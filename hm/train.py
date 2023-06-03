import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import config
import models
import torch.nn.functional as F

def gen_feature_tensor(df, embedding_features, onehot_features):
    feature_tensors = {}
    for feature in embedding_features:
        bucket_size = embedding_features[feature]["bucket_size"]
        dim = embedding_features[feature]["dim"]
        emb = torch.nn.Embedding(bucket_size, dim)
        print(feature,df[feature])
        tensor = torch.tensor(df[feature])
        feature_tensors[feature] = emb(tensor)

    for feature in onehot_features:
        tensor = torch.tensor(df[feature])
        print(feature, tensor)
        feature_tensors[feature] = F.one_hot(tensor)
    
    emb = torch.cat(tuple(feature_tensors.values()), dim = 1)
    print(emb.shape)
    return emb.float()

def gen_features_and_label(train_df):
    item_embedding = gen_feature_tensor(train_df, config.item_embedding_features, config.item_onehot_features)
    user_embedding = gen_feature_tensor(train_df, config.user_embedding_features, config.user_onehot_features)
    feature_embedding = torch.concat((item_embedding, user_embedding), dim=1)
    print(feature_embedding.shape)
    label = torch.tensor(train_df["price"])
    label = label.reshape(-1,1)
    print(label.shape)
    return feature_embedding.float(), label.float()

def train_epoch(df, model):
    feature_embedding, label = gen_features_and_label(df)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr= 0.01 , momentum= 0.5)
    print(feature_embedding.shape, label.shape)
    for i in range(10):
        optimizer.zero_grad()
        output = model(feature_embedding)
        loss = criterion(output, label)
        loss.backward(retain_graph=True)
        optimizer.step()
        print(loss.item())

def gen_unique_id(df):
    for col in df.columns:
        df[col+"_unique_id"] = pd.factorize(df[col])[0]
    for col in df.columns:
        print(col, df[col].dtype, len(df[col].unique()))
    return df

if __name__ == '__main__':
    train_df = pd.read_csv(config.sample_path, chunksize=100000)
    model = models.DNN((-1, 606))
    count = 0
    for df in train_df:
        train_epoch(df, model)
        print("lines:",count*100000)
        count += 1
    
    torch.save(model, config.model_path + "v1.pt")
