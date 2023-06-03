article_path = "./data/articles.csv"
user_path = "./data/customers.csv"
train_path = "./data/transactions_train.csv"
sample_path = "./data/sample.csv"
model_path = "./data/model_pth/"

item_embedding_features = {
    'article_id_unique_id': {
        "bucket_size": 105542,
        "dim": 64
    },
    'product_code_unique_id': {
        "bucket_size": 47224,
        "dim": 32
    },
    'prod_name_unique_id': {
        "bucket_size": 45875,
        "dim": 32
    }, 
    'product_type_no_unique_id': {
        "bucket_size": 132,
        "dim": 8
    },
    'product_type_name_unique_id': {
        "bucket_size": 131,
        "dim": 4,
    }, 
    'colour_group_code_unique_id': {
        "bucket_size": 50,
        "dim": 4
    }, 
    'colour_group_name_unique_id': {
        "bucket_size": 50,
        "dim": 4
    }, 
    'department_no_unique_id': {
        "bucket_size": 299,
        "dim": 8
    }, 
    'department_name_unique_id': {
        "bucket_size": 250,
        "dim": 8
    }
}

item_onehot_features = {
    'product_group_name_unique_id': 19,
    'graphical_appearance_no_unique_id': 30, 
    'graphical_appearance_name_unique_id': 30,
    'perceived_colour_value_id_unique_id': 8, 
    'perceived_colour_value_name_unique_id': 8, 
    'perceived_colour_master_id_unique_id': 20, 
    'perceived_colour_master_name_unique_id': 20, 
    'index_code_unique_id': 10, 
    'index_name_unique_id': 10, 
    'index_group_no_unique_id': 5, 
    'index_group_name_unique_id': 5,
    'section_no_unique_id': 57,
    'section_name_unique_id': 56, 
    'garment_group_no_unique_id': 21, 
    'garment_group_name_unique_id': 21
}

user_embedding_features = {
    'customer_id_unique_id': {
        "bucket_size": 1371980,
        "dim": 64
    },
    'postal_code_unique_id': {
        "bucket_size": 352899,
        "dim": 64
    }
}

user_onehot_features = {
    #'FN_unique_id': 2, 
    #'Active_unique_id': 2, 
    #'club_member_status_unique_id': 4, 
    #'fashion_news_frequency_unique_id': 5, 
    #'age_unique_id': 85
}
