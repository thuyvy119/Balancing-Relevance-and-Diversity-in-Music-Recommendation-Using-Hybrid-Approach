def check_user_type(user_id, df):
    return user_id in df["user_id"].values

def get_qualified_users(train_data, test_data, min_test_interaction=1):
    train_users = set(train_data['uid'].unique())
    test_users = set(test_data['uid'].unique())

    qualified_users = list(train_users.intersection(test_users))
    user_test_counts = test_data['uid'].value_counts()
    qualified_users = [user for user in qualified_users if user_test_counts.get(user, 0) >= min_test_interaction]
    return qualified_users
