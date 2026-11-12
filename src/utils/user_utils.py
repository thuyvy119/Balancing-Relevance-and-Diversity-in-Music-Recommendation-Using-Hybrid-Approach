def check_user_type(user_id, df):
    return user_id in df["user_id"].values


def build_ground_truth(train_data, test_data, warm_only=True):
    tr = train_data[train_data["label"] > 0] if "label" in train_data else train_data
    te = test_data[test_data["label"] > 0] if "label" in test_data else test_data
    seen = tr.groupby("uid")["iid"].apply(set).to_dict()
    warm = set(tr["iid"])
    gt = {}
    for uid, g in te.groupby("uid"):
        items = set(g["iid"]) - seen.get(uid, set())
        if warm_only:
            items &= warm
        if items:
            gt[uid] = items
    return gt


def get_qualified_users(gt, pool_users, min_test_interaction=1):
    # Users that have >= min_test_interaction ground-truth items AND a CF candidate pool
    return sorted(u for u, g in gt.items() if u in pool_users and len(g) >= min_test_interaction)