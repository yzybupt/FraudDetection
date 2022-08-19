from sklearn.tree import DecisionTreeClassifier

import Evaluate
import DataLoader

from imblearn.ensemble import RUSBoostClassifier
gap = 2



for test_year in range(2003,2009):
    print("-----------------------------Start!----------------------------------")
    print(f"Test year: {test_year}, Training Period: {1991}---{test_year - gap}")

    """Load the data from csv file"""
    train_data = DataLoader.DataLoader("./uscecchini28.csv", 1991, test_year - gap)
    test_data = DataLoader.DataLoader("./uscecchini28.csv", test_year, test_year)

    """Parse the data into train and test"""
    train_label = train_data.label
    train_feature = train_data.feature
    train_serial_fraud = train_data.serial_fraud

    test_label = test_data.label
    test_feature = test_data.feature
    test_serial_fraud = test_data.serial_fraud

    """Change serial fraud case in training data as 0"""
    overlapped = 0
    num_frauds = 0
    for index in range(len(train_serial_fraud)):
        if train_label[index] == 1:
            num_frauds = num_frauds + 1

        if train_label[index] == 1 and train_serial_fraud[index] in test_serial_fraud:
            overlapped = overlapped + 1
            train_label[index] = 0
    print(f"There are {overlapped} overlapped cases (i.e., change fraud label from 1 to 0)")


    base = DecisionTreeClassifier(min_samples_leaf=5)
    rusbooster = RUSBoostClassifier(base_estimator=base,n_estimators=3000,random_state=0, sampling_strategy= 1.0, learning_rate= 0.1)
    rusbooster.fit(train_feature, train_label)
    y_prob = rusbooster.predict_proba(test_feature)
    results = Evaluate.evaluate(y_prob, test_label, 0.01)
    print(f"AUC:{results.AUC}")
    print(f"NDCG:{results.NDCG}")
    print(f"Sensitivity:{results.sensitivity}")
    print(f"Precision:{results.precision}")
    print(f"BAC:{results.BAC}")
    print("-----------------------------End!----------------------------------")
