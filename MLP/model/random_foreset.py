from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import numpy as np

# hapt
# n_estimators: 90
# oob_score: True
# max_features: 23
# max_depth: 11
# min_samples_split: 2
# min_samples_leaf: 15
# splitter: best
# random state: 1


# x_train_file = [line.split(' ') for line in open('{:}HAPT/Train/X_train.txt'.format('../data/')).readlines()]
# y_train_file = [line for line in open('{:}HAPT/Train/Y_train.txt'.format('../data/')).readlines()]
# x_test_file = [line.split(' ') for line in open('{:}HAPT/Test/X_test.txt'.format('../data/')).readlines()]
# y_test_file = [line for line in open('{:}HAPT/Test/Y_test.txt'.format('../data/')).readlines()]

# x_train_data = np.array([list(map(float, line)) for line in x_train_file], dtype=np.float32)
# y_train_data = np.array(list(map(int, y_train_file)), dtype=np.long) - 1

# x_test_data = np.array([list(map(float, line)) for line in x_test_file], dtype=np.float32)
# y_test_data = np.array(list(map(int, y_test_file)), dtype=np.long) - 1

train_file = [line.split(',') for line in open('{:}/UJIndoorLoc/trainingData.csv'.format('../data/')).readlines()]
test_file = [line.split(',') for line in open('{:}/UJIndoorLoc/validationData.csv'.format('../data/')).readlines()]

x_train_data = np.array([list(map(float, line))[:520] for line in train_file], dtype=np.float32) / 100.0
y_train_data = np.array([int(line[522]) for line in train_file], dtype=np.long)

x_test_data = np.array([list(map(float, line))[:520] for line in test_file], dtype=np.float32) / 100.0
y_test_data = np.array([int(line[522]) for line in test_file], dtype=np.long)

if __name__ == '__main__':
    # 'min_samples_leaf':range(10,60,10) 'max_features':range(3,11,2)
    rf1 = RandomForestClassifier(oob_score=True, random_state=2)
    rf1.fit(x_train_data, y_train_data)
    print(rf1.oob_score_)
    y_pre = np.argmax(rf1.predict_proba(x_test_data), axis=1)
    print(metrics.accuracy_score(y_test_data, y_pre))
    '''
    param_test1 = {'n_estimators': range(10, 150, 10)}
    g_search1 = GridSearchCV(estimator=RandomForestClassifier(random_state=1),
                             param_grid=param_test1, scoring='accuracy', n_jobs=10)
    g_search1.fit(x_train_data, y_train_data)
    print(g_search1.best_params_)
    print(g_search1.best_score_)
    
    param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(10, 100, 10)}
    g_search2 = GridSearchCV(estimator=RandomForestClassifier(random_state=1, n_estimators=90),
                             param_grid=param_test2, scoring='accuracy', n_jobs=16)
    g_search2.fit(x_train_data, y_train_data)
    print(g_search2.best_params_)
    print(g_search2.best_score_)
    
    param_test3 = {'min_samples_leaf': range(10, 40, 5), 'min_samples_split': range(2, 10, 1)}
    g_search3 = GridSearchCV(estimator=RandomForestClassifier(random_state=1, n_estimators=90,
                                                              max_depth=11, min_samples_split=20),
                             param_grid=param_test3, scoring='accuracy', n_jobs=16)
    g_search3.fit(x_train_data, y_train_data)
    print(g_search3.best_params_)
    print(g_search3.best_score_)
    
    param_test4 = {'max_features': range(3, 31, 2)}
    g_search4 = GridSearchCV(estimator=RandomForestClassifier(random_state=1, n_estimators=90,
                                                              max_depth=11, min_samples_split=2,
                                                              min_samples_leaf=15),
                             param_grid=param_test4, scoring='accuracy', n_jobs=8)
    g_search4.fit(x_train_data, y_train_data)
    print(g_search4.best_params_)
    print(g_search4.best_score_)
    '''
    '''
    rf2 = RandomForestClassifier(n_estimators=90, max_depth=11, min_samples_split=2,
                                 min_samples_leaf=15, max_features=23, oob_score=True, random_state=1)
    rf2.fit(x_train_data, y_train_data)
    print(rf2.oob_score_)
    y_pre = np.argmax(rf2.predict_proba(x_test_data), axis=1)
    print(metrics.accuracy_score(y_test_data, y_pre))
    '''
