import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score

# attributes = ['age','workclass','fnlwgt','education','education.num','marital.status',
#     'occupation','relationship','race','sex','capital.gain','capital.loss',
#     'hours.per.week','native.country','income>50K']
# numerical_ats = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week', 'income>50K']
convert_these = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

# convert feature data into numerical data -- label encoding 
df_train = pd.read_csv('train_final.csv')
df_test = pd.read_csv('test_final.csv')

for i in convert_these: 
    df_train[i] = df_train[i].astype('category')
    df_train[i] = df_train[i].cat.codes

    df_test[i] = df_test[i].astype('category')
    df_test[i] = df_test[i].cat.codes

# print(df_train)
# print(df_train.dtypes)

X_train = df_train.loc[:, df_train.columns != 'income>50K']
# print(X_train)
y_train = df_train['income>50K']
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=df_test.loc[:, df_test.columns != 'ID'])

best = 0

for md in {2, 3, 4, 5, 6, 7, 8} :
    for lr in {0.1, 0.3, 0.7, 0.9, 1} :
        # for gamma in {0.1, 0.3, 0.7, 0.9, 1} : 
            param = {'max_depth': md, 'eta': lr, 'objective': 'binary:hinge', 'eval_metric':'auc'}
            num_round = 10

            # # cross validation testing and output 
            # print('max depth=',md,' lr=',lr)
            # eval_hist = xgb.cv(param, dtrain, num_round, nfold=5, metrics={'auc'}, seed=0)
            # print(eval_hist)

            # training score evaluation and output 
            bst = xgb.train(param, dtrain, num_round)

            # compute score on training data 
            train_pred = bst.predict(dtrain)
            score = accuracy_score(y_train,train_pred)
            if score > best:
                best = score
                print('max depth=',md,' lr=',lr,' score=',score,' BEST!')
            else :
                print('max depth=',md,' lr=',lr,' score=',score)

# # train and save the testing label into csv file 
# bst = xgb.train(param, dtrain, num_round)

# prediction = bst.predict(dtest)

# to_form = np.vstack((np.arange(1,prediction.shape[0]+1), prediction)).T
# to_form = pd.DataFrame(to_form)
# to_form = to_form.astype({0:'int'})
# to_form.to_csv('p05_hinge_auc.csv', index=False, header  = ['ID', 'Prediction'])
