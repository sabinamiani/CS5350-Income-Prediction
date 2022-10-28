import numpy as np
import xgboost as xgb
import pandas as pd

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

param = {'max_depth': 2, 'eta': 1, 'gamma': 1, 'objective': 'binary:logistic'}
# evallist  = [(dtest,'evals'), (dtrain,'train')]
num_round = 10
bst = xgb.train(param, dtrain, num_round)

prediction = bst.predict(dtest)

to_form = np.vstack((np.arange(1,prediction.shape[0]+1), prediction)).T
to_form = pd.DataFrame(to_form)
to_form = to_form.astype({0:'int'})
to_form.to_csv('p01.csv', index=False, header  = ['ID', 'Prediction']) 

# bst.save_model('0001.model')

