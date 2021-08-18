# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 19:24:05 2021

@author: dilip-k
"""

import pandas as pd
import numpy as np


Res_df = pd.read_excel('CBA Sheet Jan 2018 - Sep 2020 - Till E3.xlsx', parse_dates = ['Joining Date','Date'])

Res_df.head()

Res_df.shape

Res_df[Res_df['Date'] == "Jan'18"]['Annaul Salary'].mean()

Res_df[Res_df['Date'] == "Feb'18"]['Annaul Salary'].mean()

Res_df[Res_df['Date'] == "Mar'18"]['Annaul Salary'].mean()

Res_df[Res_df['Date'] == "Jan'19"]['Annaul Salary'].mean()

Res_df[Res_df['Date'] == "Feb'18"]['Annaul Salary'].mean()

Res_df[Res_df['Date'] == "Jan'20"]['Annaul Salary'].mean()

Res_df[Res_df['Date'] == "Sep'18"]['Annaul Salary'].mean()

Res_df[Res_df['Date'] == "Sep'19"]['Annaul Salary'].mean()

Res_df[Res_df['Date'] == "Sep'20"]['Annaul Salary'].mean()

Res_df.columns = ['EmpCode','Crncy','AnlSal','Conv','InUSD','MonUSD','Band','Subband','EmpLoc','ConGroup',
                 'ConGroup1','JoinDate','Customer','RelExpY','RelExpM','Skills','Date','Concat',
                 ]

res = Res_df[['EmpCode','AnlSal','Subband','RelExpY','RelExpM','Skills','Date']]

res.head()

res['Skills'].unique()

res['Subband'].unique()

res['Subband'] = res['Subband'].replace('E0.3','E1.1')
res['Subband'] = res['Subband'].replace('E0.2','E1.1')

res['Skills'].nunique()

res.isnull().sum()

res = res.dropna()

res.isnull().sum()

Skills_master = pd.read_excel('Unique Skills - updated.xlsx')

Skills_master.head()

Skills_master['Category'].unique()

Skills_master.shape

Skills_master['Category'] = Skills_master['Category'].replace('.net','.NET')
Skills_master['Category'] = Skills_master['Category'].replace('pm','PM')
Skills_master['Category'] = Skills_master['Category'].replace('sql','SQL')
Skills_master['Category'] = Skills_master['Category'].replace('ba','BA')
Skills_master['Category'] = Skills_master['Category'].replace('cloud','Cloud')


Skills_master.isnull().sum()

df = pd.merge(res,Skills_master,on="Skills")

df.head()

df.shape

df['Category'].unique()

df.isnull().sum()

df = df.dropna()

df['Category'].unique()

df.drop(columns=['Skills'],axis=1,inplace=True)

df.head()

df['EmpCode'].value_counts()

df_final = df.groupby(['EmpCode','Subband','RelExpY','RelExpM','Date','Category']).agg({'AnlSal':['max']})
df_final.columns = ['Salary']
df_final = df_final.reset_index()
df_final.head()

df_final.isnull().sum()

df_final[(df_final['RelExpY'] == '#') | (df_final['RelExpM'] == '#')].index

df_final.drop(df_final[(df_final['RelExpY'] == '#') | (df_final['RelExpM'] == '#')].index, inplace=True)

df_final['RelExpTotM'] = df_final['RelExpY'] * 12 + df_final['RelExpM']

df_final['RelExpTotM'].max()

bins = [0,24,72,120,180,216,300]
labels = [2,6,10,15,18,24]
df_final['Experience'] = pd.cut(df_final['RelExpTotM'],bins,labels=labels)

df_final.tail()

df_final['Experience'].unique()

df_final['EmpCode'].value_counts()

df_final[df_final['EmpCode'] == 49420279]

df_final.info()

df_sum = df_final.groupby(['EmpCode','Subband','Date','Category','Salary']).agg({'RelExpTotM':['max']})
df_sum.columns = ['TotExp']
df_sum = df_sum.reset_index()
df_sum.head()

bins = [0,24,72,120,180,216,300]
labels = [2,6,10,15,18,24]
df_sum['Experience'] = pd.cut(df_sum['TotExp'],bins,labels=labels)

df_sum.drop(columns=('EmpCode'),axis=1,inplace=True)

df_sum.head()

df_sum.isnull().sum()

df_sum['Category'].value_counts()

df_sum[(df_sum['Date'] == "Apr'18") & (df_sum['Category'] == 'ETL')]

df_sum['TotExp'].unique()

df_sum['TotExp'].max()

df_sum[(df_sum['Date'] == "Apr'18") & (df_sum['Category'] == 'ETL') & (df_sum['Subband'] == 'E1.1') 
       & (df_sum['Experience'] == 6)]['Salary'].mean()

df_sum['Salary'] = df_sum['Salary'].astype(int)

df_sum.isnull().sum()

df_sum.groupby(['Date','Category','Subband','Experience']).agg({'Salary':['mean']})

df_sum[(df_sum['Subband'] == 'E1.1') & (df_sum['Date'] == "Apr'18") & (df_sum['Category'] == ".NET")]['Experience'].unique()

df_sum[(df_sum['Subband'] == 'E1.1') & (df_sum['Date'] == "Apr'18") & (df_sum['Category'] == ".NET") & 
      (df_sum['Experience'] == '18')]


df_skill_sal = df_sum.groupby(['Subband','Date','Category','Experience']).agg({'Salary':['mean']})
df_skill_sal.columns = ['Salary']
df_skill_sal = df_skill_sal.reset_index()
df_skill_sal.head()


df_skill_sal[df_skill_sal['Date'] == "Apr'18"]


df_skill_sal.isnull().values.any()

df_skill_sal.isnull().sum()

df_skill_sal = df_skill_sal.dropna()

df_skill_sal.head()

df_skill_sal['Salary'] = df_skill_sal['Salary'].astype(int)

df_skill_sal.shape

df_skill_sal[df_skill_sal['Category'] == '.NET']

df_skill_sal[(df_skill_sal['Category'] == '.NET') & (df_skill_sal['Subband'] == 'E1.1') & (df_skill_sal['Experience'] == 2)]

df_skill_sal[(df_skill_sal['Category'] == '.NET') & (df_skill_sal['Subband'] == 'E1.1') 
             & (df_skill_sal['Experience'] == 2)]['Salary'].mean()

df_X = df_skill_sal.groupby(['Subband','Category','Experience']).agg({'Salary':['mean']})
df_X.columns = ['Salary']
df_X = df_X.reset_index()
df_X.head()

df_X = df_X.dropna()

df_X.head()

df_X['Salary'] = df_X['Salary'].astype(int)

df_X[df_X['Category'] == 'SQL']

data = df_X.copy()

data['Subband'].unique()

X = data.drop(['Salary'], axis = 1)
y = data['Salary']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

import category_encoders as ce
Band_encoder = ce.OrdinalEncoder(cols=['Subband'],return_df=True)
X_train = Band_encoder.fit_transform(X_train)
X_test = Band_encoder.transform(X_test)

X_train.head()

Cat_encoder = ce.BinaryEncoder(cols=['Category'],return_df = True)
X_train = Cat_encoder.fit_transform(X_train)
X_test = Cat_encoder.transform(X_test)

X_train.head()

cols = X_train.columns
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(X_train)

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])

X_test = pd.DataFrame(X_test, columns=[cols])

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train,y_train)

print(lin_reg.intercept_)

pred = lin_reg.predict(X_test)

from sklearn import metrics
from sklearn.model_selection import cross_val_score

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)


from sklearn.linear_model import Ridge

model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1, 
              precompute=True, 
#               warm_start=True, 
              positive=True, 
              selection='random',
              random_state=42)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=1000)
rf_reg.fit(X_train, y_train)

test_pred = rf_reg.predict(X_test)
train_pred = rf_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)


print('Pickle dump start')

#Dumping the model object
import pickle
pickle.dump(rf_reg, open('RF_model.pkl','wb'))
pickle.dump(Band_encoder, open('Band_encoder.pkl','wb'))
pickle.dump(Cat_encoder, open('Cat_encoder.pkl','wb'))
pickle.dump(scaler, open('scaler.pkl','wb'))


print('Pickle dump end')
