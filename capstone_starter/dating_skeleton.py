import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

#Create your df here:

df = pd.read_csv('~/Downloads/capstone_starter/profiles.csv')
pd.options.display.max_columns = 60

#list(df.columns.values)

df.income = df.income.replace(-1, np.nan)

#checking the distribution of the income data
df.income.hist(bins=50)
plt.ylabel('count')
plt.xlabel('income in US Dollars')
plt.title('histogram of income')
plt.show()

#age versus income average (median is interesting too)
df[['age','income']].groupby('age').mean().plot()
plt.ylabel('income')
plt.show()

#age distribution
plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()

#feature engineering
def mapEducation(row):
    if row in ['graduated from ph.d program','graduated from law school','graduated from med school']: return 5
    elif row in ['working on ph.d program','working on law school','working on med school','med school','law school']: return 4
    elif row in ['graduated from college/university','graduated from masters program','masters program']: return 3
    elif row in ['working on masters program','working on college/university','graduated from two-year college','college/university','two-year college']: return 2
    elif 'dropped' in str(row): return 1
    else: return 0
    
df['income_disclosed'] = df.income.apply(lambda x : 0 if np.isnan(x) else 1)
df['education_ranked'] = df.education.apply(mapEducation)
df['number_languages'] = df.speaks.apply(lambda x : str(x).count(',') + 1) 
df['drinks_ranked'] = df.drinks.map({'not at all' : 0, 'rarely' : 1, 'socially' : 2, 'often' : 3, 'very often' : 4, 'desperately' : 5})
df['drugs_ranked'] = df.drugs.map({'never' : 0, 'sometimes' : 1, 'often' : 2, np.nan : 0}) 
#np.nan = 0 for drugs is not so neat, but a hack here to avoid dropping to many rows later. 


#PART 2: Modeling 

#i'll only retain relevant columns to keep it neat
sample = df[['age','drugs_ranked','drinks_ranked','number_languages','education_ranked','income_disclosed','income']]

sample = sample.fillna(0)
features = sample[['age','drugs_ranked','drinks_ranked','number_languages','education_ranked']]
x = features.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=features.columns)

#CLASSIFICATION 
#can we predict income_disclosed with our given features? 

#Train Test Split
X_train, X_test, y_train , y_test = train_test_split(feature_data, sample['income_disclosed'], test_size=0.30, random_state=23)

#KNN
print('KNN')
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
predictions = clf.predict(X_test)
print('confusion Matrix:')
print(metrics.confusion_matrix(y_test,predictions))
print('---')

#SVC
print('SVC')
clf = SVC(C=1, kernel='rbf', gamma=0.001)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
predictions = clf.predict(X_test)
print('confusion Matrix:')
print(metrics.confusion_matrix(y_test,predictions))
print('---')


#REGRESSION
#can we predict income with our given features? 

#Train Test Split
X_train, X_test, y_train , y_test = train_test_split(feature_data, sample['income'], test_size=0.30, random_state=23)

# MLR
print('MLR')
reg = LinearRegression()
reg.fit(X_train, y_train)
print(reg.coef_)
print(reg.score(X_test, y_test))
print('-----')
#outcome is a disaster
#KNN Regressor
print('KNN Regressor')
reg = KNeighborsRegressor(n_neighbors=2)
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))