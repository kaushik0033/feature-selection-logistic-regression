# --------------
# Importing necessary libraries
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Load the data
#Loading the Spam data from the path variable for the mini challenge
#Target variable is the 57 column i.e spam, non-spam classes 
df=pd.read_csv(path)
df.describe()
df.info()
df.shape
# Overview of the data
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.3,random_state =42)
reg=LogisticRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
reg_score=reg.score(X_test,y_test)
print(reg_score)
pprint(classification_report(y_test,y_pred))

df1=df.copy()
corr_mat=df1.iloc[:,:-1].corr().abs()
temp=corr_mat.where(np.triu(np.ones(corr_mat.shape),k=1).astype(np.bool))
drop_cols=[c for c in temp.columns if any(temp[c]>0.75)]
df1.drop(drop_cols,axis=1,inplace=True)

X1=df1.iloc[:,:-1]
y1=df1.iloc[:,-1]
X_train1,X_test1,y_train1,y_test1=train_test_split(X1,y1,test_size = 0.3,random_state =42)
reg1=LogisticRegression()
reg1.fit(X_train1,y_train1)
y_pred1=reg1.predict(X_test1)
reg1_score=reg1.score(X_test1,y_test1)
print(reg1_score)
pprint(classification_report(y_test1,y_pred1))

chi_test=SelectKBest(score_func=chi2,k=30)
X_train_chi=chi_test.fit_transform(X_train,y_train)
X_test_chi=chi_test.transform(X_test)
#Dividing the dataset set in train and test set and apply base logistic model
reg_chi=LogisticRegression()
reg_chi.fit(X_train_chi,y_train)
y_pred_chi=reg_chi.predict(X_test_chi)
reg_chi_score=reg_chi.score(X_test_chi,y_test)
print(reg_chi_score)
pprint(classification_report(y_test,y_pred_chi))
# Calculate accuracy , print out the Classification report and Confusion Matrix.
Anova_test=SelectKBest(score_func=f_classif,k=55)
X_train_anv=Anova_test.fit_transform(X_train,y_train)
X_test_anv=Anova_test.transform(X_test)
reg_anv=LogisticRegression()
reg_anv.fit(X_train_anv,y_train)
y_pred_anv=reg_anv.predict(X_test_anv)
reg_anv_score=reg_anv.score(X_test_anv,y_test)
print(reg_anv_score)
pprint(classification_report(y_test,y_pred_anv))

scalar=StandardScaler()
X_train_scale=scalar.fit_transform(X_train)
X_test_scale=scalar.transform(X_test)
pca=PCA(n_components=30,random_state=42)
X_train_pca=pca.fit_transform(X_train_scale)
X_test_pca=pca.transform(X_test_scale)

reg_pca=LogisticRegression()
reg_pca.fit(X_train_pca,y_train)
y_pred_pca=reg_pca.predict(X_test_pca)
reg_pca_score=reg_pca.score(X_test_pca,y_test)
print(reg_pca_score)
pprint(classification_report(y_test,y_pred_pca))
# Copy df in new variable df1


# Remove Correlated features above 0.75 and then apply logistic model


# Split the new subset of data and fit the logistic model on training data


# Calculate accuracy , print out the Classification report and Confusion Matrix for new data


# Apply Chi Square and fit the logistic model on train data use df dataset



# Calculate accuracy , print out the Confusion Matrix 


# Apply Anova and fit the logistic model on train data use df dataset



# Calculate accuracy , print out the Confusion Matrix 


# Apply PCA and fit the logistic model on train data use df dataset

   

# Calculate accuracy , print out the Confusion Matrix 


# Compare observed value and Predicted value




