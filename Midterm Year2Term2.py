from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plot
import numpy as np
 
File_Path = 'D:/Data/'
File_Name = 'car_data.csv'
df = pd.read_csv(File_Path + File_Name)
#placeMissingValue
df.drop(columns=['User ID'], inplace=True)
df.dropna(inplace=True)
AS_mean = df['AnnualSalary'].mean()
df['AnnualSalary'].fillna(AS_mean, inplace=True)
#Encoding
encoders = []
for i in range(0, len(df.columns) - 1):
    enc = LabelEncoder()
    df.iloc[:, i] = enc.fit_transform(df.iloc[:, i])
    encoders.append(enc)
x = df.iloc[:, 0:3]
y = df['Purchased']
 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)
param_Grid = {
    'n_estimators': [25,50,100,150],
    'max_features': ['sqrt','log2',None],
    'criterion':['gini','entropy'],
    'max_depth':[3,6,9],
    'max_leaf_nodes':[3,6,9]
    }
#Create model
forest = GridSearchCV(RandomForestClassifier(), param_grid= param_Grid)
forest.fit(x_train,y_train)
#Vertify model
score_test = forest.score(x_test,y_test)
print('Accuracy :', '{:.2f}'.format(score_test))
Best_Parameter = forest.best_params_
model = DecisionTreeClassifier(criterion='entropy')
model.fit(x,y)
feature = x.columns.tolist()
Data_class = y.tolist()
plt.figure(figsize=(25,20))
_ = plot_tree(model,
              feature_names= feature,
              class_names = Data_class,
              label='all',
              impurity=True,
              precision=3,
              filled=True,
              rounded=True,
              fontsize=16)
plt.show()
#Predict data
y_pred = model.predict(x_test)
#Make Report
Label = y_train.unique()
Confu = confusion_matrix(y_test, y_pred)
CM_view = ConfusionMatrixDisplay(confusion_matrix = Confu,
                                 display_labels= Label)
CM_view.plot()
plot.show()
print(f'Accuracy_test : {model.score(x_test, y_test):.2f}')
 
import seaborn as sns
Feature_imp = model.feature_importances_
feature_names = ['Gender','Age','AnnualSalary'] 
sns.set(rc = {'figure.figsize' : (11.7,8.7)})
sns.barplot(x = Feature_imp, y = feature_names)