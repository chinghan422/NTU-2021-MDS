'''
For python install package, please run commands below:
$ pip install mlxtend
$ pip install matplotlib==3.2.1
$ pip install networkx==2.3
$ pip install decorator==4.3
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import statsmodels.api as sm
import statsmodels.stats.api as sms
from scipy import stats
from statsmodels.compat import lzip
from statsmodels.stats.stattools import durbin_watson
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve, auc, plot_roc_curve
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# 1.Linear Regression Analysis for Wine Quality
df = pd.read_csv('/Users/kanko/Documents/NTU/MDS/MDS_Assignment1/MDS_Assignment1_furnace.csv')
df.head()

x = df.drop(['f9', 'grade'], axis=1)
y = df['grade']
x.head()
x.describe()

X = sm.add_constant(x)
result = sm.OLS(y, X).fit()  # OLS computing
result.summary()

pv = result.pvalues
pv_df = pd.DataFrame(lzip(pv, pv<0.01), columns=["p_values", "<0.01"], index=[pv.index])
pv_df.sort_values(by = 'p_values')

result.resid  # residual analysis

shapiro_test = stats.shapiro(result.resid)  # normality
shapiro_test

durbin_watson(result.resid)  # independence

name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(result.resid, result.model.exog)  # homogeneity
lzip(name, test)


# 2.Data Preprocessing and Logistic Regression
df = pd.read_csv('/Users/kanko/Documents/NTU/MDS/MDS_Assignment1/MDS_Assignment1_census.csv', na_values=' ?', header = None)
df.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]
df.info()
df.head()

x = df.drop(['class'], axis=1)  # basic statistics
x.info()
x.var()

colors=['lightsteelblue', 'cornflowerblue', 'royalblue', 'midnightblue', 'navy', 'darkblue', 'mediumblue']

# visualization
count = df["age"].value_counts(sort=True)
x_axis = count.index
plt.figure(figsize=(10, 6), dpi=50)
plt.scatter(x_axis, count, alpha=0.5)
plt.xlabel("age")
plt.ylabel("amount")
plt.title("Age")
plt.show()

count = df["class"].value_counts(sort=True)
ls=['<=50K','>50K']
plt.pie(count, labels=ls, colors=colors, autopct='%.2f%%')
plt.title("Class distribution")
plt.show()
print(count)

count = df["workclass"].value_counts(sort=True, ascending=True)
x_axis = count.index
plt.figure(figsize=(10, 6), dpi=80)
plt.barh(x_axis,count)
plt.xlabel("amount")
plt.ylabel("workclass")
plt.title("Workclass data amount")
plt.show()

count = df["education"].value_counts(sort=True, ascending=True)
x_axis = count.index
plt.figure(figsize=(10, 6), dpi=80)
plt.barh(x_axis,count)
plt.xlabel("amount")
plt.ylabel("education")
plt.title("Education data amount")
plt.show()

count = df["education-num"].value_counts(sort=True)
x_axis = count.index
plt.figure(figsize=(10, 6), dpi=50)
plt.scatter(x_axis, count, alpha=0.5)
plt.xlabel("education-num")
plt.ylabel("amount")
plt.title("Education-num")
plt.show()

count = df["marital-status"].value_counts(sort=True, ascending=True)
x_axis = count.index
plt.figure(figsize=(10, 6), dpi=80)
plt.barh(x_axis,count)
plt.xlabel("amount")
plt.ylabel("marital-status")
plt.title("Marital-status data amount")
plt.show()

count = df["occupation"].value_counts(sort=True, ascending=True)
x_axis = count.index
plt.figure(figsize=(10, 6), dpi=80)
plt.barh(x_axis,count)
plt.xlabel("amount")
plt.ylabel("occupation")
plt.title("Occupation data amount")
plt.show()

count = df["relationship"].value_counts(sort=True, ascending=True)
x_axis = count.index
plt.figure(figsize=(10, 6), dpi=80)
plt.barh(x_axis,count)
plt.xlabel("amount")
plt.ylabel("relationship")
plt.title("Relationship data amount")
plt.show()

count = df["race"].value_counts(sort=True, ascending=True)
x_axis = count.index
plt.figure(figsize=(10, 6), dpi=80)
plt.barh(x_axis,count)
plt.xlabel("amount")
plt.ylabel("race")
plt.title("Race data amount")
plt.show()

count = df["sex"].value_counts(sort=True)
ls=['Male','Female']
plt.pie(count, labels=ls, colors=colors, autopct='%.2f%%')
plt.title("Sex")
plt.show()

count = df["hours-per-week"].value_counts(sort=True)
x_axis = count.index
plt.figure(figsize=(10, 6), dpi=50)
plt.scatter(x_axis, count, alpha=0.5)
plt.xlabel("hours-per-week")
plt.ylabel("amount")
plt.title("Hours-per-week")
plt.show()

# impute missing value
count = df["workclass"].value_counts(sort=True)
print(count)
df['workclass'] = df['workclass'].replace(np.nan, " Private")

count = df["occupation"].value_counts(sort=True)
print(count)
df['occupation'] = df['occupation'].replace(np.nan, " Prof-specialty")

count = df["native-country"].value_counts(sort=True)
print(count)
df['native-country'] = df['native-country'].replace(np.nan, " United-States")

# drop missing value
# df_drop = df.dropna()
# df_drop.head(28)
# print(df_drop.shape)
# print(df.shape)

# find outlier index
X = df.drop(["workclass","education","marital-status","occupation","relationship","race","sex","native-country","class"], axis=1)
ss = StandardScaler()
scaled_df = ss.fit_transform(X)
scaled_df = pd.DataFrame(scaled_df, columns = X.columns)
abs_scaled_df = (abs(scaled_df)>3)
all_abs_scaled = abs_scaled_df.any(axis=1)
outlier_index = all_abs_scaled[all_abs_scaled==True]
outlier_index.index
# abs_scaled_df.iloc[1,range(0,6,1)]

df_no_outlier = df.drop(outlier_index.index)  # drop outlier
df_no_outlier.shape  # compare amount
scaled_df.shape

# dummy variable
du = df_no_outlier
du = pd.get_dummies(du, columns=["workclass",
                                 "education",
                                 "marital-status",
                                 "occupation",
                                 "relationship",
                                 "race",
                                 "sex",
                                 "native-country",
                                 "class"], drop_first=True)
du.info()

x = du.drop(['class_ >50K'], axis=1)
y = du[['class_ >50K']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # split dataset
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

scaler = StandardScaler()
scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)
# pd.DataFrame(x_train_std).head()

clf = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=100).fit(x_train_std, y_train.values.ravel())  # classifier

predict = clf.predict(x_test_std)
clf.predict_proba(x_test_std)[:10]

# model evaluation
coef = clf.coef_[0]
for i in range(len(du.columns[:-1])):
     print(du.columns[i], ":", coef[i])

print("Accuracy score:", clf.score(x_test_std, y_test))

matric = confusion_matrix(y_test,predict)
sns.heatmap(matric,square=True,annot=True,cbar=False)
plt.xlabel("predict value")
plt.ylabel("true value")
plt.title("confusion matrix")
plt.show()

print("report:\n",classification_report(y_test,predict,labels=[1,0],target_names=[">50K","<=50K"]))

plot_roc_curve(clf, x_test_std, y_test)
plt.show()


# 3.Association Rule- Market Basket Analysis
df = pd.read_csv('/Users/kanko/Documents/NTU/MDS/MDS_Assignment1/MDS_Assignment1_groceries.csv', header=None, names = list(range(0,32)))
df.info()
df.head()
df.tail()
df.sample(10)  # check random entries

trans = []
for i in range(0, 9835):  # create transaction list
    trans.append([str(df.values[i,j]) for j in range(0, 32)])

for idx, item in enumerate(trans):  # remove NA values
    while 'nan' in trans[idx]:
        trans[idx].remove("nan")

te = TransactionEncoder()
trans = te.fit_transform(trans)  # apply transaction encoder on trans. list
trans_df = pd.DataFrame(trans, columns = te.columns_)
trans_df.shape
trans_df
te.columns_  # product catagories

df.stack().value_counts().tail()  # 找出總銷售量低者
trans_df.drop(["baby food","sound storage medium"], axis=1, inplace=True)
trans_df.shape

frequent_itemsets = apriori(trans_df, min_support=0.001, use_colnames=True)
ar = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.15)
ar.head()

ar.quantile(0.3)[["support","confidence","lift"]]  # find low threshold
ar.quantile(0.95)[["support","confidence","lift"]]

top_rules = ar[(ar["support"]>0.001118)&
               (ar["confidence"]>=0.67)&
               (ar["lift"]>=7.73)].sort_values(by=["lift"],ascending=False)
top_rules.head()
len(top_rules)

# visualization
fg = sns.FacetGrid(data=top_rules,  # distribution of top_rules parameters
                   hue='lift',
                   palette="ch:s=.25,rot=-.25",
                   aspect=1.6,
                   height=3)
fg.map(plt.scatter, 'support', 'confidence')

# create list of association rules
ar_list = ar.copy()
ar_list["antecedents"] = ar["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
ar_list["consequents"] = ar["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")

# network of red/blush wine
fig, ax = plt.subplots(figsize=(25,15))
GA = nx.from_pandas_edgelist(ar_list[ar_list["antecedents"]=="red/blush wine"],source='antecedents',target='consequents')
pos = nx.spring_layout(GA)
nx.draw_networkx_nodes(GA, pos, node_size = 15000, node_color = "orange")
nx.draw_networkx_edges(GA, pos, width = 3, edge_color = 'grey')
nx.draw_networkx_labels(GA, pos, font_size = 20, font_family = 'sans-serif')
plt.show()

# network of liquor
fig, ax = plt.subplots(figsize=(25,15))
GA = nx.from_pandas_edgelist(ar_list[ar_list["antecedents"]=="liquor"],source='antecedents',target='consequents')
pos = nx.spring_layout(GA)
nx.draw_networkx_nodes(GA, pos, node_size = 15000, node_color = "orange")
nx.draw_networkx_edges(GA, pos, width = 3, edge_color = 'grey')
nx.draw_networkx_labels(GA, pos, font_size = 20, font_family = 'sans-serif')
plt.show()

# network of bottled beer
fig, ax = plt.subplots(figsize=(25,15))
GA = nx.from_pandas_edgelist(ar_list[ar_list["antecedents"]=="bottled beer"],source='antecedents',target='consequents')
pos = nx.spring_layout(GA)
nx.draw_networkx_nodes(GA, pos, node_size = 15000, node_color = "orange")
nx.draw_networkx_edges(GA, pos, width = 3, edge_color = 'grey')
nx.draw_networkx_labels(GA, pos, font_size = 20, font_family = 'sans-serif')
plt.show()

# network of initial association rules
fig, ax = plt.subplots(figsize=(10,10))
GA = nx.from_pandas_edgelist(ar_list.head(10), source='antecedents', target='consequents')
pos = nx.spring_layout(GA)
nx.draw_networkx_nodes(GA, pos, node_size = 1500, node_color = "orange")
nx.draw_networkx_edges(GA, pos, width = 2, edge_color = 'grey')
nx.draw_networkx_labels(GA, pos, font_size = 12)
plt.show()

new_rules = top_rules.copy()
new_rules["antecedents"] = top_rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
new_rules["consequents"] = top_rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")

# network of top_rules
fig, ax = plt.subplots(figsize=(20,20))
GA = nx.from_pandas_edgelist(new_rules, source='antecedents', target='consequents')
pos = nx.spring_layout(GA)
nx.draw_networkx_nodes(GA, pos, node_size = 2000, node_color = "orange")
nx.draw_networkx_edges(GA, pos, width = 2, edge_color = 'grey')
nx.draw_networkx_labels(GA, pos, font_size = 12)
plt.show()

# popular single items
plt.figure(figsize=(15, 6), dpi=100)
colors = plt.cm.cool(np.linspace(1, 0, 50))
df.stack().value_counts().head(50).plot.bar(color = colors)  # transform dataframe into stack to calculate value, ref:https://www.796t.com/post/ajJmYW8=.html
plt.title('Top 50 popular items', fontsize = 15)
plt.show()
df.stack().value_counts().head(50)  # popular items list
