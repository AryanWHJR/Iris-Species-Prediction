import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

features=["RI",
"Na",
"Mg",
"Al",
"Si",
"K",
"Ca",
"Ba",
"Fe"]

def predict(model,feat):
  glass_type=model.predict([feat])
  glass_type=glass_type[0]
  if glass_type == 1:
    return "building windows float processed"
  elif glass_type == 2:
    return "building windows non float processed"
  elif glass_type == 3:
    return "vehicle windows float processed"
  elif glass_type == 4:
    return "vehicle windows non float processed"
  elif glass_type == 5:
    return "containers"
  elif glass_type == 6:
    return "tableware"
  else:
    return "headlamps"

st.title("Glass Type prediction web app")
st.sidebar.title("Glass Type prediction web app")
if st.sidebar.checkbox("Show Raw Data"):
  st.subheader("Glass Type Dataset")
  st.dataframe(glass_df)
st.sidebar.subheader("Visualisation Selector")

plot_type=st.sidebar.multiselect("Select Charts/Plots",("Scatter plot",
"Histogram",
"Box plot",
"Count plot",
"Pie chart",
"Correlation heatmap",
"Pair plot"))

features_list=("RI",
"Na",
"Mg",
"Al",
"Si",
"K",
"Ca",
"Ba",
"Fe",
"Glass Type")

st.set_option('deprecation.showPyplotGlobalUse', False)

if "Scatter plot" in plot_type:
    column=st.sidebar.selectbox("Select a column to create Scatter Plot",features_list)
    st.subheader(f"Scatter plot between {column} and GlassType")
    plt.figure(figsize = (12, 6))
    sns.scatterplot(x = glass_df[column], y = 'GlassType', data = glass_df)
    st.pyplot()

# Create histograms.
if "Histogram" in plot_type:
    column=st.sidebar.selectbox("Select a column to create Histogram",features_list)
    st.subheader(f"Histogram for {column}")
    plt.figure(figsize = (12, 6))
    plt.hist(glass_df[column], bins = 'sturges', edgecolor = 'black')
    st.pyplot() 

# Create box plots.
if "Box plot" in plot_type:
    column=st.sidebar.selectbox("Select a column to create Box Plot",features_list)
    st.subheader(f"Box plot for {column}")
    plt.figure(figsize = (12, 2))
    sns.boxplot(glass_df[column])
    st.pyplot()

if "Pie chart" in plot_type:
    st.subheader(f"Pie Chart for Glass Type")
    plt.figure(figsize = (12, 2))
    plt.pie(glass_df["GlassType"].value_counts(),labels=glass_df["GlassType"].value_counts().index)
    st.pyplot()

if "Count plot" in plot_type:
    st.subheader(f"Count plot for Glass Type")
    plt.figure(figsize = (12, 9))
    sns.countplot(y=glass_df["GlassType"],orient="h")
    st.pyplot()

if "Correlation heatmap" in plot_type:
    st.subheader(f"Correlation heatmap for Glass Type")
    plt.figure(figsize = (12, 12))
    sns.heatmap(glass_df.corr(),annot=True)
    st.pyplot()

if "Pair plot" in plot_type:
    st.subheader(f"Pair plot for Glass Type")
    plt.figure(figsize = (20, 20))
    sns.pairplot(glass_df)
    st.pyplot()

st.sidebar.subheader("Select your values")
lst_features=[]

for feature in features_list[:-1]:
    lst_features.append(st.sidebar.slider(f"Input {feature}",float(glass_df[feature].min()),float(glass_df[feature].max())))
st.sidebar.subheader("Choose classifier")
classifier=st.sidebar.selectbox("classifier",("SVM","RFC","Logistic Regression"))

if classifier=="SVM":
    st.sidebar.subheader("Model Hyper Parameters")
    c=st.sidebar.number_input("Error Rate", 1, 100, 1)
    k=st.sidebar.radio("kernel",("linear","rbf","poly"))
    gamma=st.sidebar.number_input("Gamma", 1, 100, 1)
    if st.sidebar.button("classify"):
        st.subheader("SVM")
        svc=SVC(c,k,gamma)
        svc.fit(X_train,y_train)
        y_pred=svc.predict(X_test)
        score=svc.score(X_test,y_test)
        glass_type1=predict(svc,lst_features)
        st.write(f"Glass Type predicted is {glass_type1}")
        st.write(f"Accuracy of the model is {round(score*100,2)}%")
        plot_confusion_matrix(svc,X_test,y_test)
        st.pyplot()

if classifier=="RFC":
    st.sidebar.subheader("Model Hyper Parameters")
    trees=st.sidebar.number_input("Number of trees",100,5000,step=10)
    depth_trees=st.sidebar.number_input("Depth of trees",1,100,1)
    if st.sidebar.button("classify"):
        st.subheader("RFC")
        rfc=RandomForestClassifier(n_estimators=trees,max_depth=depth_trees,n_jobs=-1)
        rfc.fit(X_train,y_train)
        y_pred=rfc.predict(X_test)
        score=rfc.score(X_test,y_test)
        glass_type1=predict(rfc,lst_features)
        st.write(f"Glass Type predicted is {glass_type1}")
        st.write(f"Accuracy of the model is {round(score*100,2)}%")
        plot_confusion_matrix(rfc,X_test,y_test)
        st.pyplot()

if classifier=="Logistic Regression":
    st.sidebar.subheader("Model Hyper Parameters")
    c=st.sidebar.number_input("C",1,100,1)
    max_iter=st.sidebar.number_input("Maximum iterations",10,1000,step=10)
    if st.sidebar.button("classify"):
        st.subheader("Logistic Regression")
        lr=LogisticRegression(C=c,max_iter=max_iter)
        lr.fit(X_train,y_train)
        y_pred=lr.predict(X_test)
        score=lr.score(X_test,y_test)
        glass_type1=predict(lr,lst_features)
        st.write(f"Glass Type predicted is {glass_type1}")
        st.write(f"Accuracy of the model is {round(score*100,2)}%")
        plot_confusion_matrix(lr,X_test,y_test)
        st.pyplot()