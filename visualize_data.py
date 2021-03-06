# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#Data visualization
#Univariate plots
#box and whisker plots
#plots of individual variables
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex = False, sharey = False)

#Data visualization
#Univariate plots
#histogram of individual variables
dataset.hist()

#Data visualization
#Multivariate plots
#Interaction between the variables
#Scatterplots of all pairs of attributes
####Scatterplot matrices are a great way to roughly determine if you have a linear correlation between multiple variables.####
scatter_matrix(dataset)

plt.show()