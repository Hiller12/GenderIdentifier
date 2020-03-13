from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import numpy as npy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def getModels(maleSamplePathIn, femaleSamplePathIn):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    maleSamplePath = maleSamplePathIn #"analysisResults/male/"
    femaleSamplePath = femaleSamplePathIn #"analysisResults/female/"
    models = {
        svm.SVC(kernel='rbf', degree=1, C=2, gamma='scale', class_weight='balanced') : "SVM",
        GaussianNB() : "Naive Bayes",
        DecisionTreeClassifier() : "Decision Tree",
        LogisticRegression() : "Logistic Regression",
        RandomForestClassifier() : "RandomForest",
        KNeighborsClassifier(n_neighbors=7) : "KNN",
        KNeighborsClassifier(n_neighbors=1) : "NN"
    }
    maledf = pd.read_csv(maleSamplePath + "Results.csv")
    femaledf = pd.read_csv(femaleSamplePath + "Results.csv")
    maleY = maledf.values[:, 7]
    femaleY = femaledf.values[:, 7]
    maleX = maledf.values[:, 1:7]
    femaleX = femaledf.values[:, 1:7]
    X = npy.vstack([maleX, femaleX])
    Y = npy.hstack([maleY, femaleY])
    normX = preprocessing.normalize(X)
    strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    for model in models:
        scores = cross_val_score(model, X, y=Y, cv=strat_k_fold, scoring="accuracy")
        print(models[model] + '\n' + str(scores) + " n mean is " + str(npy.mean(scores)) + '\n')
    df_subset = pd.DataFrame(npy.vstack([maledf.values[:, 1:8], femaledf.values[:, 1:8]]), index=None,
                             columns=['longWordsCnt', 'allnessTermsCnt', 'uniqueWordsCnt',
                                      'negWords', 'nproCnt', 'nounCnt', 'gender'])
    X_embedded = TSNE(n_components=2).fit_transform(X)
    df_subset['tsne-2d-one'] = X_embedded[:, 0]
    df_subset['tsne-2d-two'] = X_embedded[:, 1]
    sns.relplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="gender",
        palette=sns.color_palette("hls", 2),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    plt.show()
    for model in models:
        model.fit(normX, Y)
    return models

