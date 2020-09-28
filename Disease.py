
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pdpbox import pdp, get_dataset, info_plots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import eli5
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


def cnf_matrix_plotter(cm, classes):
    """
    传入混淆矩阵和标签名称列表，绘制混淆矩阵
    """
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title('Confusion Matrix_Task3')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 fontsize=25)

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    plt.savefig('results/ConfusionMatrix_Task3.png')

df = pd.read_csv('Disease.csv')
print(df.shape)
#print(df.dtypes)
X = df.drop('target',axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=5)
model.fit(X_train, y_train)
feature_names = X_train.columns
estimator = model.estimators_[7]
y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'Healthy'
y_train_str[y_train_str == '1'] = 'Disease'
y_train_str = y_train_str.values


print('特征排序：')
feature_names = X_test.columns
feature_importances = model.feature_importances_
indices = np.argsort(feature_importances)[::-1]

for index in indices:
    print("feature %s (%f)" %(feature_names[index], feature_importances[index]))
#
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
confusion_matrix_model = confusion_matrix(y_test, y_pred)
cnf_matrix_plotter(confusion_matrix_model, ['Healthy','Disease'])
print(classification_report(y_test, y_pred, target_names=['Healthy','Disease']))
y_pred_quant = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)

#print('fpr:',fpr)
#print('tpr:',tpr)


# plt.plot(fpr, tpr)
# plt.plot([0, 1], [0, 1],ls="--", c=".3")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.rcParams['font.size'] = 12
# plt.title('ROC_Task3 curve')
# plt.xlabel('False Positive Rate (1 - Specificity)')
# plt.ylabel('True Positive Rate (Sensitivity)')
# plt.grid(True)
# print(auc(fpr, tpr))
# plt.show()
#
# plt.savefig('results/ROC_Task3.png')
#
eli5.show_weights(estimator,feature_names=feature_names.to_list())
plt.figure(figsize=(20,16))
plt.title("Feature Importance_Task3")
plt.bar(range(len(feature_importances)), feature_importances[indices], color='b')
plt.xticks(range(len(feature_importances)), np.array(feature_names)[indices], color='b', rotation=90)
plt.show()
plt.savefig('results/FeatureImportance_Task3.png')