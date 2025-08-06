import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

iris = load_iris(as_frame=True)
df = iris.frame
X = df.drop(columns='target')
y = df['target']

np.random.seed(42)
extra_classes = X.sample(50, replace=True).copy()
extra_target = np.random.randint(3, 5, size=50)
X_aug = pd.concat([X, extra_classes], axis=0)
y_aug = pd.concat([y, pd.Series(extra_target)], axis=0)

clf = RandomForestClassifier()
##clf.fit(X_aug, y_aug)
clf.fit(X, y)


joblib.dump(clf, 'model.pkl')
print("✅ Model saved as model.pkl")
