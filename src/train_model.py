import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from feature_engineering import prepare_dataset

df = prepare_dataset('../data/matches.csv', '../data/deliveries.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(
    [('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'),
      ['batting_team','bowling_team','city'])],
    remainder='passthrough'
)

pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)

print("ROC AUC:", roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1]))

pickle.dump(pipe, open('../pipe.pkl', 'wb'))
