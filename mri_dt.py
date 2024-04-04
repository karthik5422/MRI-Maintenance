from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def data_preprocessing(data):
    # X -> features, y -> label
    X = data.iloc[:, :-1]
    le = LabelEncoder()
    X['scan_type_enc'] = le.fit_transform(X['scan_type'])
    X['coil_type_enc'] = le.fit_transform(X['coil_type'])
    X.drop(columns=['scan_type', 'coil_type'], inplace=True)
    y = data.iloc[:, -1:]
    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    return X, y, X_train, X_test, y_train, y_test


def decision_tree_classifer_model(train_input, train_output):
    # training a DescisionTreeClassifier
    from sklearn.tree import DecisionTreeClassifier

    dtree_model = DecisionTreeClassifier(max_depth=20).fit(train_input, train_output)

    return dtree_model


def mri_classifier_model():
    # loading the data from csv
    df = pd.read_csv('mri_scan_data.csv')

    # split the data into training and testing and convert categorical variables to numerical
    X, y, X_train, X_test, y_train, y_test = data_preprocessing(df)

    # Decision Tree classifier model
    dt_model = decision_tree_classifer_model(X_train, y_train)

    X_total = pd.concat([X_train, X_test], ignore_index=True, sort=False)
    y_total = pd.concat([y_train, y_test], ignore_index=True, sort=False)
    y_pred = dt_model.predict(X_total)

    score = dt_model.score(X_total, y_total)
    print(score)

    # creating a confusion matrix
    cm = confusion_matrix(y_total, y_pred, labels=['No Error', '201', '111', '7', '102'])
    print("Confusion Matrix for Test Data")
    print(cm)

    y_total['preds'] = y_pred

    df_out = pd.merge(df, y_total[['preds']], how='left', left_index=True, right_index=True)
    df_out.drop(columns=['error_code'], inplace=True)

    # write predicted data to file
    df_out.to_csv('output.csv')


if __name__ == '__main__':
    mri_classifier_model()
