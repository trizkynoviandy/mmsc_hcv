import matplotlib.pyplot as plt
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier

# Function for model training and evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=5)
    print(f"Classification report for {model_name} (Cross-validation):")
    print(classification_report(y_train, y_pred_cv, digits=4))
    
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    print(f"Classification report for {model_name} (Test set):")
    print(classification_report(y_test, y_pred_test, digits=4))
    
    return y_pred_test

# Function to scale features
def scale_features(X_train, X_test):
    scaler = StandardScaler().set_output(transform="pandas")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, class_names=['Active', 'Inactive']):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots()
    disp.plot(cmap=plt.cm.Blues, colorbar=False, ax=ax)
    
    ax.set_xlabel('Actual', fontsize=14)
    ax.set_ylabel('Predicted', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    
    for text in ax.texts:
        text.set_fontsize(16)
    
    plt.show()

if __name__ == "__main__":

    df = pd.read_csv('dataset/hcv_ns3_dataset.csv')
    X = df.drop(["activity"], axis=1)
    y = pd.DataFrame(df["activity"]).copy()
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2, random_state=13)

    # Scale the features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Define classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(probability=True, random_state=42, kernel='poly'),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(random_state=0, max_iter=100, solver='sag')
    }

    # Evaluate each model
    predictions = {}
    for name, model in classifiers.items():
        predictions[name] = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, model_name=name)

    # Stacking Classifier
    stacking_clf = StackingClassifier(
        classifiers=[classifiers["Random Forest"], classifiers["SVM"], classifiers["Gradient Boosting"], classifiers["KNN"]],
        meta_classifier=LogisticRegression(),
        use_probas=True,
        average_probas=True,
        use_features_in_secondary=True,
        use_clones=True
    )

    # Evaluate Stacking Classifier
    stacking_pred = evaluate_model(stacking_clf, X_train_scaled, y_train, X_test_scaled, y_test, model_name="Stacking Classifier")

    # Plot confusion matrix for the stacking classifier
    plot_confusion_matrix(y_test, stacking_pred)