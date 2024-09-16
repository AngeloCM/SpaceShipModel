from typing import Any
from sklearn.base import ClassifierMixin
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import (
    RocCurveDisplay,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def base_model_performance(
    classifier: ClassifierMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    """
    Trains a classifier, makes predictions, and evaluates performance using cross-validation and ROC-AUC score.

    Args:
        classifier (ClassifierMixin): The classifier to be trained and evaluated.
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        None
    """
    classifier.fit(X_train, y_train)
    y_val_pred = classifier.predict(X_val)
    y_val_pred_proba = classifier.predict_proba(X_val)[:, 1]

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

    cross_val_score_mean = cross_val_score(
        classifier, X_val, y_val, cv=cv, scoring="accuracy"
    ).mean()

    auc_roc = roc_auc_score(y_val, y_val_pred_proba)

    accuracy = accuracy_score(y_val, y_val_pred)

    precision = precision_score(y_val, y_val_pred)

    recall = recall_score(y_val, y_val_pred)

    f1 = f1_score(y_val, y_val_pred)

    print("Cross Validation Mean Score: ", "{0:.2%}".format(cross_val_score_mean))
    print("AUC ROC Score: ", "{0:.2%}".format(auc_roc))
    print("Accuracy: ", "{0:.2%}".format(accuracy))
    print("Precision: ", "{0:.2%}".format(precision))
    print("Recall: ", "{0:.2%}".format(recall))
    print("F1: ", "{0:.2%}".format(f1))


def plot_roc_curve(
    classifier: ClassifierMixin, x_test: np.ndarray, y_test: np.ndarray
) -> None:
    """
    Plots the ROC curve for a given classifier and test data.

    Args:
        classifier (ClassifierMixin): The trained classifier.
        x_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        None
    """
    RocCurveDisplay.from_estimator(classifier, x_test, y_test)
    plt.title("ROC_AUC Plot")
    plt.show()


def model_evaluation(
    classifier: ClassifierMixin, x_test: np.ndarray, y_test: np.ndarray
) -> None:
    """
    Evaluates the performance of a classifier using a confusion matrix and classification report.

    Args:
        classifier (ClassifierMixin): The trained classifier.
        x_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        None
    """
    # Confusion Matrix
    cm = confusion_matrix(y_test, classifier.predict(x_test))
    names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    counts = [value for value in cm.flatten()]
    percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(names, counts, percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cm, annot=labels, fmt="")

    # Classification Report
    print(classification_report(y_test, classifier.predict(x_test)))
