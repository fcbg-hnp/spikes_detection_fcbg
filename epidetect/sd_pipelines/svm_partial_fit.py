import numpy as np
from sklearn.metrics import hinge_loss
from sklearn.model_selection import train_test_split
from ..utils.parse_configuration_file import varname


def svm_partial_fit(X, y, svm_model):
    def hinge_loss_with_elasticnet(y_true, pred_decision, weights, labels, alpha, l1_ratio):
        return hinge_loss(y_true, pred_decision, labels) + alpha * (l1_ratio * np.sum(np.abs(weights)) +
                                                                    (1 - l1_ratio) * np.sum(weights * weights))

    class PartialFitNotPossibleError(Exception):
        pass

    if not hasattr(svm_model, 'partial_fit'):
        raise PartialFitNotPossibleError(f"Model {varname(svm_model)} does not have partial fit method")
    else:
        svm_model.set_params(**{'warm_start': True, 'early_stopping': False, 'class_weight': None})

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=svm_model.validation_fraction, random_state=100,
                                                      shuffle=True)
    best_loss = np.inf

    print("Starting partial fit of SVM model")

    n_criterion = 0

    for i_iter in range(svm_model.max_iter):
        np.random.seed(i_iter)
        np.random.shuffle(X_train)
        np.random.seed(i_iter)
        np.random.shuffle(y_train)

        svm_model.partial_fit(X_train, y_train, classes=np.array([-1, 1]))

        val_loss = hinge_loss_with_elasticnet(y_true=y_val, pred_decision=svm_model.decision_function(X_val),
                                              weights=svm_model.coef_, labels=[1, -1], alpha=svm_model.alpha,
                                              l1_ratio=svm_model.l1_ratio)

        criterion = val_loss > best_loss - svm_model.tol
        n_criterion += 1 if criterion else 0

        if criterion and n_criterion == svm_model.n_iter_no_change:
            print(f'Finished in {i_iter}/{svm_model.max_iter} iterations')
            break

        best_loss = val_loss if val_loss < best_loss else best_loss

    return svm_model
