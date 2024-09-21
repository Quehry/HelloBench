import json
import numpy as np
from scipy.optimize import minimize


def loss_function(w, X, y):
    """
    Loss function for regression
    :param w: weights
    :param X: features
    :param y: target
    """
    return np.mean((y - X @ w) ** 2)


def regression(task_type, regress_category):
    """
    Regression for human evaluation results
    :param task_type: task type
    :param regress_category: category to regress, "ALL" for all categories
    """
    load_path = f"ckwise_results_hm/stats_{task_type}.jsonl"
    with open(load_path, "r", encoding="utf-8") as f:
        data_list = [json.loads(line) for line in f]

    category_list = list(set({data["category"] for data in data_list})) 
    category_list = [regress_category] if regress_category != "ALL" else category_list
    return_dict = {}
    r2_dict = {}
    
    for category in category_list:
        x, y = [], []
        for data_dict in data_list:
            if data_dict["category"] == category:
                evaluation_results = data_dict["evaluation_results"]
                checklist = data_dict["checklist"]
                x.append([score / 4 for score in evaluation_results[:len(checklist)]])  # divide 4 for the reason that the annotation interface has scores from 0-4
                y.append(evaluation_results[-1])

        # y = w1 x1 + w2 x2 + ... + wn xn
        X, y = np.array(x), np.array(y)
        initial_w = np.zeros(X.shape[1])
        # we set the lower bound of weights to 0.5 to avoid the situation that some checklists are ignored
        bounds = [(0.5, None) for _ in range(X.shape[1])]
        result = minimize(loss_function, initial_w, args=(X, y), bounds=bounds)
        optimal_w = result.x
        r2 = 1 - np.sum((y - X @ optimal_w) ** 2) / np.sum((y - np.mean(y)) ** 2)
        r2_dict[category] = r2
        # rescale optimal_w to 100
        optimal_w = optimal_w / np.sum(optimal_w) * 100 
        return_dict[category] = optimal_w
    return return_dict, r2_dict


if __name__ == "__main__":
    task_type = "chat"
    category = "ALL"
    regression(task_type, regress_category=category)
