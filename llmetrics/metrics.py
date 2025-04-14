import numpy as np
from typing import List, Tuple

def _compute_entropy( probabilities: np.ndarray) -> np.ndarray:
        """
        Computes the entropy of the multiclass distribution specified by the given probabilities.

        :param probabilities (numpy.ndarray): Array of probabilities
            (parameters) for Bernoulli distributions.

        :return: array containing the entropy of each Bernoulli distribution.
        """
        # Ensure probabilities are within valid range [0, 1]
        probabilities = np.clip(probabilities, 0.00001, 1.0)  # Avoid log(0)
        return -np.sum(probabilities * np.log2(probabilities))

def compute_entropy(
    num_promt_versions: int,
    label_counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the predictions' entropy as a form of uncertainty.
        
        :param num_promt_versions: The number of alternative questions
        :param label_counts: the number of partial predictions for each label,
            assuming a multilabel classification scenario. The parameter counts
            the number of time a label has been included in a prediction,
             regardless of the ground truth.
        :return: tuple of numpy arrays, the first containing the entropy for
            each label and the second containing the probability that a label
             was predicted over all queries to the LLM
        """
        label_distribution = label_counts / (num_promt_versions)
        label_entropy = _compute_entropy(label_distribution)/len(label_counts)

        return label_entropy, label_distribution


def TVD(distribution1, distributions):
    return 0.5*np.abs(np.expand_dims(distribution1, 0) - distributions).sum(1)

def consistency_matrix(distributions: np.ndarray):
    _consistency_matrix = np.zeros((distributions.shape[0], distributions.shape[0])) # shape: (n_samples, n_samples) 
    for idx, row in enumerate(distributions):
        _consistency_matrix[idx, :] = 1. - TVD(row, distributions)
    return _consistency_matrix

def compute_consistency(TVD_matrix: np.ndarray) -> float:
    consistency = 0
    if TVD_matrix.shape[0] == 1 and TVD_matrix.shape[1] == 1:
        return np.nan
        # return TVD_matrix[0,0] # TODO: check if this is correct
    for idx, row in enumerate(TVD_matrix):
        _row = np.concatenate((row[:idx], row[idx+1:]))
        consistency += _row.sum()
    consistency = consistency / (TVD_matrix.shape[0]*TVD_matrix.shape[1] - TVD_matrix.shape[0])
    return consistency

