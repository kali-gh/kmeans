from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

import numpy as np


def generate_random_assignment(data, num_clusters):
    """
    Generates a random initial assignment with num_clusters.

    :param data: MxN data input
    :param num_clusters:  number of clusters we want
    :return: dict with assignments {cluster number : list of assigned indexes}
    """
    candidate_labels = range(num_clusters)

    assignment = {}

    # assignment maps the index of the point to the label we assign.
    for k in range(data.shape[0]):
        assignment[k] = int(np.random.choice(candidate_labels))

    # assignments is a dict of lists with points indexes assigned to each
    assignments = {}
    for k, v in assignment.items():
        if assignments.get(v) is None:
            assignments[v] = []
        else:
            pass

        assignments[v].append(k)

    return assignments


def get_distance_to_all_points(
        points): # MxN
    """
    Returns a matrix MxM with the euclidean distance of point i to point j
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    :param points: MxN data input to get distances
    :return: the distances to each point, MxM
    """
    return cdist(points, points)


def get_centroids_from_assignment(X, assignments):
    """
    Returns the centroids given the assignments.
    :param X: MxN data input
    :param assignments: dict of assignments {cluster number : list of assigned indexes}
    :return: dict of centroids {cluster number : index of the point in the overall space}
    """

    centroids = {}
    for k, v in assignments.items():

        #v = assignments[0

        S = X[v, :]
        dists = get_distance_to_all_points(S)

        dists_squared = dists**2

        sum_of_squared_distances = np.sum(dists_squared, axis=1)

        idx_min_local = int(np.argmin(sum_of_squared_distances))

        centroids[k] = v[idx_min_local] # index of the point in the overall space.

    return centroids

def get_assignment_labels_given_centroids(X, centroids):
    """
    Returns the assignment labels given the centroids.
    :param X: MxN data input
    :param centroids: dict of centroids {cluster number : index of the point in the overall space}
    :return: array of labels
    """

    centroid_numbers = np.array(list(centroids.keys()))
    centroid_numbers.sort(axis=-1)

    distances = None
    for current_centroid_number in centroid_numbers:
        current_centroid = X[centroids[current_centroid_number], :].reshape(1,-1)
        distances_to_current_centroid = cdist(X, current_centroid)
        squared_distances_to_current_centroid = distances_to_current_centroid ** 2

        if distances is None:
            distances = squared_distances_to_current_centroid
        else:
            distances = np.hstack([distances, squared_distances_to_current_centroid])

    assignment_label = np.argmin(distances, axis=1)

    return assignment_label


def update_assignments_given_new_centroids_and_new_assignment_labels(centroids, new_assignment_labels):
    """
    Updates the assignments given centroids and labels
    :param centroids:  dict of centroids {cluster number : index of the point in the overall space}
    :param new_assignment_labels: updated assignment labels
    :return: new assignments as a dict {label : list of assigned indexes}
    """

    centroid_numbers = np.array(list(centroids.keys()))
    centroid_numbers.sort(axis=-1)
    assignments_updated = {}

    for cn in centroid_numbers:
        idx_is_assigned_to_cn = np.where(new_assignment_labels == cn)[0]
        int_idx_is_assigned_to_cn = [int(idx) for idx in idx_is_assigned_to_cn]
        assignments_updated[int(cn)] = int_idx_is_assigned_to_cn # 0'th index is the tuple returned from np.where

    return assignments_updated

def assignments_to_assignments_array(assignments):
    """
    converts a dict of assignments e.g.
        0 : [1,2,3]
        1 : [0,4,5]
     to an array
        [1,0,0,0,1,1]
    :param assignments: the input assignments
    :return: list of assignments as explained above
    """
    array_size = 0
    for k, v in assignments.items():
        array_size += len(v)

    assignments_array = np.zeros(shape=(array_size,))

    for k, v in assignments.items():
        if k != 0:
            assignments_array[v] = k

    return assignments_array


def plot(X, assignments, centroids, name):
    from pathlib import Path
    Path('plots').mkdir(exist_ok=True, parents=True)
    assignments_array = assignments_to_assignments_array(assignments)

    centroid_keys = list(centroids.keys())
    base_centroid_color = max(centroid_keys) + 1
    for c_name, c_idx in centroids.items():
        assignments_array[c_idx] = base_centroid_color

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['k', 'r', 'y', 'b', 'g', 'c'])

    plt.scatter(X[:, 0], X[:, 1], c=assignments_array)
    plt.savefig(f'plots/plot_{name}.png')
    plt.close()