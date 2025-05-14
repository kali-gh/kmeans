import libs
import numpy as np

from sklearn.datasets import make_classification

np.random.seed(42)

X, _ = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_classes=1)

### PARAMETERS ###
K = 3

# Generate initial random assignment
assignments = libs.generate_random_assignment(data=X, num_clusters=K)
centroids = libs.get_centroids_from_assignment(X, assignments)

libs.plot(X, assignments, centroids, 'initial_random_assignment')

iterations = 10

# Iterate
#    1. identifying the best clusters centroids from the assignment
#    2. updating the assignments based on the new centroids

for iter in range(iterations):
    centroids = libs.get_centroids_from_assignment(X, assignments)

    new_assignment_labels = libs.get_assignment_labels_given_centroids(X, centroids)

    assignments_updated = libs.update_assignments_given_new_centroids_and_new_assignment_labels(centroids, new_assignment_labels)

    assignments = assignments_updated.copy()

    name = f'iteration_{iter}'
    print(name)
    print('assigments')
    print(assignments)

    print('updated assignments')
    print(assignments_updated)

    libs.plot(X, assignments_updated, centroids = centroids, name=name)