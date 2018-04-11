from numpy import linspace, linalg, transpose, zeros, array, random, arange, meshgrid, unravel_index, argmax
from cluster import Cluster
from math import pi, exp, pow, e
from utils import generate_probs, get_max_prob
MU = "mu"
SIGMA = "sigma"


def get_prob(cluster, coordinate):
    """
    Return a not normalized prob the j-coordinate belongs to i-cluster
    """
    exponent = (transpose(array(coordinate) - array(cluster.mu))
                @ array(linalg.inv(cluster.sigma)) @array((array(coordinate) - array(cluster.mu))))
    det = abs(linalg.det(array(cluster.sigma)))

    return (cluster.pi * (2*pi)**(-1) * det**(-1/2) * pow(e, (-0.5 * exponent)))


def get_function(type):
    """
    Given a type, returns a function that calculates the value
    """
    if(type == MU):
        return lambda point_prob, point, mu: point_prob * point
    else:
        return lambda point_prob, point, mu: point_prob * (point-mu) * transpose([point - mu])


def update_parameter(counter, points, cluster_probability, cluster_index, cluster, type):
    """
    Returns the new parameter to update
    """
    update_function = get_function(type)
    for index, element in enumerate(points):
        counter += update_function(cluster_probability[cluster_index]
                                   [index], array(element),  array(cluster.mu))
    return counter/sum(cluster_probability[cluster_index])


def update_gaussian(cluster, cluster_index, cluster_probability, points):
    """
    Update the paramters (sigms, mu, pi) of a gaussian
    """
    cluster.pi = sum(cluster_probability[cluster_index]) / len(points)
    temp_u = zeros(2)
    temp_sigma = zeros((2, 2))
    new_mu = update_parameter(temp_u, points, cluster_probability,
                              cluster_index, cluster, MU)
    cluster.update_mu(new_mu)
    new_sigma = update_parameter(temp_sigma, points, cluster_probability,
                                 cluster_index, cluster, SIGMA)
    cluster.update_sigma(new_sigma)


def get_set_of_points(cluster_probability, coordinates, number_clusters):
    """
    Split the coordinates into the clusters based on the probability that
    cluster-i belongs to coordinate-j
    Return an array with the points of each
    """
    points_cluster = [[] for i in cluster_probability]
    for index, point in enumerate(coordinates):
        i = get_max_prob(number_clusters, cluster_probability, index)
        points_cluster[i].append(point)
    return points_cluster


def probability_from_point(clusters, point, colors):
    """
    Given a point, return the probs that a point belong to i-cluster
    """
    point_prob = zeros(len(clusters))
    R = 0.0
    for index, cluster in enumerate(clusters):
        point_prob[index] = get_prob(cluster, point)
        R += point_prob[index]
    for index, cluster in enumerate(clusters):
        point_prob[index] = point_prob[index] / R
    color = colors[unravel_index(point_prob.argmax(), point_prob.shape)[0]]
    return point_prob, color


def expectation_maximization(coordinates, number_clusters, max_iterations):
    # generates n_probs
    probs = generate_probs(number_clusters)
    # initalize clusters
    clusters = [Cluster(probs[i]) for i in range(number_clusters)]
    iterations = 0
    cluster_probability = zeros((number_clusters, len(coordinates)))
    converge = False
    while(not converge and iterations < max_iterations):
        # STEP E:
        for coordinate_index, coordinate in enumerate(coordinates):
            R = 0
            # Update not normalized probability that point-j belongs to cluster-i
            for index, cluster in enumerate(clusters):
                cluster_probability[index][coordinate_index] = get_prob(
                    cluster, coordinate)
                R += cluster_probability[index][coordinate_index]
            # Calculate normalized probability
            for index, cluster in enumerate(clusters):
                cluster_probability[index][coordinate_index] = cluster_probability[index][coordinate_index] / R
        # Update gaussians
        for index, cluster in enumerate(clusters):
            update_gaussian(cluster, index, cluster_probability, coordinates)
        converge = [c.check_converge for c in clusters].count(
            True) == number_clusters
        iterations += 1

    return cluster_probability, clusters
