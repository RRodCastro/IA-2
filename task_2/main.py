file_name = "points.txt"
from numpy import linspace, linalg, transpose, zeros, array, random, arange, meshgrid, argmax, unravel_index, shape, degrees, arctan2, sqrt
from random import sample
from cluster import Cluster
from math import pi, exp, pow, e
from random import uniform
from time import sleep
from scipy.stats import multivariate_normal
from matplotlib.pyplot import plot, show, subplots
from matplotlib.patches import Ellipse


def clean_coordinate(coordinate):
    """
    Get a coordinate, remove dirty chars and parse to float
    """
    for char in ignore_characters:
        coordinate = coordinate.replace(char, '')
    return list(map(lambda axis: float(axis), coordinate.split(',')))


def parse_coordinates(coordinates):
    """
    Return list of coordinates cleaned
    """
    return list(map(clean_coordinate, coordinates))


def get_prob(cluster, coordinate):
    """
    Return a not normalized prob the j-coordinate belongs to i-cluster
    """
    exponent = (transpose(array(coordinate) - array(cluster.mu))
                @ array(linalg.inv(cluster.sigma)) @array((array(coordinate) - array(cluster.mu))))
    det = abs(linalg.det(array(cluster.sigma)))

    return (cluster.pi * (2*pi)**(-1) * det**(-1/2) * pow(e, (-0.5 * exponent)))


def update_gaussian(cluster, cluster_index, cluster_probability, points):
    """
    Update the paramters (sigms, mu, pi) of a gaussian
    """
    cluster.pi = sum(cluster_probability[cluster_index]) / len(points)
    temp_u = array([0., 0.])
    temp_sigma = array([[0., 0.], [0., 0.]])
    for index, element in enumerate(points):
        temp_u += ((cluster_probability[cluster_index]
                    [index]) * array(element))
    cluster.update_mu(temp_u/sum(cluster_probability[cluster_index]))
    for index, element in enumerate(points):
        temp_sigma += cluster_probability[cluster_index][index] * (array(
            element) - array(cluster.mu)) * transpose([(array(element) - array(cluster.mu))])
    cluster.update_sigma((temp_sigma/sum(cluster_probability[cluster_index])))


def get_max_prob(number_clusters, cluster_prob, point_index):
    """
    Return the best cluster (gretear probability) index for a given a point
    """
    max = -1
    max_index = -1
    for i in range(number_clusters):
        prob = cluster_prob[i][point_index]
        if (prob > max):
            max = prob
            max_index = i
    return max_index


def expectation_maximization(coordinates, number_clusters, max_iterations):
    # Get n coordinates
    # Crear n_clusters, initialization step
    probs = generate_probs(number_clusters)
    clusters = [Cluster(probs[i]) for i in range(number_clusters)]
    iterations = 0
    cluster_probability = zeros((number_clusters, len(coordinates)))
    converge = False
    while(not converge and iterations < max_iterations):
        # print("I ", iterations)
        # STEP E:
        for coordinate_index, coordinate in enumerate(coordinates):
            R = 0
            # Update not normalized probability
            # Point-j belongs to cluster-i
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
        # Check if it converge
        converge = [c.check_converge for c in clusters].count(
            True) == number_clusters
        iterations += 1

    return cluster_probability, clusters


def generate_probs(number_clusters):
    """
    Generates an array of probs (they satisfies the Kolmogorov axioms)
    """
    probs = []
    for i in range(number_clusters):
        if(i == 0):
            probs.append(uniform(0, float(1/number_clusters)))
        elif(i == number_clusters-1):
            probs.append(1-sum(probs))
        else:
            probs.append(uniform(0, 1 - sum(probs)))
    return probs


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


nFile = open(file_name, 'r')
coordinates = nFile.readlines()
nFile.close()
number_clusters = 4
ignore_characters = [' ', '\n']
# number_clusters = input("Number of clusters: \n")
# number_clusters = int(number_clusters)
coordinates = parse_coordinates(coordinates)
cluster_probability, clusters = expectation_maximization(
    coordinates, number_clusters, 100)
COLORS = ['red', 'blue', 'green', 'yellow', 'gray', 'orange', 'violet', 'brown',
          'cyan', 'magenta']

points_set = get_set_of_points(
    cluster_probability, coordinates, number_clusters)


def probability_from_point(clusters, point):
    point_prob = zeros(len(clusters))
    R = 0.0
    for index, cluster in enumerate(clusters):
        point_prob[index] = get_prob(cluster, point)
        R += point_prob[index]
    for index, cluster in enumerate(clusters):
        point_prob[index] = point_prob[index] / R
    return point_prob


def eigsorted(cov):
    vals, vecs = linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


fig, ax = subplots()
for index, c in enumerate(clusters):
    xm = c.mu[0]
    ym = c.mu[1]
    cov = c.sigma
    vals, vecs = eigsorted(cov)
    theta = degrees(arctan2(*vecs[:, 0][::-1]))
    for j in range(0, 6):
        w, h = j * sqrt(vals)
        ell = Ellipse(xy=(xm, ym), width=w, height=h,
                      angle=theta, color=COLORS[index])
        ell.set_facecolor('none')
        ax.add_artist(ell)
        ax.scatter(xm, ym, marker="^", color=COLORS[index])
for cluster_index in range(len(points_set)):
    for point_index in points_set[cluster_index]:
        ax.scatter(point_index[0], point_index[1],
                   color=COLORS[cluster_index])

# for i in range(0, len(points)):
#     [xp, yp] = points[i]
# ax.scatter(xp, yp, color=colors[cluster_type[i]])


def plot_points(points_set):
    for i in range(len(points_set)):
        plot([x[0] for x in points_set[i]], [x[1]
                                             for x in points_set[i]], 'o', color=COLORS[i])


point_x = input("Ingrese x: \n")
point_y = input("Ingtrdr y: \n")
prob = (probability_from_point(clusters, [int(point_x), int(point_y)]))
print("Punto pertenece a cluster de color: ")
print(COLORS[unravel_index(prob.argmax(), prob.shape)[0]])
show()
