from numpy import linalg, degrees, arctan2, sqrt
from random import uniform
from matplotlib.pyplot import plot, show, subplots, title, suptitle, xlabel, ylabel
from matplotlib.patches import Ellipse

ignore_characters = [' ', '\n']

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

def eigsorted(cov):
    vals, vecs = linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def plot_results(clusters, points_set, colors):
    fig, ax = subplots()
    title("EM for GMM")
    suptitle('Task 2', fontsize=16)
    ylabel('Axis y')
    xlabel('Axis x')
    for index, c in enumerate(clusters):
        xm = c.mu[0]
        ym = c.mu[1]
        cov = c.sigma
        vals, vecs = eigsorted(cov)
        theta = degrees(arctan2(*vecs[:, 0][::-1]))
        for j in range(0, 6):
            w, h = j * sqrt(vals)
            ell = Ellipse(xy=(xm, ym), width=w, height=h,
                          angle=theta, color=colors[index])
            ell.set_facecolor('none')
            ax.add_artist(ell)
            ax.scatter(xm, ym, marker="o", color=colors[index])
    for cluster_index in range(len(points_set)):
        for point_index in points_set[cluster_index]:
            ax.scatter(point_index[0], point_index[1],
                       color=colors[cluster_index])
    show()
