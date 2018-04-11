from em_functions import get_set_of_points, expectation_maximization, probability_from_point
from utils import parse_coordinates, plot_results
file_name = "points.txt"
COLORS = ['red', 'blue', 'green', 'yellow', 'orange', 'cyan', 'black']
MAX_ITERS = 100

# Read coordiantes
n_file = open(file_name, 'r')
coordinates = n_file.readlines()
n_file.close()

#number_clusters = 3
number_clusters = input("Number of clusters: \n")
number_clusters = int(number_clusters)

# Get coordinates
coordinates = parse_coordinates(coordinates)
# Get clusters probability
cluster_probability, clusters = expectation_maximization(
    coordinates, number_clusters, MAX_ITERS)

# Get sets of points for each cluster
points_set = get_set_of_points(
    cluster_probability, coordinates, number_clusters)

point_x = input("Enter x: \n")
point_y = input("Enter y: \n")
# Get best cluster for point
prob, cluster_color = (probability_from_point(
    clusters, [int(point_x), int(point_y)], COLORS))
message_cluster = "Point {}, {} belongs to {} cluster".format(
    point_x, point_y, cluster_color)
print(message_cluster)

plot_results(clusters, points_set, COLORS)
