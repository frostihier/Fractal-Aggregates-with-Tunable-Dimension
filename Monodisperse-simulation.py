import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import copy
from scipy.spatial import KDTree
#!pip install numpy-quaternion
import quaternion as quat
import torch
from torch import tensor
from time import perf_counter

## Collect all the font names available to matplotlib
import matplotlib.font_manager as fm
import matplotlib as mpl

font_names = [f.name for f in fm.fontManager.ttflist]
#print(font_names)

# Edit the font
mpl.rcParams['font.family'] = 'Montserrat'


## # SIMULATION PARAMETERS

k = 1.2      # fractal parameter (form factor)
D = 2        # fractal dimension, should be between 1 to 3
a = 1        #  particles radius (for monodisperse case uniform)


## FUNCTIONS

# mass of a sphere with Radius a
def compute_mass(a):
  m_pp = 4/3*np.pi*a*a*a
  return m_pp


# define class for primary particle (sphere)
class Partikel:

    def __init__(self, pos, radius):
      if not isinstance(pos, np.ndarray):
        raise TypeError("Die Position muss ein NumPy-Array sein.")

      self.pos = pos
      self.radius = radius
      self.masse = compute_mass(radius)
      self.rg = np.sqrt(3/5)*self.radius

    def __deepcopy__(self, memo):
      cls = self.__class__
      result = cls(copy.deepcopy(self.pos, memo), copy.deepcopy(self.radius, memo))
      result.masse = copy.deepcopy(self.masse, memo)
      memo[id(self)] = result
      return result


# Function which creates a dimer
# takes as input radius and center of mass
def create_dimer(r, com = np.array([0.,0.,0.])):
    # create particles composing dimer
    particle1 = Partikel(com, r)

    # generate two random angles in the range 0 to 2π
    theta = random.uniform(0, 2*math.pi)
    phi = random.uniform(0, 2*math.pi)

    # calculate x, y, z coordinates of the random point on the sphere
    x = particle1.pos[0] + 2 * r * math.sin(phi) * math.cos(theta)
    y = particle1.pos[1] + 2 * r * math.sin(phi) * math.sin(theta)
    z = particle1.pos[2] + 2 * r * math.cos(phi)

    x = particle1.pos[0] + 2 * r * math.sin(phi) * math.cos(theta)
    y = particle1.pos[1] + 2 * r * math.sin(phi) * math.sin(theta)
    z = particle1.pos[2] + 2 * r * math.cos(phi)

    # particle 2 in 3D space
    particle2 = Partikel(np.array([x,y,z]), r)

    dimer = [particle1, particle2]

    return dimer



# function to compute center of mass of a cluster
def center_of_mass(polymere):
    # compute center of mass
    center_of_mass = np.array([0., 0., 0.])
    m_ges = 0
    # sum over position and weights
    for particle in polymere:
        center_of_mass += particle.masse * particle.pos
        m_ges += particle.masse

    center_of_mass = center_of_mass / m_ges

    return center_of_mass



# compute radius of gyration out of the radii of gyration of the previous two clusters
# see formula x in the Thomchuk et al.
def radius_gyration(polymere1, polymere2, rg1, rg2, gamma):
    # compute mass of clusters
    m1 = m2 = 0
    for p, q in zip(polymere1, polymere2):
        m1 += p.masse
        m2 += q.masse
    # total mass
    m = m1 + m2
    # compute radius of gyration according to thomchuk
    Rg = np.sqrt(m1*rg1**2/m + m1*rg2**2/m + m1*m2*gamma*gamma/(m*m))

    return Rg



# compute cluster distance to satisfy the basic equation
def compute_gamma(polymere1, polymere2, Rg1, Rg2):

  # mass of the clusters
  m1 = m2 = 0
  for p,q in zip(polymere1, polymere2):
    m1 += p.masse
    m2 += q.masse
  # total mass
  m = m1 + m2

  # current aggregation number
  N = len(polymere1) + len(polymere2)

  # distance between center of mass of the two clusters
  gamma = np.sqrt((m*m*a*a*(N/k)**(2/D))/(m1*m2) - m*Rg1*Rg1/m2 - m*Rg2*Rg2/m1)

  return gamma



# shift cluster
def shift_cluster(polymere1, polymere2, gamma):

  #center of mass of the clusters (coms)
  com1 = center_of_mass(polymere1)
  com2 = center_of_mass(polymere2)

  # distance between coms
  dist = np.linalg.norm(com1 - com2)

  # verbindungsvektor zwischen coms und normieren
  # normalised connection vector between coms
  con_vec = com2 - com1
  con_vec = con_vec / np.linalg.norm(con_vec)

  # create copy of cluster 2
  polymere3 = copy.deepcopy(polymere2)

  # shift all particles of cluster 3 along this vector by the amount of gamma minus the present distance.
  for particle in polymere3:
    particle.pos += abs(gamma-dist) * con_vec

  return polymere3




# checks if cluster are connected but do not overlap
def check_connection(polymere1, polymere2):  # später thresh,rmin und rmax übergeben

    # merge cluster
    cluster = polymere1 + polymere2

    # extract positions
    positions = [particle.pos for particle in cluster]
    # print(positions)

    # create kd-Tree out of cluster positions
    tree = KDTree(positions)

    # Calculate all pairs of points that have a distance of less than 2*radius
    thresh = 2*a - 0.001
    # define range at which two particles are considered as connected  ('covalent bound')
    rmin = 2*a + 0.001*a
    rmax = 2*a + 0.05*a   # or choose + 0.1*(2*a) -> 10%

    # checks if there is at least one overlapping pair that is the
    # list of particle pairs with distance lower than 2*a
    pairs = tree.query_pairs(r=thresh)
    if len(pairs) > 0:
        # print("Overlap")
        check = False

    # otherwise check if there is at least one pair which is connected
    else:
        # calculate again all pairs with a distance up to rmax.
        # then, filter for pairs which have a distance between rmin and rmax
        pairs = tree.query_pairs(r=rmax)
        filtered_pairs = set((i, j) for (i, j) in pairs if rmin < np.linalg.norm(positions[i] - positions[j]) < rmax)

        # filter again, so that only indices from cluster 1 are compared with indices from cluster 2.
        # i.e. sort out all pairs that come from the same half.
        filtered_pairs = [(i, j) for (i, j) in filtered_pairs if
                          ((i < len(polymere1) and j >= len(polymere1)) or (
                                      j < len(polymere1) and i >= len(polymere1)))]

        if len(filtered_pairs) > 0:
            check = True
            # print('Clusters connected!')
        else:
            check = False
            # print('Clusters not connected' )

    return check





## ROTATION WITH QUATERNION THEORY

# concepts from quaterion theory are used to implement a fast rotation of the clusters around their respective
# center of mass

# for a good introduction / explanation see !!!!!!!!!!!!!!!!________________!!!!!!!!!!

# function which computes the roation of a cluster using quaterions
def compute_quat_rot(vector, axis, theta):

  # concatenate a zero to the begin of a vector to convert it to a quaterion
  vector = np.concatenate(([0], vector))
  rot_axis = np.concatenate(([0], axis))

  # fast rotation with quaterions
  axis_angle = (theta*0.5) * rot_axis/np.linalg.norm(rot_axis)

  vec = quat.quaternion(*vector)
  qlog = quat.quaternion(*axis_angle)

  q = np.exp(qlog)
  v_prime = q * vec * np.conjugate(q)

  # keep only imaginary part
  v_prime_vec = v_prime.imag

  return v_prime_vec



# rotiere cluster sodass sie sich nur berühren aber nicht überlappen

# vollführe mit Quaternion berechnung
# # rotiere Cluster um 'angle' in Grad
def rotate_cluster_quat(polymere, axis, theta):
    polymere_new = copy.deepcopy(polymere)

    # center of mass des clusters, um das rotiert wird
    com_new = center_of_mass(polymere_new)

    # Rotation durchführen
    for particle in polymere_new:
        # Berechne die relative Position des Partikels zum Center of Mass
        rel_pos = particle.pos - com_new

        # Rotiere die relative Position zuerst um phi in der x-Achse und danach um theta in der y-Achse
        rotated_rel_pos = compute_quat_rot(rel_pos, axis, theta)

        # Addiere die gedrehte relative Position zum Center of Mass, um die neue Position des Partikels zu erhalten
        particle.pos = rotated_rel_pos + com_new

    return polymere_new



# this function rotates two cluster until they are connected but not overlapping
# Therefore clusters are rotated alternately around random axes in 3d (Monte Carlo approach).
def rotate_until_connected_montecarlo(polymere1, polymere2):

  # counter for termination criterion, counts how often a cluster was rotated
  # if more often than stop was rotated, function breaks
  N = 0
  stop = 20000

  #counts how often two clusters cound not be connected even after many rotations.
  # global variable because it continues to be counted until the cluster is completely finished
  global fail_counter
  # termination criterion, if failed to often break off creation of cluster
  fail_stop = 550

  # while clusters are not connected: random rotation
  while check_connection(polymere1, polymere2) == False:

    # choose random angle
    theta = random.uniform(0, 2*math.pi)
    # choose random axis: x,y,z
    axis = random.choice([np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])])
    # rotate cluster 1 and cluster 2 alternately
    if N % 2 == 0:
      polymere1 = rotate_cluster_quat(polymere1, axis, theta)
    else:
      polymere2 = rotate_cluster_quat(polymere2, axis, theta)

    # termination criterion
    if N > stop:
      print(f"Rotated over {stop} times - no connection was found")
      fail_counter += 1
      print(fail_counter)
      if fail_counter == fail_stop:
          raise RuntimeError('To much connection fails, abort creation of the cluster')
      return polymere1, False

    N += 1

  return polymere1, polymere2



## FUNCTIONS FOR THE CREATION OF HIERARCHICAL CLUSTERS


# function merges to cluster by shifting them and subsequently rotating them until they are connected
def merge_cluster(polymere1, polymere2, Rg1, Rg2):

  #shift cluster 2
  gamma = compute_gamma(polymere1, polymere2, Rg1, Rg2)
  polymere2 = shift_cluster(polymere1,polymere2,gamma)

  # rotate until connected
  polymere1, polymere2 = rotate_until_connected_montecarlo(polymere1, polymere2)

  # if it fails, return False
  if polymere2 == False:
    return polymere1, False, Rg1, gamma

  # radius of gyration
  Rg = radius_gyration(polymere1, polymere2, Rg1, Rg2, gamma)

  return polymere1, polymere2, Rg, gamma



# function which creates a new cluster consisting out of 4 particles (two dimers)
def new_cluster():

    # new dimer
    cluster1 = create_dimer(1)
    # first, compute radius of gyration for the merge_cluster function
    Rg1 = radius_gyration([cluster1[0]], [cluster1[1]], np.sqrt(3/5)*cluster1[0].radius, np.sqrt(3/5)*cluster1[1].radius, 2)

    # try to connect with second dimer
    merged = False
    while merged == False:

        cluster2 = create_dimer(1)
        # compute radius of gyration of second dimer
        Rg2 = radius_gyration([cluster2[0]], [cluster2[1]], np.sqrt(3/5) * cluster2[0].radius,
                                      np.sqrt(3/5) * cluster2[1].radius, 2)

        cluster1, cluster2, Rg1, gamma = merge_cluster(cluster1, cluster2, Rg1, Rg2)

        if cluster2 is not False:
            merged = True
            # print('Cluster succesfully connected')

    # berechne radius of gyration of the unified tetramer cluster
    Rg = radius_gyration(cluster1, cluster2, Rg1, Rg2, gamma)

    return cluster1 + cluster2, Rg




# the main funtion to create a hierarchical cluster,
# gets as input the order -> cluster consist of 2^order particles

#this function is recursive, it always calls itself with order-1.
#when the order reaches two, a new cluster of 4 particles is created,
#these small clusters are then joined together to form larger and larger clusters.
def create_cluster(order):
    if order == 2:
        # creates new tetramere-cluster (4 particles)
        return new_cluster()

    # otherwise function calls itselfs
    cluster1, Rg1 = create_cluster(order-1)

    merged = False
    while merged == False:

        cluster2, Rg2 = create_cluster(order-1)
        cluster1, cluster2, Rg, gamma = merge_cluster(cluster1, cluster2, Rg1, Rg2)

        if cluster2 is not False:
            merged = True

    return cluster1 + cluster2, Rg




# function for plotting clusters
# creates 3d-spheres for each particle in the cluster
# spheres have the color c
def plot_3d_cluster(cluster, c):

  for particle in cluster:
    r = particle.radius
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = r * np.sin(v) * np.cos(u) + particle.pos[0]
    y = r * np.sin(v) * np.sin(u) + particle.pos[1]
    z = r * np.cos(v) + particle.pos[2]
    ax.plot_wireframe(x, y, z, color=c)
  #plot center of mass of cluster
  #com = center_of_mass(cluster)
  #ax.scatter(com[0], com[1], com[2], color='r', s = 40)



## FUNCTIONS TO COMPUTE THE STRUCTURE FACTOR OF A CLUSTER

# functions to compute the 1d structure factor according to Tomchuk et al.
# here the structure factor is approximately calculated by dividing the intensity I(q) by the averaged particle form factor
# <p(q)> so that S(q) = I(q)/p(q)
# for monodisperse particles this approximation is exact


# function to compute the (scattering) intensity I(q) of a cluster
# parallel computation on the GPU
def compute_intensity_GPU(cluster, q_array):

    # convert q_array, the grid of wavevectors, to a pytorch tensor
    Gqs = tensor(q_array, dtype = torch.float64).to(device='cuda')

    # extract positions and masses of the clusters send them to the GPU
    pos = np.array([particle.pos for particle in cluster])
    Gpos = tensor(pos).to(device='cuda')
    masses = np.array([particle.masse for particle in cluster])
    Gmasses = tensor(masses).to(device='cuda')
    radii = np.array([particle.radius for particle in cluster])
    Gradii = tensor(radii).to(device='cuda')

    # array which stores the
    I = torch.zeros(len(q_array), dtype = torch.float64, device='cuda')

    # number of particles in cluster
    L = range(len(cluster))

    # for each q in q_array we compute in parallel the sum over the particles
    for i in L:
        for j in L:
            if i != j:
                # compute distance betwenn to particles in a cluster
                d = torch.norm(Gpos[i] - Gpos[j])
                # sum to get intensity
                I += b_gpu(Gqs*Gradii[i], Gmasses[i]) * b_gpu(Gqs*Gradii[j], Gmasses[j]) * torch.sin(Gqs*d)/(Gqs*d)

            # if i = j: according to L'Hospital: sin(x)/x -> 1 if x -> 0
            else:
                I += b_gpu(Gqs*Gradii[i], Gmasses[i]) * b_gpu(Gqs*Gradii[j], Gmasses[j])

    # number particles
    N = len(cluster)
    # mean volume -> squared below
    v = np.mean(masses)
    # send back to CPU
    I = np.array(I.cpu())
    # normalise intensity
    I = I/(N*v*v)

    return I


# scattering amplitude of a spherical particle
def b_gpu(x, masse):
    res = 3*masse*(torch.sin(x)-x*torch.cos(x))/(x*x*x)
    return res


# averaged particle form factor
def compute_form_factor_GPU(cluster, q_array):

    # convert q_array, the grid of wavevectors, to a pytorch tensor
    Gqs = tensor(q_array, dtype=torch.float64).to(device='cuda')
    # extract positions and masses of the clusters send them to the GPU
    masses = np.array([particle.masse for particle in cluster])
    Gmasses = tensor(masses).to(device='cuda')
    radii = np.array([particle.radius for particle in cluster])
    Gradii = tensor(radii).to(device='cuda')

    # array of computed form factors in dependence of q
    GP = torch.zeros(len(q_array), dtype = torch.float64, device='cuda')

    # number particles
    N = len(cluster)

    # sum over b's
    for i in range(len(cluster)):
        GP += b_gpu(Gqs * Gradii[i], Gmasses[i])**2
    GP = GP / N

    # Averaged volume (here equivalent to mass since density is assumed to be uniform)
    v = np.mean(masses)

    GP = np.array(GP.cpu())
    # divide by mean volume
    GP = (GP/(v*v))

    return GP


# divide I/P to get S
def compute_SF_1d(cluster, q_array):
    I = compute_intensity_GPU(cluster, q_array)
    P = compute_form_factor_GPU(cluster, q_array)
    return I/P




###  --------- SIMULATION ---------------
##

# here you can modify simulation parameters again
# D = 1.65
# k = 1.8

o = 7   # order

# create clusters
for i in range(1):

    start_time = perf_counter()

    cluster1, Rg1 = create_cluster(o)

    end_time = perf_counter()
    duration = end_time - start_time
    print("Dauer der Operation: ", duration, "Sekunden")


## PLOT CLUSTER

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_3d_cluster(cluster1, c='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_aspect('equal', adjustable='box')
plt.show()



## COMPUTE STRUCTURE FACTOR
##
# Wave vector
q1 = np.linspace(0.001,1,300)
q2 = np.linspace(1,6,400)
qs = np.concatenate((q1, q2))

# compute structure factor of cluster
SF1d = compute_SF_1d(cluster1, qs)


## plot structure factor


plt.figure(figsize=(8,8))
plt.xlabel("qa", fontsize=20)
plt.ylabel("S(q)", fontsize=20)
plt.axhline(y=1, c='black', ls = '--', lw = 1.7)
ax = plt.gca()
ax.xaxis.set_tick_params(which='major', size=6, width=2, direction='out')
ax.yaxis.set_tick_params(which='major', size=6, width=2, direction='out')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Powder averaged structure factors', fontsize = 20)

# change for non-logscale plot
#plt.ylim(0.3, 2.2)
plt.xscale('log')
plt.yscale('log')

# plot structure factor
plt.plot(qs, SF1d, c = 'black', label = 'Structure factor')

# first correlation shell?
#plt.axvline(3.7)

plt.legend(fontsize=17, handletextpad = 0.5, edgecolor = 'black', frameon = True, fancybox = False, facecolor = None)
#os.makedirs(resultpath + 'SF-plots', exist_ok = True)
#plt.savefig('Figures/All-SF-logscale.png', bbox_inches='tight', pad_inches=0.02, transparent = True)
plt.show()





