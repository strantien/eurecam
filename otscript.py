from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import imageio

#=======================================================#
                   #- DONNEES -#
#=======================================================#

#changer le répertoire de travail en fonction de votre PC
os.chdir("/home/trantien/Bureau/icj/doctorat/challenge_amies/Challenge_AMIES_EURECAM/")
#os.chdir("pedro")
#os.chdir("kiki")

from trajectories_new import trajectory
from tuerTrajectoires import *

##Jeux de données du sujet

dataset = "001"
# dataset = "002"
# dataset = "003"
# dataset = "004"
# dataset = "005"
# dataset = "006"
# dataset = "007"
# dataset = "008"
# dataset = "009"
# dataset = "010"

detfile = "data_detection/" + dataset + "/detection.txt"
imgdir  = "data_detection/" + dataset + "/images/"

file = open(detfile, "r")
l0 = file.readline()
l1 = file.readline()
f,cx,cy = np.array(l1.split()).astype(int)
file.close()

detections = pd.read_csv(detfile,delimiter=" ",skiprows=2)
images = np.unique(detections["#image"].values)

# i = 114
# maskX = (detections["#image"]==i)
# maskY = (detections["#image"]==i+1)
# X =  detections[maskX][:].values
# Y = detections[maskY][:].values

#=======================================================#
                   #-FONCTIONS-#
#=======================================================#

#u[1:2] correspondent à x_u, y_u et u[-1] correspond à la hauteur
def distance(u,v):
    return(np.linalg.norm(u[1:2]-v[1:2]) + np.linalg.norm(u[-1]-v[-1]))

method = 'revised simplex'

def computeTransport(X, Y):

    nbPtsNuage1 = X.shape[0]
    nbPtsNuage2 = Y.shape[0]
    C = np.zeros([nbPtsNuage1, nbPtsNuage2])

    for i in range(nbPtsNuage1):
        for j in range(nbPtsNuage2):
            C[i,j] = distance(X[i],Y[j])

    c = np.ravel(C)
    #l'ordre de notre vecteur (pour le problème linèaire) est :
    #p11, p12, p13, p14, p15, p21, p22, ... etc
    #où "pij"= connexion entre le point i du nuage1 et le point j du nuage2.

    nbVariables = nbPtsNuage1*nbPtsNuage2
    nbContraintes = nbPtsNuage2 + nbPtsNuage1

    ##Cas 1 : autant de points dans les 2 nuages

    if nbPtsNuage1 == nbPtsNuage2:
        b_eq = np.ones(nbContraintes)

        A_eq = np.zeros([nbContraintes,nbVariables])
        #les premières lignes correspondent aux contraintes de sortie. c'est à dire pour
        #chaque point du Nuage1, il faut le connecter à un seul point du nuage 2.
        for i in range(nbPtsNuage1):
            for j in range(nbPtsNuage2):
                A_eq[i,i*nbPtsNuage2 +j] = 1
        #les dernieres lignes correspondent aux contraintes d'entrée. Pour chaque point
        #dans le nuage d'arrivée, il peut recevoir 1 seul nuage de sortie.
        for i in range(nbPtsNuage2):
            for j in range(nbPtsNuage1):
                A_eq[i + nbPtsNuage1, i + j*nbPtsNuage2] = 1

        P = linprog(c, A_eq = A_eq, b_eq = b_eq, bounds=(0,1), method=method)

    ##Cas 2 : nbPtsNuage1 > nbPtsNuage2

    elif nbPtsNuage1 > nbPtsNuage2:
        b_eq = np.zeros(nbContraintes)
        b_eq[nbPtsNuage1:] = np.ones(nbPtsNuage2)

        b_ub = np.zeros(nbContraintes)
        b_ub[:nbPtsNuage1] = np.ones(nbPtsNuage1)

        A_eq = np.zeros([nbContraintes,nbVariables])
        #les premieres lignes sont nulles ; les contraintes d'égalité sont :
        #dans le nuage d'arrivée, un point doit recevoir pile 1 point de sortie.
        for i in range(nbPtsNuage2):
            for j in range(nbPtsNuage1):
                A_eq[i + nbPtsNuage1, i + j*nbPtsNuage2] = 1

        A_ub = np.zeros([nbContraintes,nbVariables])
        #les contraintes d'inégalité sont les contrainte de sortie, cad pour
        #chaque point du Nuage1, il faut le connecter à au plus 1 point du nuage 2.
        for i in range(nbPtsNuage1):
            for j in range(nbPtsNuage2):
                A_ub[i,i*nbPtsNuage2 +j] = 1

        P = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds=(0,1), method=method)

    ##Cas 3 : nbPtsNuage1 < nbPtsNuage2

    else:
        b_eq = np.zeros(nbContraintes)
        b_eq[:nbPtsNuage1] = np.ones(nbPtsNuage1)

        b_ub = np.zeros(nbContraintes)
        b_ub[nbPtsNuage1:] = np.ones(nbPtsNuage2)

        A_eq = np.zeros([nbContraintes,nbVariables])
        #les dernières lignes sont nulles ; les contraintes d'égalité sont :
        #chaque point du nuage de départ doit être connecté à exactement 1 point
        #du nuage d'arrivée.
        for i in range(nbPtsNuage1):
            for j in range(nbPtsNuage2):
                A_eq[i,i*nbPtsNuage2 +j] = 1

        A_ub = np.zeros([nbContraintes,nbVariables])
        #les contraintes d'inégalité sont les contrainte d'entrée, cad que
        #chaque point du Nuage2 ne peut recevoir qu'au plus 1 point du nuage 2.
        for i in range(nbPtsNuage2):
            for j in range(nbPtsNuage1):
                A_ub[i + nbPtsNuage1, i + j*nbPtsNuage2] = 1

        P = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds=(0,1), method=method)

    ##Résultat : la matrice de connexion x

    #on 'unravel' le résultat P.x, à la main :
    x = np.array([P.x[k*nbPtsNuage2:(k+1)*nbPtsNuage2] for k in range(nbPtsNuage1)])
    #x est la matrice de couplage, cad la matrice des pij
    return(x)

def plotTransport(X, Y):

    nbPtsNuage1 = X.shape[0]
    nbPtsNuage2 = Y.shape[0]
    x = computeTransport(X, Y)

    plt.clf()
    #tracé des deux nuages de points initiaux
    plt.scatter(X[:,1], X[:,2], marker = '+', color = 'blue')
    plt.scatter(Y[:,1], Y[:,2], marker = 'x', color = 'red')
    #annotation des hauteurs de chaque point
    #X[i,-1] correspond à la hauteur de Xi
    for i in range(nbPtsNuage1):
        plt.annotate("h =" + str(X[i,-1]), (X[i,1], X[i,2]), color = 'blue', fontsize = 7)
    for j in range(nbPtsNuage2):
        plt.annotate("h =" + str(Y[j,-1]), (Y[j,1], Y[j,2]), color = 'red', fontsize = 7)
    #tracé des connexions
    for i in range(nbPtsNuage1):
        for j in range(nbPtsNuage2):
            if x[i,j] == 1.:
                plt.plot([X[i,1], Y[j,1]], [X[i,2], Y[j,2]], color = 'green', linewidth = .7)
    plt.grid()
    plt.show()

#plotTransport(X, Y)

#=======================================================#
                 #-TRAJECTOIRES-#
#=======================================================#

def createTrajInit():
    XInit = detections[(detections["#image"]==images[0])][:].values
    trajInit = [[XInit[i].tolist()] for i in range(XInit.shape[0])]
    for i in images[:-1]:
        iNext = images[int(np.argwhere(images == i))+1]
        maskY = (detections["#image"]==iNext)
        Y = detections[maskY][:].values
        trajInit = trajectory(trajInit, Y)
    return(trajInit)

trajectories = createTrajInit()

for k in range(len(trajectories)):
    np.savetxt('trajectory' + str(k) +'.txt', trajectories[k])

#list to be plotted = deadTrajectories = [t1, t2, ...]

#=======================================================#
                 #-FILM AVEC TRACE ENTRE i ET i+1-#
#=======================================================#

# plt.ion()
# fig = plt.figure()
# for i in images[:-1]:
#     print("==> image : ",i)
#     iNext = images[int(np.argwhere(images == i))+1] #oui, c'est sale
#     maskX = (detections["#image"]==i)
#     maskY = (detections["#image"]==iNext)
#     X =  detections[maskX][:].values
#     Y = detections[maskY][:].values
#     couplage = computeTransport(X, Y)
#     nbPtsNuage1 = X.shape[0]
#     nbPtsNuage2 = Y.shape[0]
#
#     plt.clf()
#
#     try:
#         im = imageio.imread(imgdir+"/image-"+str(i).zfill(3)+".png")
#     except:
#         im = imageio.imread(imgdir+"/image-"+str(i).zfill(3)+".jpg")
#
#     height, width  = im.shape
#
#     ixX = 0.5 * (X[:,1] / X[:,3] * f + cx)
#     iyX = 0.5 * (X[:,2] / X[:,3] * f + cy)
#     ixY = 0.5 * (Y[:,1] / Y[:,3] * f + cx)
#     iyY = 0.5 * (Y[:,2] / Y[:,3] * f + cy)
#
#     plt.imshow(im)
#     plt.scatter(ixX, iyX, marker="+",color="blue")
#     plt.scatter(ixY, iyY, marker="x",color="red")
#     for i in range(nbPtsNuage1):
#         for j in range(nbPtsNuage2):
#             if couplage[i,j] == 1.:
#                 plt.plot([ixX[i], ixY[j]], [iyX[i], iyY[j]], color = 'pink', linewidth = .7)
#     plt.pause(0.5)  #plt.pause(0.05) #input('type enter to continue')
#     plt.draw()
#
# plt.ioff()
# plt.show()

#=======================================================#
                 #-FILM AVEC TRAJECTOIRES COMPLETES-#
#=======================================================#

# t1 = [(4,1,2,1,9),(5,1,1,1,9),(6,0.5,0.5,1,9),(7,0.2,0.2,1,9), (8,0.1,0.2,1,9)]
# t2 = [(4,2,2,2,8),(5,1,1,2,8),(6,2.1,0.5,2,8),(7,2,2,2,8),(8,1,1,2,8),(9,2.1,0.5,2,8),(10,2,2,2,8),(11,1,1,2,8)]

# t1 = [[4.0, 151.0, -23.0, 179.0, 1.0], [5.0, 154.0, -26.0, 183.0, 1.0] ,[6, 152, -41, 287, 103]]
# t2 = [[4.0, 153.0, 25.0, 174.0, 2.0], [5.0, 153.0, 25.0, 174.0, 2.0], [6, 152, -27, 286, 107], [7, 158, 92, 174, 1]]
#
# trajectories = [t1, t2]
#

colors = ['red', 'pink', 'blue', 'green', 'grey']

plt.ion()
fig = plt.figure()
for i in images:
    print("==> image : ",i)
    plt.clf()

    try:
        im = imageio.imread(imgdir+"/image-"+str(i).zfill(3)+".png")
    except:
        im = imageio.imread(imgdir+"/image-"+str(i).zfill(3)+".jpg")

    height, width  = im.shape

    plt.imshow(im)

    for traj in trajectories:
        for k in range(len(traj)-1):
            if traj[k][0] == i:  #si la trajectoire part de l'image i
                ix_trajk = 0.5 * (traj[k][1] / traj[k][3] * f + cx)
                iy_trajk = 0.5 * (traj[k][2] / traj[k][3] * f + cy)
                ix_trajkPlus1 = 0.5 * (traj[k+1][1] / traj[k+1][3] * f + cx)
                iy_trajkPlus1 = 0.5 * (traj[k+1][2] / traj[k+1][3] * f + cy)
                plt.plot([ix_trajk, ix_trajkPlus1], [iy_trajk, iy_trajkPlus1], color = colors[trajectories.index(traj)], linewidth = .4)
                plt.arrow(ix_trajkPlus1, iy_trajkPlus1, 5, 5, shape='full', lw=0, length_includes_head=True, head_width=.05, color = colors[trajectories.index(traj)])
    plt.pause(1.5)
    #input('type enter to continue')
    plt.draw()

plt.ioff()
plt.show()
