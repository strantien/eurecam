from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import imageio
import matplotlib.colors as mcolors

os.chdir("/home/trantien/Bureau/icj/doctorat/challenge_amies/Challenge_AMIES_EURECAM/")

import traitement as t
import optimalTransport as ot

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

#=======================================================#
                 #-TRACE DU COUPLAGE ENTRE X ET Y-#
#=======================================================#

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

#=======================================================#
                 #-FILM AVEC TRACE ENTRE i ET i+1-#
#=======================================================#

def plotTransportSuccessif(dataset):

    detfile = "data_detection/" + dataset + "/detection.txt"
    imgdir  = "data_detection/" + dataset + "/images/"

    file = open(detfile, "r")
    l0 = file.readline()
    l1 = file.readline()
    f,cx,cy = np.array(l1.split()).astype(int)
    file.close()

    detections = pd.read_csv(detfile,delimiter=" ",skiprows=2)
    images = np.unique(detections["#image"].values)

    plt.ion()
    fig = plt.figure()
    for i in images[:-1]:
        print("==> image : ",i)
        iNext = images[int(np.argwhere(images == i))+1] #oui, c'est sale
        maskX = (detections["#image"]==i)
        maskY = (detections["#image"]==iNext)
        X =  detections[maskX][:].values
        Y = detections[maskY][:].values
        couplage = computeTransport(X, Y)
        nbPtsNuage1 = X.shape[0]
        nbPtsNuage2 = Y.shape[0]

        plt.clf()

        try:
            im = imageio.imread(imgdir+"/image-"+str(i).zfill(3)+".png")
        except:
            im = imageio.imread(imgdir+"/image-"+str(i).zfill(3)+".jpg")

        height, width  = im.shape

        ixX = 0.5 * (X[:,1] / X[:,3] * f + cx)
        iyX = 0.5 * (X[:,2] / X[:,3] * f + cy)
        ixY = 0.5 * (Y[:,1] / Y[:,3] * f + cx)
        iyY = 0.5 * (Y[:,2] / Y[:,3] * f + cy)

        plt.imshow(im)
        plt.scatter(ixX, iyX, marker="+",color="blue")
        plt.scatter(ixY, iyY, marker="x",color="red")
        for i in range(nbPtsNuage1):
            for j in range(nbPtsNuage2):
                if couplage[i,j] == 1.:
                    plt.plot([ixX[i], ixY[j]], [iyX[i], iyY[j]], color = 'pink', linewidth = .7)
        plt.pause(0.5)  #plt.pause(0.05) #input('type enter to continue')
        plt.draw()

    plt.ioff()
    plt.show()

#=======================================================#
                 #-FILM AVEC TRAJECTOIRES COMPLETES-#
#=======================================================#

t1 = [[4.0, 151.0, -23.0, 179.0, 1.0], [5.0, 154.0, -26.0, 183.0, 1.0] ,[6, 152, -41, 287, 103], [7, 153, 26, 174, 1], [8, 153, 26, 174, 1], [9, 152, -22, 286, 105]]
t2 = [[4.0, 153.0, 25.0, 174.0, 2.0], [5.0, 153.0, 25.0, 174.0, 2.0], [6, 152, -27, 286, 107], [7, 158, 92, 174, 1], [8, 158, 92, 174, 1]]

trajectories = [t1, t2]


#colors = ['red', 'pink', 'blue', 'green', 'grey']
colors = list(mcolors.CSS4_COLORS.keys())
colors = colors[colors.index('red'):]

def plotComplete(dataset, trajectories):

    detfile = "data_detection/" + dataset + "/detection.txt"
    imgdir  = "data_detection/" + dataset + "/images/"

    file = open(detfile, "r")
    l0 = file.readline()
    l1 = file.readline()
    f,cx,cy = np.array(l1.split()).astype(int)
    file.close()

    detections = pd.read_csv(detfile,delimiter=" ",skiprows=2)
    images = np.unique(detections["#image"].values)

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
            for k in range(len(traj)):
                if traj[k][0] == i:
                    for j in range(k):
                        ix_trajJ = 0.5 * (traj[j][1] / traj[j][3] * f + cx)
                        iy_trajJ = 0.5 * (traj[j][2] / traj[j][3] * f + cy)
                        ix_trajJPlus1 = 0.5 * (traj[j+1][1] / traj[j+1][3] * f + cx)
                        iy_trajJPlus1 = 0.5 * (traj[j+1][2] / traj[j+1][3] * f + cy)
                        plt.plot([ix_trajJ, ix_trajJPlus1], [iy_trajJ, iy_trajJPlus1], color = colors[trajectories.index(traj)], linewidth = .4)
                        #if j > 0:
                        if j == k-1:
                            plt.arrow(0.5 * (ix_trajJ + ix_trajJPlus1), 0.5 * (iy_trajJ + iy_trajJPlus1), 0.5 * (ix_trajJPlus1 - ix_trajJ), 0.5 * (iy_trajJPlus1 - iy_trajJ), shape='full', lw=0, length_includes_head=True, head_width=5, color = colors[trajectories.index(traj)])

        plt.pause(0.1)
        #input('type enter to continue')
        plt.draw()

    plt.ioff()
    plt.show()

#=======================================================#
                 #-TEST-#
#=======================================================#

#plotComplete(dataset, trajectories)