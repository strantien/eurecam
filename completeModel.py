import numpy as np
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import imageio
import os

os.chdir("/home/trantien/Bureau/icj/doctorat/challenge_amies/Challenge_AMIES_EURECAM/")

import traitement as t
import optimalTransport as ot


#==============================================================================#
#                  Main Objects
#==============================================================================#
#Les objets avec lesquels nous allons travailler (pour le moment).
trajectoiresMortes = [] # liste de trajectoires
trajectoiresActives = [] # liste de trajectoires
noiseL = [] #liste de points
#==============================================================================#

#==============================================================================#
#                        System code
#==============================================================================#

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

detections = pd.read_csv(detfile, delimiter=" ", skiprows=2)
images = np.unique(detections["#image"].values) #keep image indices because there are numbers missing between 4 and 167

#==============================================================================#
#                        Parameters
#==============================================================================#

xmin = min(detections["x"].values)
xmax = max(detections["x"].values)
ymin = min(detections["y"].values)
ymax = max(detections["y"].values)

#==============================================================================#
#                      Initialization
#==============================================================================#
IndexImage1=(detections["#image"]==images[0]) # sous-tableau correspondant aux point de l'image 4.
X_im1   =  detections[IndexImage1][:].values #dataframe corresponding to image 1
trajets1 = list(X_im1)
trajets1 = [[tuple(p)] for p in trajets1] # liste de trajectoires initiales

#==============================================================================#
#                      Initialization
#==============================================================================#

# first function of trajectories when P is constructed only with equalities-inequalites with 1
def trajectory(trajets,Y):

    newTrajectories = [] #


    X = np.array([ t[-1] for t in trajets])
    P = ot.computeTransport(X, Y)
    nbPtsX = X.shape[0]
    nbPtsY = Y.shape[0]
    Ylist=[]
    newTrajets=[]
    Xlist=[]

    if nbPtsX<=nbPtsY:
        for i in range(nbPtsX):
            for j in range(nbPtsY):
                if P[i,j]==1:
                    Ylist.append(tuple(Y[j,:]))
        for i in range(len(trajets)):
            trajets[i].append(Ylist[i])
            newTrajets.append(trajets[i])

    else:
        Xcote = []
        Ylist = [tuple(p) for p in list(Y)]
        for i in range(nbPtsX):
            for j in range(nbPtsY):
                if P[i,j]==1:
                    Xlist.append(trajets[i])
            if sum([P[i,j] for j in range(nbPtsY)]) == 0:
                Xcote.append(trajets[i])
        #newTrajets = Xlist
        for i in range(nbPtsY):
            Xlist[i].append(Ylist[i])
            newTrajets.append(Xlist[i])
        newTrajets = newTrajets + Xcote

    return(newTrajets)

for i in images[:-1]:
    iNext = images[int(np.argwhere(images == i))+1]
    maskY = (detections["#image"]==iNext)
    Y = detections[maskY][:].values
    print(len(trajets1), iNext, Y.shape[0])
    trajets1 = trajectory(trajets1, Y)

