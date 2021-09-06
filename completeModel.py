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
borderToleranceX0 = 10
borderToleranceY0 = 30
nbPastVelocities0 = 3 #vitesses qu'on va considerer pour faire des moyennes des dernieres vitesses d'une trajectoire
nbTrajectoires0 = 10 # nb de trajectoires qu'on va considerer pour quand on fait de stat sur l'ensemble de trajectoires mortes
duplicateTolerance0 = 20 #tolerance de la "distance" pour differencier 2 personnes dans une meme image.
speedTolerance0 = 10 # tolerance qu'on utlise quand on compare des vitesses entre eux.
baseLifeTime0 = 1000 # Temps de vie max qu'on considere raissonable quand on n'a pas de donnés. (unités: en frames)
lifeTimeTolerance0 = 10 # Tolerance pour comparer le temps de vie moyennes des trajectoires mortes.
directionChangeTolerance0 = 4 # Tolerance  pour comparer le nombre de changements de sense d'une trajectoire.
noiseEqTolerance0 = 10 # Tolerance utlisée pour montrer egalité entre un point d'une image et un point qu'on sait c'est du bruit.
#==============================================================================#
#                      Initialization
#==============================================================================#
IndexImage1=(detections["#image"]==images[0]) # sous-tableau correspondant aux point de l'image 4.
X_im1   =  detections[IndexImage1][:].values #dataframe corresponding to image 1
trajets1 = list(X_im1)

#----Objets principales
trajectoiresMortes0 = [] # liste de trajectoires mortes
noiseList0 = [] #liste de points bruit initiales
trajectoiresActives0 = [t.nouveauTrajet(p) for p in trajets1] # liste de trajectoires initiales
#==============================================================================#
#                      Linking Process
#==============================================================================#
# first function of trajectories when P is constructed only with equalities-inequalites with 1
def trajectory(trajets,Y):
    X = np.array([ t[-1] for t in trajets])
    P = ot.computeTransport(X, Y)
    nbPtsX = X.shape[0]
    nbPtsY = Y.shape[0]
    Ylist=[]
    Xlist=[]
    newTrajets=[]
    if nbPtsX<=nbPtsY:
        #Nous réarrangeons le nuage de points Y
        #Ceux qui sont en trop on les oublie
        for i in range(nbPtsX):
            for j in range(nbPtsY):
                if P[i,j]==1:
                    Ylist.append(tuple(Y[j,:]))
        #On colle les points (dans le bon ordre) Y avec chaque trajectoire de X
        for i in range(len(trajets)):
            trajets[i].append(Ylist[i])
            newTrajets.append(trajets[i])
        #Le reste de Points dans Y sont oublies en tant que 'bruit'
    else:
        #Dans ce cas, nous réarrangeons les trajectoires dans trajets au lieu des points dans Y.
        Xcote = [] # Ici nous allons mettre les trajectoires que ne seront pas connectées avec un point dans Y
        Ylist = [tuple(p) for p in list(Y)]
        for i in range(nbPtsX):
            for j in range(nbPtsY):
                if P[i,j]==1:
                    Xlist.append(trajets[i]) #Nous rearrangeons les trajectoires. Il faut quelles matchent l'ordre de Y
            if sum([P[i,j] for j in range(nbPtsY)]) == 0:
                Xcote.append(trajets[i]) # Nous prenons celles qui n'ont pas matché dans cette liste (on les oublie pas!)
        #Ici nous allons coller les points de Ylist et les trajectoires pertinentes.
        for i in range(nbPtsY):
            Xlist[i].append(Ylist[i])
            newTrajets.append(Xlist[i])
        #Ici nous rajoutons les trajectoires qu'on na pas connecté á des points dans Y.
        newTrajets = newTrajets + Xcote
    return(newTrajets)

#==============================================================================#
#                      Cleaning Process
#==============================================================================#
#First we clean the points in the next frame "Y" (c'est un array)
def cleanPoints(Y, noiseList, duplicateTol, noiseEqTolerance): #Prende un array et donne un array comme réponse (et la nouvelle liste de noise!)
    noNoiseY = []
    YClean = []
    for p in (Y.shape([0])) :
        if t.isNoise(Y[p], noiseList, noiseEqTolerance):
            tuplePoint = tuple(Y[p])
            noiseList.append(tuplePoint)
        else:
            tuplePoint = tuple(Y[p])
            noNoiseY.append(tuplePoint)
    noNoiseY = np.array(noNoiseY)
    YClean.append(tuple(noNoiseY[0]))
    for i in (noNoiseY.shape([0])):
        if t.isIn(noNoise[i], YClean, t.peopleEqualityFunc, duplicateTol):
            #On fait rien!
        else:
            pointTuple = tuple(noNoiseY[i])
            YClean.append(pointTuple)
    YClean = np.array(YClean)
    return(YClean, noiseList)
#Now we clean trajectories. This means killing the ones that are leaving and eliminating the false ones.
def cleanTrajectories(Trajectories, noiseList, deadTrajectories,speedTolerance, nbPastVelocities, lifeTimeTolerance, baseLifeTime, DirectionChangeTolerance):
  stillAlive = []
  for traj in Trajectories:
      if t.leavesDomain(traj,xmin,xmax,ymin,ymax,speedTolerance, nbPastVelocities):
          deadTrajectories.append(traj)
      elif t.remainsTooLong(traj, deadTrajectories, lifeTimeTolerance, baseLifeTime):
          noiseList = noiseList + traj
      elif t.changesDirectionTooMuch(traj, deadTrajectories, DirectionChangeTolerance):
          noiseList = noiseList + traj
      else:
          stillAlive.append(traj)
  return(stillAlive, noiseList, deadTrajectories)
