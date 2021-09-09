import numpy as np
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import imageio
import os
import random
from functools import partial

os.chdir("/Users/pjaramil/Documents/gitFiles/Challenge_AMIES_EURECAM/")

import traitement as t
import optimalTransport as ot
import affichage
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
borderToleranceY0 = 10
tooBigSpeedNorm0 = 15 # This is used to help avoid linking noisy trajectories with actual trajectories
trendTolerance0 =  5 # Value at which we consider that our camera sees a chaotic environment or a one directional movement
velocityImportanceCoeff0 = 10 #Used to compared expected values with actualy matches.
heightCoeff0 = 10 # used to give importance to the heightCost
duplicateTolerance0 = 10 #tolerance de la "distance" pour differencier 2 personnes dans une meme image.
speedComparisonTolerance0 = 5 # tolerance qu'on utlise quand on compare des vitesses entre eux.
baseLifeTime0 = 10 # Temps de vie max qu'on considere raissonable quand on n'a pas de donnés. (unités: en frames)
lifeTimeTolerance0 = 5 # Tolerance pour comparer le temps de vie moyennes des trajectoires mortes.
directionChangeTolerance0 = 1# Tolerance  pour comparer le nombre de changements de sense d'une trajectoire.
noiseEqTolerance0 = 15 # Tolerance utlisée pour montrer egalité entre un point d'une image et un point qu'on sait c'est du bruit.
#==============================================================================#
#                      Initialization
#==============================================================================#
IndexImage1=(detections["#image"]==images[0]) # sous-tableau correspondant aux point de l'image 4.
X_im1   =  detections[IndexImage1][:].values #dataframe corresponding to image 1
trajets1 = list(X_im1)

#----Objets principales
# point = (frame,x,y,z,h)
# trajectory = [trajectoryState]
# trajectoryState = (frame, x, y, z, h, lastVelocity, meanVelocity)
# noiseList = [points]
trajectoiresMortes0 = [] # liste de trajectoires mortes
noiseList0 = [] #liste de points bruit
trajectoiresActives0 = [t.nouveauTrajet(p) for p in trajets1] # liste de trajectoires initiales
#==============================================================================#
#                      Linking Process
#==============================================================================#
# first function of trajectories when P is constructed only with equalities-inequalites with 1
def trajectory(trajets,Y, noiseList, deadTrajectories, velocityImportanceCoeff, heightCoeff,tooBigSpeedNorm): #Takes in a list of trajectories and a list of points. Returns list of trajectories
    #We assume that the points in Y are legitimate.
    if len(Y) == 0:
        #if no points next frame we could assume that we are looking at noise

        #OPTION 1: add trajectoires to noise list.
        #states = []
        #for traj in trajets:
        #    states = states + traj
        #noisePoints = [t.stateToPoint(st) for st in states]
        #noiseList = noiseList + noisePoints
        #returns([],noiseList)

        #OPTION 2: Continue trajectories whle arbitrarily making them stay put or dvanceOneFrameTraj
        #          or removing them (or adding them to noiseList, etc).
        newTrajets = []
        for traj in trajets:
            (lastVx,lastVy) = traj[-1][5]
            lastVNorm = lastVx**2 + lastVy**2
            if lastVNorm < tooBigSpeedNorm:
                stationnary = True
                newTraj = t.advanceOneFrameTraj(traj, stationnary)
                newTrajets.append(newTraj)
            else:
                stationnary = True
                newTraj = t.advanceOneFrameTraj(traj, stationnary)
                newTrajets.append(newTraj)
        return(newTrajets, noiseList)

        #OPTION 3:

    if len(trajets) == 0:
        #if there are no trajectories, then we take all points in the next frame as new trajectories.
        return([t.nouveauTrajet(p) for p in Y], noiseList)


    else:#both trajets and Y are not empty.
        X = [ t[-1] for t in trajets]
        #Calculate trendVector
        if len(deadTrajectories) == 0:
            vTrendVec = (0,0)
        else:
            meanVs = [traj[-1][-1] for traj in deadTrajectories]
            vTrendVec = t.meanOfTuples(meanVs)

        P = ot.computeTransport(X, Y, velocityImportanceCoeff, heightCoeff, vTrendVec, trendTolerance0)
        nbPtsX = len(trajets)
        nbPtsY = len(Y)
        newTrajets=[]

        if nbPtsX<=nbPtsY: #In this case we consider extra points to be new trajectories.
            #Nous réarrangeons le nuage de points Y
            orderedYPoints=[]
            for i in range(nbPtsX):
                for j in range(nbPtsY):
                    if P[i,j]==1:
                        orderedYPoints.append(Y[j])
            #Ceux qui sont pas utilisés, on les ajoute en tant que nouvelles trajectoires
            nouveauTrajs = []
            for j in range(nbPtsY):
                    if sum([P[i,j] for i in range(nbPtsX)]) == 0:
                        newTraj = t.nouveauTrajet(Y[j])
                        nouveauTrajs.append(newTraj) # Nous prenons celles qui n'ont pas matché dans cette liste.
            #On colle les POINTS (dans le bon ordre) dans Y avec chaque TRAJECTOIRE dans trajets
            for i in range(nbPtsX):
                print("X<=Y")
                newTraj = t.addPointToTraj(orderedYPoints[i],trajets[i])
                newTrajets.append(newTraj)
            #On rajoutte les trajectoires qu'on avait crée avant.
            newTrajets = newTrajets + nouveauTrajs
            return(newTrajets, noiseList)

        else: #there are more points in X than in Y.
            orderedTrajs=[]
            trajetsDeCote = []
            #Dans ce cas, nous réarrangeons les trajectoires cette fois.
            for i in range(nbPtsX):
                for j in range(nbPtsY):
                    if P[i,j]==1:
                        orderedTrajs.append(trajets[j]) #Les trajectoires matchent l'ordre de Y
            #Ceux qui ne sont pas connectés on les met de coté.
            for i in range(nbPtsX):
                if sum([P[i,j] for j in range(nbPtsY)]) == 0:
                    #stationnary = True
                    #nextTraj = t.advanceOneFrameTraj(trajets[i], stationnary)
                    nextTraj = trajets[i]
                    trajetsDeCote.append(nextTraj) # Nous prenons celles qui n'ont pas matché dans cette liste (on les oublie pas!)
            #Ici nous allons COLLER les POINTS dans Y et les TRAJECTOIRES pertinentes.
            for j in range(nbPtsY):
                print("X>Y")
                newTraj = t.addPointToTraj(Y[j], orderedTrajs[j])
                newTrajets.append(newTraj)
            #Ici nous rajoutons les trajectoires qu'on na pas connecté á des points dans Y.
            newTrajets = newTrajets+ trajetsDeCote
            return(newTrajets,noiseList)

#==============================================================================#
#                      Cleaning Process
#==============================================================================#
#First we clean the points in the next frame "Y" (c'est un array)
def cleanPoints(Y, noiseList, deadTrajectories, duplicateTol, noiseEqTolerance, minBordX, maxBordX, minBordY, maxBordY, borderToleranceX, borderToleranceY): #Prend un array et donne une liste comme réponse (et la nouvelle liste de noise!)
    newTrajectories = []
    noNoiseY = []
    cleanY = []
    for p in range(len(Y)) :
        if t.isNoise(Y[p], noiseList, noiseEqTolerance):
            tuplePoint = tuple(Y[p])
            noiseList.append(tuplePoint)
        else:
            tuplePoint = tuple(Y[p])
            noNoiseY.append(tuplePoint)
    #Remove duplicate points
    n = len(deadTrajectories)
    if n == 0:
        velocityTrend = (0,0)
    else:
        meanVs = [traj[-1][-1] for traj in deadTrajectories]
        velocityTrend = t.meanOfTuples(meanVs)
    for i in range(len(noNoiseY)):
        equalityCriteria = partial(t.peopleEqualityFunc,trendTolerance0, velocityTrend)
        if not t.tupleIsIn(noNoiseY[i], cleanY, equalityCriteria, duplicateTol):
            cleanY.append(noNoiseY[i])
    return(cleanY, noiseList)
#Now we clean trajectories. This means killing the ones that are leaving and eliminating the false ones.
def cleanTrajectories(Trajectories, noiseList, deadTrajectories, speedComparisonTolerance, lifeTimeTolerance, baseLifeTime, DirectionChangeTolerance):
  stillAlive = []
  for traj in Trajectories:
      #print("1", traj) Seems OK
      if t.leavesDomain(traj,xmin,xmax,ymin,ymax,speedComparisonTolerance):
          deadTrajectories.append(traj)
      elif t.remainsTooLong(traj, deadTrajectories, lifeTimeTolerance, baseLifeTime):
          noiseList = noiseList + traj
      elif t.changesDirectionTooMuch(traj, deadTrajectories, DirectionChangeTolerance):
          noiseList = noiseList + traj
      else:
          stillAlive.append(traj)
  return(stillAlive, noiseList, deadTrajectories)

#==============================================================================#
#                      Iteration
#==============================================================================#
for i in images[:-1]:
    iNext = images[int(np.argwhere(images == i))+1]
    maskY = (detections["#image"]==iNext)
    Y = detections[maskY][:].values # np.array
    Y = list(Y)
    Y = [tuple(p) for p in Y]
    #clean les points
    (Y,noiseList0) = cleanPoints(Y, noiseList0, trajectoiresMortes0, duplicateTolerance0, noiseEqTolerance0,  xmin, xmax, ymin, ymax, borderToleranceX0, borderToleranceY0)
    #clean trajectoires
    (trajectoiresActives0, noiseList0, trajectoiresMortes0) = cleanTrajectories(trajectoiresActives0, noiseList0, trajectoiresMortes0, speedComparisonTolerance0, lifeTimeTolerance0, baseLifeTime0, directionChangeTolerance0)
    #Linking Process
    (trajectoiresActives0, noiseList0) = trajectory(trajectoiresActives0, Y, noiseList0, trajectoiresMortes0, velocityImportanceCoeff0, heightCoeff0, tooBigSpeedNorm0)

resultTrajectories = [t.trajectoryToListOfPoints(traj) for traj in trajectoiresMortes0]

affichage.plotComplete(dataset, resultTrajectories)
#print(trajectoiresActives0)
#print(trajectoiresMortes0)
