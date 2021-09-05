#!/usr/bin/env python
# coding: utf-8

# In[340]:


import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import imageio
import os
os.chdir("/home/trantien/Bureau/icj/doctorat/challenge_amies/Challenge_AMIES_EURECAM/")


# In[341]:


from scipy.optimize import linprog


# In[342]:


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


# In[343]:


detfile = "data_detection/" + dataset + "/detection.txt"
imgdir  = "data_detection/" + dataset + "/images/"


# In[344]:


file = open(detfile, "r")
l0 = file.readline()
l1 = file.readline()
f,cx,cy = np.array(l1.split()).astype(int)
file.close()


# In[345]:


detections = pd.read_csv(detfile,delimiter=" ",skiprows=2)
images = np.unique(detections["#image"].values) #keep image indices because there are numbers missing between 4 and 167


# In[346]:


images.size


# In[347]:


counter=[] #countes number of points in each image


# In[348]:


for i in images:
    df=detections[detections['#image']==i]
    n=df.shape[0]
    counter.append(n)


# In[349]:


M=max(counter)
M #the maximum number of points in an image


# In[350]:


def distance(u,v):
    return(np.linalg.norm(u[1:2]-v[1:2]) + np.linalg.norm(u[-1]-v[-1]))


# In[351]:


method = 'revised simplex'

def computeTransport(X, Y):

    nbPtsNuage1 = X.shape[0]
    nbPtsNuage2 = Y.shape[0]
    C = np.zeros([nbPtsNuage1, nbPtsNuage2])

    for i in range(nbPtsNuage1):
        for j in range(nbPtsNuage2):
            C[i,j] = distance(X[i],Y[j])

    c = np.ravel(C)
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


# first function of trajectories when P is constructed only with equalities-inequalites with 1
def trajectory(trajet,Y):
    imageActuelle=Y[0,0]
    #from trajet until now, we will construct the X to put in f=computeTransport, 
    #otherwise, if we let f treat initial dataframes it will not take into consideration the rearrangements done in previous steps while construction trajectories
    d=np.array([traj[-1] for traj in trajet])
#    d.tolist()
    ttt=[]
    for i in range(d.shape[0]):
        if d[i,-1] !=0:
            ttt.append(d[i,:])
    X=np.array(ttt)   
    X=Y
    P=computeTransport(X, Y)
    nbPtsX = X.shape[0]
    nbPtsY = Y.shape[0]
    newV=np.zeros((nbPtsX,5)) #the new points (from image i+1) to be added in our  trajectory 
    if nbPtsX==nbPtsY:        
        for i in range(nbPtsX):
            for j in range(nbPtsY):
                if P[i,j]==1:
                    newV[i,:]=Y[j,0:5]          
    elif nbPtsX > nbPtsY:
        for i in range(nbPtsX):
            for j in range(nbPtsY):
                if P[i,j]==1:
                    newV[i,:]=Y[j,0:5]
            if (all(P[i,j] == 0 for j in range(nbPtsY))):  
                newV[i,:]=X[i,0:5]
    else:
        Matchedpoints=[] #Rem: Not all points in Y will be associated to some point of X
        k=0
        for i in range(nbPtsX):
            for j in range(nbPtsY):
                if P[i,j]==1:
                    newV[i,:]=Y[j,0:5]
                    k=j
                    Matchedpoints.append(k)
        for j in range(nbPtsY):
            if j not in Matchedpoints:
                newV=np.append(newV,Y[j,0:5])
    m=len(newV)
    for i in range(m,M+1):
        newV=np.append(newV,[[imageActuelle,0,0,0,0]],0)
    for k in range(len(trajet)):
        trajet[k].append(newV[k,:])
    return(trajet) 
