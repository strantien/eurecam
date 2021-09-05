#!/usr/bin/env python
# coding: utf-8

# In[340]:


import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import imageio


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


# In[352]:


# first trajectory 
IndexImage1=(detections["#image"]==images[0])
X_im1 =  detections[IndexImage1][:].values #dataframe corresponding to image 1
trajet1=X_im1
m=len(trajet1)


# In[353]:


trajet1


# In[354]:


for i in range(m,M+1):
    trajet1=np.append(trajet1,[[4,0,0,0,0]],0) #complete with zeros so that all vectors have same dimensions
    


# In[355]:


d=np.array([traj[0:5] for traj in trajet1]) #test to see how I can regain the non zero lines.
#will be usefull to construct X inside the function :Trajectory(trajet,Y) because X will be constructed using order of poits in trajet


# In[356]:


ttt=[]


# In[357]:


d


# In[358]:


d.tolist()


# In[359]:


len(d)


# In[360]:


for i in range(len(d)):
    if d[i,-1] !=0:
        ttt.append(d[i,0:5])


# In[361]:


ttt=np.array(ttt)


# In[362]:


ttt


# In[363]:


# first function of trajectories when P is constructed only with equalities-inequalites with 1
def trajectory(trajet,Y):
    imageActuelle=Y[0,0]
    #from trajet until now, we will construct the X to put in f=computeTransport, 
    #otherwise, if we let f treat initial dataframes it will not take into consideration the rearrangements done in previous steps while construction trajectories
    d=np.array([traj[-5:] for traj in trajet])
    d.tolist()
    ttt=[]
    for i in range(len(d)):
        if d[i,-1] !=0:
            ttt.append(d[i,-5:])
    X=np.array(ttt)       
    P=computeTransport(X, Y)
    nbPtsX = X.shape[0]
    nbPtsY = Y.shape[0]
    tr=np.array(trajet)
    newV=np.zeros((nbPtsX,5)) #the new points (from image i+1) to be added in our  trajectory 
    #IGNORE THIS COMMENT! dim=tr.shape #dimensions of trajet i.e if there are m points in an image, dim=(m,2)! m=nbPtsX 
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
    res=np.append(tr,newV,1)    
    trajet=res.tolist()
    return(trajet) 
#initially trajet has one column. Each line is a point. After one iteration trajet has two columns. Point trajet[m,n] goes to trajet[m,n+1] in the trajectory path 


# In[364]:


trajet1


# In[365]:


#first trajectory important to initialize any procedure
maskX=(detections["#image"]==4)
maskY = (detections["#image"]==5)
X1 =  detections[maskX][:].values #dataframe corresponding to image 4
Y1 = detections[maskY][:].values #dataframe corresponding to image 5
trajet=trajectory(trajet1,Y1)


# In[366]:


trajet


# In[367]:


d=np.array([traj[-5:] for traj in trajet])
d


# In[368]:


d.tolist()
ttt=[]
for i in range(len(d)):
    if d[i,-1] !=0:
        ttt.append(d[i,-5:])
X=np.array(ttt)


# In[369]:


X


# In[370]:


Y1 #just to verify if results are correct


# In[371]:


X1 #for verification


# In[372]:


P=computeTransport(X1,Y1)


# In[373]:


P #for verification


# In[374]:


#for verification when nbPntsY< nbPntsX i.e there are points that are associated to nothing~ stay unchanged
maskX=(detections["#image"]==5) #cause we want to start with image 5
maskY = (detections["#image"]==7)
#X =  detections[maskX][:].values #dataframe corresponding to image i
Y = detections[maskY][:].values #dataframe corresponding to image i+1
trajet=trajectory(trajet,Y)


# In[375]:


trajet


# In[376]:


X


# In[377]:


Y


# In[378]:


Y[0,0]


# In[379]:


P=computeTransport(X, Y)


# In[380]:


P


# In[61]:


for i in range(5):
    if (all(P[i,j] == 0 for j in range(4))):
        print(i)


# In[114]:


nbPtsX = X.shape[0]
nbPtsY = Y.shape[0]
newV=np.zeros((nbPtsX,2))


# In[117]:


for i in range(nbPtsX):
    for j in range(nbPtsY):
        if P[i,j]==1:
            newV[i,:]=Y[j,1:3]
    if (all(P[i,j] == 0 for j in range(nbPtsY))):  
                newV[i,:]=X[i,1:3]


# In[118]:


newV


# In[ ]:




