from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import imageio
import math as m

#=======================================================#
                   #-FONCTIONS AUXILIARES-#
#=======================================================#
#point = [frame,x,y,z,h]
def distance(p1,p2):
    (frame1,x1,y1,z1,h1) = p1
    (frame2,x2,y2,z2,h2) = p2
    if frame2==frame1:
        return(m.sqrt((x2-x1)**2 + (y2-y1)**2 + min(min((h2-h1)**2,h2**2),h1**2)))
    else:
        return(m.sqrt((x2-x1)**2 + (y2-y1)**2 + min(min((h2-h1)**2,h2**2),h1**2)) + 100/abs(frame1-frame2) )

#=======================================================#
                   #-Transport Optimal-#
#=======================================================#

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
        #Contraintes du Nuage 1. Pour chaque point du Nuage1,
        # il faut le connecter à un seul point du Nuage 2.
        for i in range(nbPtsNuage1):
            for j in range(nbPtsNuage2):
                A_eq[i,i*nbPtsNuage2 +j] = 1
        #Contraintes du Nuage 2. Pour chaque point du Nuage2,
        # il faut le connecter à un seul point du Nuage 1.
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

    ##Cas 3 : nbPtsNuage1 < nbPtsNuage2 #Normalement, nous essayons d'eviter ce cas!

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
