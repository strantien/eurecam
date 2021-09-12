from scipy.optimize import linprog
import numpy as np
import math as m
import traitement as t
#=======================================================#
                   #-FONCTIONS AUXILIARES-#
#=======================================================#
meanPreference0 = 0.75
#point = (frame,x,y,z,h)
#traj = [trajState]
#trajState = (frame, x, y,z h, vf, vm)
def distance(p1,p2, velocityCoeff, heightCoeff, VelocityTrendVector, trendTolerance): #prends deux tuples de taille 7 et 5 respectivement
    (frame1,x1,y1,z1,h1,lastVelocity, meanVelocity) = p1
    (frame2,x2,y2,z2,h2) = p2
    vx = lastVelocity[0]
    vy = lastVelocity[1]
    meanVx = meanVelocity[0]
    meanVy = meanVelocity[1]
    expectedX = x1 + vx
    expectedY = y1 + vy
    meanExpectedX = x1 + meanVx
    meanExpectedY = y1 + meanVy
    (vTx, vTy) = VelocityTrendVector
    velocityTrendVectorNorm = m.sqrt(vTx**2 + vTy**2)
    if velocityTrendVectorNorm < trendTolerance: #We don't take into accounts trend.
        heightCost = min(min(abs(h2-h1),abs(h2)),abs(h1))
        velocityCost =  (1-meanPreference0)*m.sqrt((x2-expectedX)**2 + (y2-expectedY)**2) +  meanPreference0*m.sqrt((x2-meanExpectedX)**2 + (y2-meanExpectedY)**2)
        distanceCost = m.sqrt((x2-x1)**2 + (y2-y1)**2)
        return( distanceCost + heightCoeff*heightCost + velocityCoeff*velocityCost)
    else: #We realise there may be a trend. For example people mainly go across in one direction (North-South & South-North).
        unitVelocityTrendVector = (vTx/velocityTrendVectorNorm, vTy/velocityTrendVectorNorm)
        dpNorm = m.sqrt((x2-x1)**2 + (y2-y1)**2)
        if dpNorm > 0 :
            (unitdpx,unitdpy) = ((x2-x1)/dpNorm, (y2-y1)/dpNorm)
        else:
            (unitdpx,unitdpy) = (0,0)
        (unitVTx,unitVTy) = unitVelocityTrendVector
        trendAlignment = abs(unitdpx*unitVTx + unitdpy*unitVTy)
        heightCost1 = abs(h2-h1)
        heightCost2 = min(min(abs(h2-h1),abs(h2)),abs(h1))
        heightCost = (1-trendAlignment)*heightCost1 + trendAlignment*heightCost2
        distanceCost = m.sqrt((x2-x1)**2 + (y2-y1)**2)
        velocityCost =  (1-meanPreference0)*m.sqrt((x2-expectedX)**2 + (y2-expectedY)**2) +  meanPreference0*m.sqrt((x2-meanExpectedX)**2 + (y2-meanExpectedY)**2)
        totalCost = distanceCost + heightCoeff*heightCost + velocityCoeff*velocityCost
        return(totalCost)

#=======================================================#
                   #-Transport Optimal-#
#=======================================================#

method = 'revised simplex'

def computeTransport(X, Y, velocityImportanceCoeff, heightCoeff, VelocityTrendVector, trendTolerance): #takes in two lists of tuples, and some coefficient used for the distance

    nbPtsNuage1 = len(X)
    nbPtsNuage2 = len(Y)
    C = np.zeros([nbPtsNuage1, nbPtsNuage2])

    for i in range(nbPtsNuage1):
        for j in range(nbPtsNuage2):
            C[i,j] = distance(X[i],Y[j],velocityImportanceCoeff, heightCoeff, VelocityTrendVector, trendTolerance)

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
