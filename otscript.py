from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt

#=======================================================#
                   #- DONNEES -#
#=======================================================#
x1 = [4, 151, -23, 1]
x2 = [4, 153, 25, 2]
x3 = [4, 158, 94, 3]
x4 = [4, 158, 92, 4]
x5 = [4, 155, 86, 5]
y1 = [5, 154, -26, 1]
y2 = [5, 153, 25, 2]
y3 = [5, 158, 92, 3]
y4 = [5, 158, 94, 4]
y5 = [5, 155, 85, 5]

#=======================================================#
                   #-MATRICE DE COUT-#
#=======================================================#
X = np.array([x1,x2,x3,x4,x5])
Y = np.array([y1,y2,y3,y4,y5])


def distance(u,v):
    return(np.linalg.norm(u[1:]-v[1:]))

C = np.zeros([5,5])

for i in range(5):
    for j in range(5):
        C[i,j] = distance(X[i],Y[j])

c = np.ravel(C) #lignes de C
#l'ordre de notre vecteur (pour le problème linèaire) est :
#p11, p12, p13, p14, p15, p21, p22, ... etc
#où "pij"= connexion entre le point i du nuage1 et le point j du nuage2.

#=======================================================#
                    #- Contraintes -#
                 #-MATRICE DE COUPLAGE-#
#=======================================================#

nbPtsNuage1 = X.shape[0]
nbPtsNuage2 = Y.shape[0]
nbVariables = nbPtsNuage1*nbPtsNuage2 # En général c'est de la meme taile que c.
nbContraintesSortantes = nbPtsNuage1
nbContraintesEntrantes = nbPtsNuage2
nbContraintes = nbContraintesEntrantes + nbContraintesSortantes

b_eq = np.ones(nbContraintes)
#print(b_eq)

A_eq = np.zeros([nbContraintes,nbVariables])
#print(A_eq.shape)
#print(c.shape)
#les premières lignes correspondent aux contraintes de sortie. c'est à dire pour
#chaque point du Nuage1, il faut le connecter à un seul point du nuage 2.
for i in range(nbContraintesSortantes):
    for j in range(nbContraintesEntrantes):
        A_eq[i,i*nbContraintesEntrantes +j] = 1
#les dernieres lignes correspondent aux contraintes d'entrée. Pour chaque point
#dans le nuage d'arrivée, il peut recevoir 1 seul nuage de sortie.
for i in range(nbContraintesEntrantes):
    for j in range(nbContraintesSortantes):
        A_eq[i + nbContraintesSortantes, i + j*nbContraintesEntrantes] = 1

#print(A_eq)

#=======================================================#
                 #-RESOLUTION DU PROBLEME DE PL-#
#=======================================================#

P = linprog(c, A_eq = A_eq, b_eq = b_eq, bounds=(0,1), method='simplex')
#print(P.x)

#on 'unravel' le résultat P.x, à la main
x = np.array([P.x[k*nbPtsNuage2:(k+1)*nbPtsNuage2] for k in range(nbPtsNuage1)])
#x est la matrice de connexion, cad la matrice des pij
print(x)


#=======================================================#
                 #-TRACE DES SOLUTIONS-#
#=======================================================#

plt.clf()
#tracé des deux nuages de points initiaux
plt.scatter(X[:,1], X[:,2], marker = '+')
plt.scatter(Y[:,1], Y[:,2], marker = 'x')
#tracé des connexions
for i in range(nbPtsNuage1):
    for j in range(nbPtsNuage2):
        if x[i,j] == 1.:
            plt.plot([X[i,1], Y[j,1]], [X[i,2], Y[j,2]], color = 'green', linewidth = .7)
plt.grid()
plt.show()
