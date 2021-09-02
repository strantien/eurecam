from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt

#=======================================================#
                   #- DONNEES -#
#=======================================================#

'''Pour l'instant, on n'utilise que des données "jouet" fabriquées par nous-mêmes'''

#Jeu de données A

# x1 = [4, 151, -23, 1]
# x2 = [4, 153, 25, 2]
# x3 = [4, 158, 94, 3]
# x4 = [4, 158, 92, 4]
# x5 = [4, 155, 86, 5]
# y1 = [5, 154, -26, 1]
# y2 = [5, 153, 25, 2]
# y3 = [5, 158, 92, 3]
# y4 = [5, 158, 94, 4]
# y5 = [5, 155, 85, 5]
# X = np.array([x1,x2,x3,x4,x5])
# Y = np.array([y1,y2,y3,y4,y5])

#RESULTAT : les tailles sont toutes distinctes, et pourtant, linprog envoie x3 sur y4 et x4 sur y3. Peut-être les écarts entre les tailles ne sont-ils pas assez grands ?

#Jeu de données B

# x1 = [4, 151, -23, 5]
# x2 = [4, 153, 25, 10]
# x3 = [4, 158, 94, 15]
# x4 = [4, 158, 92, 20]
# x5 = [4, 155, 86, 25]
# y1 = [5, 154, -26, 5]
# y2 = [5, 153, 25, 10]
# y3 = [5, 158, 92, 15]
# y4 = [5, 158, 94, 20]
# y5 = [5, 155, 85, 25]
# X = np.array([x1,x2,x3,x4,x5])
# Y = np.array([y1,y2,y3,y4,y5])

#RESULTAT : on n'a changé, par rapport au jeu de données A, que les tailles. Les écarts sont plus significatifs, de l'ordre de 10 cm. Alors, les tailles ne sont plus mélangées dans le résultat. J'interprète ça comme un point positif : on n'a pas à distinguer deux personnes si leurs tailles diffèrent de quelques centimètres, mais on arrive à la distinguer si leurs tailles diffèrent de 10cm.

#Jeu de données C (issu de la détection 001 ; pas de pieds ; nbPtsNuage1 = nbPtsNuage2)

X = np.array([[116, -89, -94, 123], [116, 114, -119, 82], [116, 122, -116, 90], [116, 55, -67, 136], [116, 57, -67, 136], [116, 107, -34, 137], [116, -39, -23, 119], [116, 158, 92, 104], [116, 155, 85, 117], [116, 158, 94, 103]])
Y = np.array([[117, -74, -84, 136], [117, 115, -108, 100], [117, -76, -74, 137], [117, 58, -48, 141], [117, -84, -49, 90], [117, 107, -17, 135], [117, -38, 2, 120], [117, 158, 92, 104], [117, 155, 86, 118], [117, 158, 94, 103]])

#=======================================================#
                   #-MATRICE DE COUT-#
#=======================================================#

def distance(u,v):
    return(np.linalg.norm(u[1:]-v[1:]))

nbPtsNuage1 = X.shape[0]
nbPtsNuage2 = Y.shape[0]
C = np.zeros([nbPtsNuage1, nbPtsNuage2])

for i in range(nbPtsNuage1):
    for j in range(nbPtsNuage2):
        C[i,j] = distance(X[i],Y[j])

c = np.ravel(C) #lignes de C
#l'ordre de notre vecteur (pour le problème linèaire) est :
#p11, p12, p13, p14, p15, p21, p22, ... etc
#où "pij"= connexion entre le point i du nuage1 et le point j du nuage2.

#=======================================================#
                    #- Contraintes -#
                 #-MATRICE DE COUPLAGE-#
#=======================================================#

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
plt.scatter(X[:,1], X[:,2], marker = '+', color = 'blue')
plt.scatter(Y[:,1], Y[:,2], marker = 'x', color = 'red')
#annotation des hauteurs de chaque point
for i in range(nbPtsNuage1):
    plt.annotate("h =" + str(X[i,3]), (X[i,1], X[i,2]), color = 'blue', fontsize = 7)
for j in range(nbPtsNuage2):
    plt.annotate("h =" + str(Y[j,3]), (Y[j,1], Y[j,2]), color = 'red', fontsize = 7)
#tracé des connexions
for i in range(nbPtsNuage1):
    for j in range(nbPtsNuage2):
        if x[i,j] == 1.:
            plt.plot([X[i,1], Y[j,1]], [X[i,2], Y[j,2]], color = 'green', linewidth = .7)
plt.grid()
plt.show()
