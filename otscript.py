from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#=======================================================#
                   #- DONNEES -#
#=======================================================#

#changer le répertoire de travail en fonction de votre PC
os.chdir("/home/trantien/Bureau/icj/doctorat/challenge_amies")
#os.chdir("pedro")
#os.chdir("kiki")

##Jeu de données A

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

##Jeu de données B

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

##Jeu de données C (issu de la détection 001 ; pas de pieds ; nbPtsNuage1 = nbPtsNuage2)

# X = np.array([[116, -89, -94, 123], [116, 114, -119, 82], [116, 122, -116, 90], [116, 55, -67, 136], [116, 57, -67, 136], [116, 107, -34, 137], [116, -39, -23, 119], [116, 158, 92, 104], [116, 155, 85, 117], [116, 158, 94, 103]])
# Y = np.array([[117, -74, -84, 136], [117, 115, -108, 100], [117, -76, -74, 137], [117, 58, -48, 141], [117, -84, -49, 90], [117, 107, -17, 135], [117, -38, 2, 120], [117, 158, 92, 104], [117, 155, 86, 118], [117, 158, 94, 103]])

#RESULTAT : plutôt cohérent, non ?

##Jeu de données D

# x1 = [4, 151, -23, 5]
# x2 = [4, 153, 25, 10]
# x3 = [4, 158, 94, 15]
# x4 = [4, 158, 92, 20]
# x5 = [4, 155, 86, 25]
# x6 = [4, 155, 70, 30]
# y1 = [5, 154, -26, 5]
# y2 = [5, 153, 25, 10]
# y3 = [5, 158, 92, 15]
# y4 = [5, 158, 94, 20]
# y5 = [5, 155, 85, 25]
# X = np.array([x1,x2,x3,x4,x5,x6])
# Y = np.array([y1,y2,y3,y4,y5])

#RESULTAT : ok, x6 est ignoré (pas de connexion partant de x6).

##Jeu de données E

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
# y6 = [5, 153, 70, 30]
# X = np.array([x1,x2,x3,x4,x5])
# Y = np.array([y1,y2,y3,y4,y5,y6])

#RESULTAT : ok, y6 est ignoré (pas de connexion arrivant vers x6).

##Jeu de données E de Kyriaki

# df = pd.read_csv("testKiki.txt",sep=" ",names=["Image", "x", "y", "z", "height"])
# df.drop(0,0,inplace=True)
# #on transforme le tableau obtenu en un tableau d'entiers
# data = np.vectorize(int)(np.asarray(df))
# #je supprime la colonne des z ; sinon, il faut modifier la définition de distance
# data = np.delete(data, obj = 3, axis = 1)
# #print(data)
# X = data[:5,:]
# Y = data[5:,:]

##Jeu de données F : détection 001, transport entre l'image i et l'image i+1

#dans ce jeu données, la façon d'extraire le fichier et de construire le tableau
#d'indices est un peu pourrie. Je n'avais pas vu que c'était déjà fait dans show_
#detection.

df = pd.read_csv("file://"+os.getcwd()+"/Challenge_AMIES_EURECAM/data_detection/001/detection.txt",sep=" ",names=["Image", "x", "y", "z", "height"],skiprows=2)
df.drop(0,0,inplace=True)

#on transforme le tableau obtenu en un tableau d'entiers
data = np.vectorize(int)(np.asarray(df))

#on crée un tableau d'indices à 2 lignes, dont la ligne i donne les indices min et max des points de l'image numéro i + imageMin
#par exemple, pour la détection 001, indices[0, :] = [0, 4] car dans data, les lignes de 0 à 4 correspondent à l'image numéro 4
imageMin, imageMax = np.min(data[:,0]), np.max(data[:,0])
indices = np.zeros([imageMax - imageMin + 1, 2], dtype = int)
imageCourante = imageMin
jmin = 0
j = jmin
while imageCourante < imageMax:
    if data[j, 0] == imageCourante:
        j += 1
    else: #dans ce cas, data[j, 0] > imageCourante
        indices[imageCourante - imageMin, :] = [jmin, j-1]
        imageCourante += 1
        jmin = j
        j += 1
#print(indices)

#on veut comparer les images i et i+1, i étant à choisir :
i = 5
X = data[indices[i-imageMin, 0]:indices[i-imageMin, 1]]
Y = data[indices[i-imageMin+1, 0]:indices[i-imageMin+1, 1]]

##Jeux de données du sujet

# #copier-coller de show_detection.py
#
# dataset = "001"
# # dataset = "002"
# # dataset = "003"
# # dataset = "004"
# # dataset = "005"
# # dataset = "006"
# # dataset = "007"
# # dataset = "008"
# # dataset = "009"
# # dataset = "010"
#
# detfile = "data_detection/" + dataset + "/detection.txt"
# imgdir  = "data_detection/" + dataset + "/images/"
#
# file = open(detfile, "r")
# l0 = file.readline()
# l1 = file.readline()
# f,cx,cy = np.array(l1.split()).astype(int)
# file.close()
#
# detections = pd.read_csv(detfile,delimiter=" ",skiprows=2)
# # print(detections.head(20))
#
# images = np.unique(detections["#image"].values)
# # print("images : ",images)

#=======================================================#
                   #-MATRICE DE COUT-#
#=======================================================#

#u[-1] correspond à la hauteur, que l'on ait supprimé la colonne z ou non
def distance(u,v):
    return(np.linalg.norm(u[1:2]-v[1:2]) + np.linalg.norm(u[-1]-v[-1]))

def computeTransport(X, Y):

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
                #-RESOLUTION DU PROBLEME DE PL-#
    #=======================================================#

    nbVariables = nbPtsNuage1*nbPtsNuage2 # En général c'est de la meme taile que c.
    nbContraintesSortantes = nbPtsNuage1
    nbContraintesEntrantes = nbPtsNuage2
    nbContraintes = nbContraintesEntrantes + nbContraintesSortantes

    ##Cas 1 : autant de contraintes entrantes que sortantes

    if nbContraintesSortantes == nbContraintesEntrantes:
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

        P = linprog(c, A_eq = A_eq, b_eq = b_eq, bounds=(0,1), method='simplex')
        #print(P.x)

    ##Cas 2 : plus de contraintes entrantes que sortantes

    elif nbContraintesSortantes > nbContraintesEntrantes:
        b_eq = np.zeros(nbContraintes)
        b_eq[nbContraintesSortantes:] = np.ones(nbContraintesEntrantes)
        #print('b_eq =', b_eq)

        b_ub = np.zeros(nbContraintes)
        b_ub[:nbContraintesSortantes] = np.ones(nbContraintesSortantes)
        #print('b_ub =', b_ub)

        A_eq = np.zeros([nbContraintes,nbVariables])
        #les premieres lignes sont nulles ; les contraintes d'égalité sont :
        #dans le nuage d'arrivée, un point doit recevoir pile 1 point de sortie.
        for i in range(nbContraintesEntrantes):
            for j in range(nbContraintesSortantes):
                A_eq[i + nbContraintesSortantes, i + j*nbContraintesEntrantes] = 1
        #print('A_eq =', A_eq)

        A_ub = np.zeros([nbContraintes,nbVariables])
        #les contraintes d'inégalité sont les contrainte de sortie, cad pour
        #chaque point du Nuage1, il faut le connecter à au plus 1 point du nuage 2.
        for i in range(nbContraintesSortantes):
            for j in range(nbContraintesEntrantes):
                A_ub[i,i*nbContraintesEntrantes +j] = 1
        #print('A_ub =', A_ub)

        P = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds=(0,1), method='simplex')
        #print(P.x)

    ##Cas 3 : plus de contraintes sortantes qu'entrantes

    else:
        b_eq = np.zeros(nbContraintes)
        b_eq[:nbContraintesSortantes] = np.ones(nbContraintesSortantes)
        #print('b_eq =', b_eq)

        b_ub = np.zeros(nbContraintes)
        b_ub[nbContraintesSortantes:] = np.ones(nbContraintesEntrantes)
        #print('b_ub =', b_ub)

        A_eq = np.zeros([nbContraintes,nbVariables])
        #les dernières lignes sont nulles ; les contraintes d'égalité sont :
        #chaque point du nuage de départ doit être connecté à exactement 1 point
        #du nuage d'arrivée.
        for i in range(nbContraintesSortantes):
            for j in range(nbContraintesEntrantes):
                A_eq[i,i*nbContraintesEntrantes +j] = 1
        #print('A_eq =', A_eq)

        A_ub = np.zeros([nbContraintes,nbVariables])
        #les contraintes d'inégalité sont les contrainte d'entrée, cad que
        #chaque point du Nuage2 ne peut recevoir qu'au plus 1 point du nuage 2.
        for i in range(nbContraintesEntrantes):
            for j in range(nbContraintesSortantes):
                A_ub[i + nbContraintesSortantes, i + j*nbContraintesEntrantes] = 1
        #print('A_ub =', A_ub)

        P = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds=(0,1), method='simplex')

    ##Résultat : la matrice de connexion x

    #on 'unravel' le résultat P.x, à la main
    x = np.array([P.x[k*nbPtsNuage2:(k+1)*nbPtsNuage2] for k in range(nbPtsNuage1)])
    #x est la matrice de connexion, cad la matrice des pij
    #print(x)
    return(x)

#=======================================================#
                 #-TRACE DU TRANSPORT ENTRE X ET Y-#
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
    #X[i,-1] correspond toujours à la hauteur de Xi, que l'on ait supprimé la
    #colonne z ou non.
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

#plotTransport(X, Y)

#=======================================================#
                 #-FILM COMPLET-#
#=======================================================#

#ATTENTION : cette partie n'est pas encore terminée. Tout le reste marche.

# plt.ion()
# fig = plt.figure()
# for i in images:
#     print("==> image : ",i)
#     mask = (detections["#image"]==i)
#     x = detections[mask]["x"].values
#     y = detections[mask]["y"].values
#     z = detections[mask]["z"].values
#     h = detections[mask]["h"].values
#
#     plt.clf()
#
#     try:
#         im = imageio.imread(imgdir+"/image-"+str(i).zfill(3)+".png")
#     except:
#         im = imageio.imread(imgdir+"/image-"+str(i).zfill(3)+".jpg")
#
#     height, width  = im.shape
#     # print("width = ",width," height = ",height) ## width = 448,  height = 480
#
#     ix = 0.5*(x/z*f+cx)
#     iy = 0.5*(y/z*f+cy)
#
#     plt.imshow(im) ## image camera + image capteur
#     plt.scatter(ix, iy, marker="+",color="red");
#     input('type enter to continue')
#     plt.pause(0.05)
#     plt.draw()
#
# plt.ioff()
# plt.show()
