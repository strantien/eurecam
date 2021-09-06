import numpy as np
import tuerTrajectoires as T
import math as m

#Une trajectoire est une liste de tuples [(image0,x0,y0,z0,h0), (image1,x1,y1,z1,h1),...]
# lActif = [t1,t2,...]
# lMorte = [t0, t-1, t-2]
# lFaussesTrajectoires = [t1f, ] (optionnelle)
# (lActif, frame i+1) -????-> (lActif)
#------------------------------------------------------------------ -----------#
#                          Parameters
#------------------------------------------------------------------------------#

xmin,xmax,ymin,ymax = (0,4,0,4) # pour tester le code
borderToleranceX = (xmax-xmin)/5
borderToleranceY = (ymax-ymin)/5
speedTolerance = 0.1 # fixed for now. Later, it may depend on the trajectory!
lifeTimeTolerance = 10 # fixed for now. Later, it may depend on [trajectoire]!
duplicateTolerance = 20 #This could depend on the trajectory later on.
DirectionChangeTolerance = 2
nPastVs = 2 #this is an arbitrary value

#------------------------------------------------------------------------------#
#                          Auxilary Functions
#------------------------------------------------------------------------------#

def distanceFunc(p1,p2):
    (frame1,x1,y1,z1,h1) = p1
    (frame2,x2,y2,z2,h2) = p2
    if frame2==frame1:
        return(m.sqrt((x2-x1)**2 + (y2-y1)**2 + min(min((h2-h1)**2,h2**2),h1**2)))
    else:
        return(m.sqrt((x2-x1)**2 + (y2-y1)**2 + min(min((h2-h1)**2,h2**2),h1**2)) +  1/abs(frame1-frame2))

def vitesses(trajectoire):
    nPoints = len(trajectoire)
    if nPoints < 2:
        return(np.array([]))
    else:
        trajectoireArray = np.array(trajectoire) #turn it into an array
        endPs = trajectoireArray[1:]
        startPs = trajectoireArray[:-1]
        vitessesX = (endPs[:,1] - startPs[:,1])/(endPs[:,0]-startPs[:,0])
        vitessesY = (endPs[:,2] - startPs[:,2])/(endPs[:,0]-startPs[:,0])
        return(np.stack((vitessesX,vitessesY), axis= 1))

def vitesseMoyenne(trajectoire, nPastVelocities):
    if nPastVelocities < 1 :
        return(np.array([]))
    else:
        n = min(len(trajectoire) - 1, nPastVelocities)
        vs = np.flip(vitesses(trajectoire),0)[:n]
        moyenne = np.sum(vs, axis=0)*(1/n)
        return(moyenne)

def tempsDeVie(trajectoire):
    framesArray= np.array(trajectoire)[:,0]
    return(framesArray[-1] - framesArray[0])

def tempsDeVieMoyenne(trajectoires, nbTrajectoires):
    n = len(trajectoires)
    consideredTrajectories = trajectories[(n - nbTrajectoires):]
    totalTime = np.sum(np.array(map(tempsDeVie, consideredTrajectories)))
    return(totalTime/nbTrajectoires)

#------------------------------------------------------------------------------#
#                     Functions applied on frame/image/nuage
#------------------------------------------------------------------------------#

#totally arbitrary condition for now
def isNewTrajectory(point, minBordX, maxBordX, minBordY, maxBordY):
    (frame,x,y,z,h) = point
    nearBorderX = (abs(x-minBordX) < borderToleranceX) | (abs(x-maxBordX) < borderToleranceX)
    nearBorderY = (abs(y-minBordY) < borderToleranceY) | (abs(y-maxBordY) < borderToleranceY)
    nearBorder = nearBorderX | nearBorderY
    return(nearBorder)

def isNoise(point, noiseList): # point is from an image (nuage de points)
    (frame,x,y,z,h) = point
    noiseBool = False
    for ps in noiseList:
        if (x,y) == ps[1:3]:
            noiseBool = True
    return(noiseBool)

def isDuplicate(point, image): #First attempt. Bound to fail!
    #I assume that point is in image!
    # Pour l'instant je suppose que "image" est une liste de points.
    restOfPoints = [p for p in image if p != point]
    distances = [distanceFunc(point,p) for p in image]
    duplicateBool = False
    for d in distances:
        if d < duplicateTolerance:
            duplicateBool = True
    return(duplicateBool)
#------------------------------------------------------------------------------#
#                           Functions applied on trajectories
#------------------------------------------------------------------------------#


#for now this will give a boolean. Later it may give a probability. Other changes may apply.
def leavesDomain(trajectoire, xMin,xMax,yMin,yMax):
    vs = vitesses(trajectoire)
    vMoyenne = vitesseMoyenne(trajectoire, nPastVs)
    if np.size(vs) < 1:
        return(False)#trajectory just started and so it probably won't leave! We could take <2 as well.
    else:
        (frame,x,y,z,h) = trajectoire[-1] #last/current position
        vxMoyenne = vMoyenne[0] #Mean velocity in x axis
        vyMoyenne = vMoyenne[1] #Mean velocity in y axis
        vxFinal   = vs[-1][0] #Last velocity in x axis
        vyFinal   = vs[-1][1] #Last velocity in y axis
        if vxFinal > speedTolerance: #moving right
            slowingDown =  vxFinal < (vxMoyenne - speedTolerance)
            if (abs(x-xMax) < abs(vxFinal)) & (not slowingDown):
                return(True)
        if vxFinal < - speedTolerance: #moving left
            slowingDown =  vxFinal > (vxMoyenne + speedTolerance)
            if (abs(x-xMin) < abs(vxFinal)) & (not slowingDown):
                return(True)
            else:
                return(False)
        if vyFinal > speedTolerance: #moving up
            slowingDown =  vyFinal < (vyMoyenne - speedTolerance)
            if (abs(y-yMax) < abs(vyFinal)) & (not slowingDown):
                return(True)
            else:
                return(False)
        if vyFinal < - speedTolerance: #moving down
            slowingDown =  vyFinal > (yMoyenne + speedTolerance)
            if (abs(y-yMin) < abs(vyFinal)) & (not slowingDown):
                return(True)
            else:
                return(False)
        return(False)

#Nous considerons deux choix. On peut tuer les trajectoires, ça veut dire qu'on
#les transfere dans une liste de vraies trajectoires qu'on traversé le domain.
# On peut aussi les suprimmer, ça veut dire que c'étaient des fausses trajectoires.
# Nous pourrions ajouter d'autres listes enventuellement pour modéliser le probleme.
def remainsTooLong(trajectoire, trajectoiresMortes):
    #Condition 1: Trajectory remains inside Domain for too long!
    nbMortes = len(trajectoiresMortes)
    if nbMortes < 1:
        tempsDeVieMoyenne = 0
    else:
        tempsDeVieMoyenne = np.sum(np.array(map(tempsDeVie, trajectoiresMortes)))*(1/nbMortes)
    return( tempsDeVie(trajectoire) > (tempsDeVieMoyenne + lifeTimeTolerance))

def changesDirectionTooMuch(trajectoire, trajectoiresMortes):
    #Condition 2: Trajectory changes direction too much (or too fast! --> to implement later)
    vs  = vitesses(trajectoire)
    vxs = np.sign(vs[:,0]) # direction of speeds in x axis
    vys = np.sign(vs[:,1]) # directions of speeds in y axis
    vxTransitions = vxs[:-1]*vxs[1:] # an array of {0,1,-1}, where -1 means a change in direction
    vyTransitions = vys[:-1]*vys[1:] # an array of {0,1,-1}, where -1 means a change in direction
    change_DirectionX = vxTransitions < 0
    change_DirectionY = vyTransitions < 0
    nbVxChanges = np.size(vxTransitions[change_DirectionX])
    nbVyChanges = np.size(vyTransitions[change_DirectionY])
    changesDirectionTooMuch = nbVxChanges > DirectionChangeTolerance | nbVyChanges > DirectionChangeTolerance
    return(changesDirectionTooMuch)

#More conditions could be added later. More complex ones too.
#--------------------------------------------------------#
#                     CODE TESTING                       #
#--------------------------------------------------------#

#t1 = [(1,1,2,1,9),(2,1,1,1,9),(3,0.5,0.5,1,9),(4,0.2,0.2,1,9), (5,0.1,0.2,1,9)]
#t2 = [(1,2,2,2,8),(2,1,1,2,8),(3,2.1,0.5,2,8),(4,2,2,2,8),(6,1,1,2,8),(7,2.1,0.5,2,8),(8,2,2,2,8),(9,1,1,2,8),(10,2.1,0.5,2,8),(11,2,2,2,8),(12,1,1,2,8)]#,(2.1,0.5,2),(2,2,2)] # turns around
#deadTrajectories = []

#print("trajectoire 2")
#print("t2")
#print("vitesses")
#print(vitesses(t2))
#print("vitesses moyenne (last 2 velocities considered)")
#print(vitesseMoyenne(t2, nPastVs))
#print("boundary =(xmin,xmax,ymin,ymax)")
#print((xmin,xmax,ymin,ymax))
#print("speed Tolerance")
#print(speedTolerance)
#print("leaves the domain?")
#print(leavesDomain(t2, xmin,xmax,ymin,ymax))
#print("#changes direction Too many times ?")
#print(changesDirectionTooMuch(t2, deadTrajectories))
#print("remains in domain for too long?")
#print(remainsTooLong(t2,deadTrajectories))
