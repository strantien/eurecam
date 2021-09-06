import numpy as np
import math as m
import optimalTransport as ot

#------------------------------------------------------------------------------#
#                          Auxilary Functions
#------------------------------------------------------------------------------#

def vitesses(trajectoire): #Prend une liste de tuples. Returns a list of tuples [(vx,vy)].
    nPoints = len(trajectoire)
    if nPoints < 2:
        return(np.array([]))
    else:
        trajectoireArray = np.array(trajectoire) #turn it into an array
        endPs = trajectoireArray[1:]
        startPs = trajectoireArray[:-1]
        vitessesX = (endPs[:,1] - startPs[:,1])/(endPs[:,0]-startPs[:,0])
        vitessesY = (endPs[:,2] - startPs[:,2])/(endPs[:,0]-startPs[:,0])
        vs0 = np.stack((vitessesX,vitessesY), axis= 1)
        vs1 = list(vs0)
        vs = [tuple(v) for v in vs1]
        return(vs)

def vitesseMoyenne(trajectoire, nPastVelocities): #prends une liste de tuples. Returns a tuple (vx,vy)
    if nPastVelocities < 1 :
        return((0,0)) # Choix arbitraire (mais raisonnable!)
    else:
        n = min(len(trajectoire) - 1, nPastVelocities)
        vs0 = np.array(vitesses(trajectoire))
        vs = np.flip(vs0 ,0)[:n]
        moyenne0 = np.sum(vs0, axis=0)*(1/n)
        moyenne = tuple(moyenne0)
        return(moyenne)

def tempsDeVie(trajectoire): #Prend une liste de tuples. returns a number
    framesArray= np.array(trajectoire)[:,0]
    return(framesArray[-1] - framesArray[0])

def tempsDeVieMoyenne(trajectoires, nbTrajectoires):#Prend une liste de trajectoires. Returns a real value (positif)
    n = len(trajectoires)
    consideredTrajectories = trajectories[(n - nbTrajectoires):] # c'est une liste de trajectoires cet objet
    listOfTimes = map(tempsDeVie, consideredTrajectories)
    arrayOfTimes = np.array(listOfTimes)
    totalTime = np.sum(arrayOfTimes)
    return(totalTime/nbTrajectoires)

def peopleEqualityFunc(p1,p2, equalityTolerance): #prends deux tuples de taille 5, returns Boolean.
    (frame1,x1,y1,z1,h1) = p1
    (frame2,x2,y2,z2,h2) = p2
    areEqual = (m.sqrt((x2-x1)**2 + (y2-y1)**2 + min(min((h2-h1)**2,h2**2),h1**2))) < equalityTolerance
    return(areEqual)

def equalFuncNoise(p1,p2, equalityTolerance): #prends deux tuples de taille 5, returns Boolean.
    (frame1,x1,y1,z1,h1) = p1
    (frame2,x2,y2,z2,h2) = p2
    areEqual = (m.sqrt((x2-x1)**2 + (y2-y1)**2 + (h2-h1)**2)) < equalityTolerance
    return(areEqual)

def isIn(point, pointList, equalityFunc, equalityTolerance): #Prends un tuple (5 valeurs) et une liste de tuples (5 valeurs). Returns a Boolean.
    equalityList = [equalityFunc(point, p, equalityTolerance) for p in pointList]
    for isEqual in equalityList:
        if isEqual:
            return(True)
    return(False)


#------------------------------------------------------------------------------#
#                     Functions applied on frame/image/nuage
#------------------------------------------------------------------------------#

#totally arbitrary condition for now
def isNewTrajectory(point, minBordX, maxBordX, minBordY, maxBordY, borderToleranceX, borderToleranceY): #Prends un tuple ou un np.array. Returns a Boolean.
    (frame,x,y,z,h) = point
    nearBorderX = (abs(x-minBordX) < borderToleranceX) | (abs(x-maxBordX) < borderToleranceX)
    nearBorderY = (abs(y-minBordY) < borderToleranceY) | (abs(y-maxBordY) < borderToleranceY)
    nearBorder = nearBorderX | nearBorderY
    return(nearBorder)

def isNoise(point, noiseList, noiseTol):#Prends un np.array ou un tuple, et une liste de tuples. returns a Boolean.
    #Something smarter can be done easily.
    return(isIn(point,noiseList, equalFuncNoise, noiseTol))

def nouveauTrajet(point): # Prends un array 1D ou tuple (avec 5 elements exactement!). Returns a trajectory.
    p = tuple(point)
    (frame,x,y,z,h) = p
    velocity = (0,0) # we take every new trajectory to have a null velocity component.
    return([(frame,x,y,z,h,velocity)])
#------------------------------------------------------------------------------#
#                           Functions applied on trajectories
#------------------------------------------------------------------------------#
def leavesDomain(trajectoire, xMin,xMax,yMin,yMax, speedTolerance, nPastVs):
    #prends une liste de tuples, des bounds, une tolerance et le nombre de Vitesses passées á regarder. Returns a Boolean
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
        if vyFinal > speedTolerance: #moving up
            slowingDown =  vyFinal < (vyMoyenne - speedTolerance)
            if (abs(y-yMax) < abs(vyFinal)) & (not slowingDown):
                return(True)
        if vyFinal < - speedTolerance: #moving down
            slowingDown =  vyFinal > (yMoyenne + speedTolerance)
            if (abs(y-yMin) < abs(vyFinal)) & (not slowingDown):
                return(True)
        return(False)

def remainsTooLong(trajectoire, trajectoiresMortes, lifeTimeTolerance, baseLifeTime): #Prend une liste de tuples, liste de trajectoires, et deux reel. Returns a Boolean.
    #Condition 1: Trajectory remains inside Domain for too long!
    nbMortes = len(trajectoiresMortes)
    if nbMortes < 1:
        tempsDeVieMoyenne = 0
    else:
        tempsDeTrajectoiresMortes0 = map(tempsDeVie, trajectoiresMortes)
        tempsDeTrajectoiresMortesArray =np.array(tempsDeTrajectoiresMortes0)
        tempsDeVieMoyenne = np.sum(tempsDeTrajectoiresMortesArray)*(1/nbMortes)
    if tempsDeVieMoyenne ==0:
        return(tempsDeVie(trajectoire) > baseLifeTime)
    else:
        return(tempsDeVie(trajectoire) > min(baseLifeTime,tempsDeVieMoyenne + lifeTimeTolerance))

def changesDirectionTooMuch(trajectoire, trajectoiresMortes, DirectionChangeTolerance):
    #Condition 2: Trajectory changes direction too much (or too fast! --> to implement later)
    vs0 = vitesses(trajectoire)
    vs  = np.array(vs)
    vxs = np.sign(vs[:,0]) # direction of speeds in x axis
    vys = np.sign(vs[:,1]) # directions of speeds in y axis
    vxTransitions = vxs[:-1]*vxs[1:] # an array of {0,1,-1}, where -1 means a change in direction
    vyTransitions = vys[:-1]*vys[1:] # an array of {0,1,-1}, where -1 means a change in direction
    change_DirectionX = vxTransitions < 0
    change_DirectionY = vyTransitions < 0
    nbVxChanges = np.size(vxTransitions[change_DirectionX])
    nbVyChanges = np.size(vyTransitions[change_DirectionY])
    changesDirectionTooMuch = (nbVxChanges + nbVyChanges) > DirectionChangeTolerance
    return(changesDirectionTooMuch)
