import numpy as np



#--------------------   Parameters   --------------------#
#boundary = (xMin, xMax, yMin, yMax)
xmin,xmax,ymin,ymax = (0,4,0,4)
nPastVs = 2 #this is an arbitrary value
speedTolerance = 0.1 # fixed for now. Later, it may depend on the trajectory!
lifeTimeTolerance = 10 # fixed for now. Later, it may depend on [trajectoire]!
DirectionChangeTolerance = 2
#--------------------   End of Parameters   --------------------#

#--------------------------------------------------------#
#                FONCTIONS AUXILIAIRES                   #
#--------------------------------------------------------#

#Une trajectoire est une liste de tuples [(x0,y0,h0),(x1,y1,h1),...]

def vitesses(trajectoire):
    nPoints = len(trajectoire)
    if nPoints < 2:
        return(np.array([]))
    else:
        trajectoireArray = np.array(trajectoire) #turn it into an array
        endPs = trajectoireArray[1:]
        startPs = trajectoireArray[:-1]
        vitesses = endPs[:,:-1] - startPs[:,:-1] #time interval of 1. This may change according to the frame number.
        return(vitesses)

def vitesseMoyenne(trajectoire, nPastVelocities):
    if nPastVelocities < 1 :
        return(np.array([]))
    else:
        n = min(len(trajectoire) - 1, nPastVelocities)
        vs = np.flip(vitesses(trajectoire),0)[:n]
        moyenne = np.sum(vs, axis=0)*(1/n)
        return(moyenne)

def tempsDeVie(trajectoire):
    return(len(trajectoire))

def tempsDeVieMoyenne(trajectoires, nbTrajectoires):
    n = len(trajectoires)
    consideredTrajectories = trajectories[(n - nbTrajectoires):]

#--------------------------------------------------------#
#                         TESTS                          #
#--------------------------------------------------------#
#Avec ces tests on peut tuer ou pas une trajectoire.

#for now this will give a boolean. Later it may give a probability. Other changes may apply.
def leavesDomain(trajectoire, xMin,xMax,yMin,yMax):
    vs = vitesses(trajectoire)
    vMoyenne = vitesseMoyenne(trajectoire, nPastVs)
    if np.size(vs) < 1:
        return(False)#trajectory just started and so it probably won't leave! We could take <2 as well.
    else:
        (x,y,h) = trajectoire[-1] #last/current position
        vxMoyenne = vMoyenne[0] #Mean velocity in x axis
        vyMoyenne = vMoyenne[1] #Mean velocity in y axis
        vxFinal   = vs[-1][0] #Last velocity in x axis
        vyFinal   = vs[-1][1] #Last velocity in y axis
        if vxFinal > speedTolerance: #moving right
            slowingDown =  vxFinal < (vxMoyenne - speedTolerance)
            if abs(x-xMax) < abs(vxFinal) & (not slowingDown):
                return(True)
            else:
                return(False)
        if vxFinal < - speedTolerance: #moving left
            slowingDown =  vxFinal > (vxMoyenne + speedTolerance)
            if abs(x-xMin) < abs(vxFinal) & (not slowingDown):
                return(True)
            else:
                return(False)
        if vyFinal > speedTolerance: #moving up
            slowingDown =  vyFinal < (vyMoyenne - speedTolerance)
            if abs(y-yMax) < abs(vyFinal) & (not slowingDown):
                return(True)
            else:
                return(False)
        if vyFinal < - speedTolerance: #moving down
            slowingDown =  vyFinal > (yMoyenne + speedTolerance)
            if abs(y-yMin) < abs(vyFinal) & (not slowingDown):
                return(True)
            else:
                return(False)
        return(False)


#Nous considerons deux choix. On peut tuer les trajectoires, ça veut dire qu'on
#les transfere dans une liste de vraies trajectoires qu'on traversé le domain.
# On peut aussi les suprimmer, ça veut dire que c'étaient des fausses trajectoires.
# Nous pourrions ajouter d'autres listes enventuellement pour modéliser le probleme.
def isNoiseTrajectory(trajectoire, trajectoiresMortes):
    #Condition 1: Trajectory remains inside Domain for too long!
    nbMortes = len(trajectoiresMortes)
    if nbMortes < 1:
        tempsDeVieMoyenne = 0
    else:
        tempsDeVieMoyenne = np.sum(np.array(map(tempsDeVie, trajectoiresMortes)))*(1/nbMortes)
    remainsTooLong =  tempsDeVie(trajectoire) > (tempsDeVieMoyenne + lifeTimeTolerance)
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
    #More conditions could be added later. More complex ones too.
    return(remainsTooLong | changesDirectionTooMuch)


#--------------------------------------------------------#
#                     CODE TESTING                       #
#--------------------------------------------------------#
t1 = [(1,2,1),(1,1,1),(0.5,0.5,1),(0.2,0.2,1), (0.1,0.2,1)]
t2 = [(2,2,2),(1,1,2),(2.1,0.5,2),(2,2,2),(1,1,2),(2.1,0.5,2),(2,2,2),(1,1,2),(2.1,0.5,2),(2,2,2),(1,1,2),(2.1,0.5,2),(2,2,2)] # turns around
deadTrajectories = []

print("trajectoire 2")
print(t2)
print("vitesses")
print(vitesses(t2))
print("vitesses moyenne (last 2 velocities considered)")
print(vitesseMoyenne(t2, nPastVs))
print("boundary =(xmin,xmax,ymin,ymax)")
print((xmin,xmax,ymin,ymax))
print("speed Tolerance")
print(speedTolerance)
print("leaves the domain?")
print(leavesDomain(t2, xmin,xmax,ymin,ymax))
print("changes direction Too many times ?")
print(isNoiseTrajectory(t2, deadTrajectories))
