import numpy as np
import math as m

#====================MAIN OBJECTS====================+#
# point wil be a 5-tuple: (frame, x, y, z, h)
# trajectory state will be a n-tuple (n => 5): (frame,x,y,z,h,velocity,meanVelocity)
# trajectory will be a list of trajectory states
#====================================================+#

#------------------------------------------------------------------------------#
#                          Auxilary Functions
#------------------------------------------------------------------------------#
def dirChange(velocity1, velocity2):#Takes in 2 tuples and returns a Bool
    (vx1,vy1) = velocity1[:2]
    (vx2,vy2) = velocity2[:2]
    change = np.sign(vx1*vx2 + vy1*vy2)
    abruptChange = change <= 0
    return(abruptChange)

def meanOfTuples(tuples):# takes a non-empty list of n-tuples, returns an n-tuple
   n = len(tuples)
   tab = np.array(tuples)
   mean = np.sum(tab,axis=0)*(1/n)
   mean = tuple(mean)
   return(mean)

def stateToPoint(state):#Takes in tuple state and returns a point
    point = list(state)[:5]
    point = tuple(point)
    return(point)

def newState(point): #Prends un tuple (point) et returns a tuple (trajectory state)
   p = list(point)
   lastVelocity = (0,0)
   meanVelocity = (0,0)
   p = p + [lastVelocity, meanVelocity]
   p = tuple(p)
   return(p)

def trajectoryToListOfPoints(trajectory):
    pointList = [stateToPoint(st) for st in trajectory]
    return(pointList)

def tempsDeVie(trajectoire): #Prend une liste de tuples. returns a number
    framesArray = np.array(trajectoryToListOfPoints(trajectoire))[:,0]
    return(framesArray[-1] - framesArray[0])

def tempsDeVieMoyenne(trajectoires, nbTrajectoires):#Prend une liste de trajectoires. Returns a real value (positif)
    n = min(len(trajectoires) , nbTrajectoires)
    k = len(trajectoires) - n
    consideredTrajectories = trajectoires[k:] # c'est une liste de trajectoires cet objet
    listOfTimes = [ tempsDeVie(traj) for traj in consideredTrajectories]
    totalTime = sum(listOfTimes)
    return(totalTime/n)

def peopleEqualityFunc(trendTolerance,  velocityVector, p1, p2, equalityTolerance): #prends deux tuples. returns Boolean.
    (vx, vy) = velocityVector
    vNorm = m.sqrt(vx**2 + vy**2)
    if vNorm > trendTolerance:
            unitVelocityVector = (vx/vNorm, vy/vNorm)
            (x1,y1,z1,h1) = tuple(p1[1:5])
            (x2,y2,z2,h2) = tuple(p2[1:5])
            dpNorm = m.sqrt((x2-x1)**2 + (y2-y1)**2)
            (unitdpx,unitdpy) = ((x2-x1)/dpNorm, (y2-y1)/dpNorm)
            (unitVx,unitVy) = unitVelocityVector
            alignment = abs(unitdpx*unitVx + unitdpy*unitVy)
            heightCost1 = (h2-h1)**2
            heightCost2 = min(min((h2-h1)**2,h2**2),h1**2)
            heightCost = (1-alignment)*heightCost1 + alignment*heightCost2
            areEqual = (m.sqrt((x2-x1)**2 + (y2-y1)**2 + heightCost)) < equalityTolerance
            return(areEqual)
    else:
        (x1,y1,z1,h1) = tuple(p1[1:5])
        (x2,y2,z2,h2) = tuple(p2[1:5])
        heightCost = min(min((h2-h1)**2,h2**2),h1**2)
        areEqual = (m.sqrt((x2-x1)**2 + (y2-y1)**2 + heightCost)) < equalityTolerance

def equalFuncNoise(p1,p2, equalityTolerance): #prends deux tuples, returns Boolean.
    (x1,y1,z1,h1) = tuple(p1[1:5])
    (x2,y2,z2,h2) = tuple(p2[1:5])
    areEqual = (m.sqrt((x2-x1)**2 + (y2-y1)**2 + (h2-h1)**2)) < equalityTolerance
    return(areEqual)

def tupleIsIn(point, pointList, equalityFunc, equalityTolerance): #Prends un tuple et une liste de tuples. Returns a Boolean.
    equalityList = [equalityFunc(point, p, equalityTolerance) for p in pointList]
    for matches in equalityList:
        if matches:
            return(True)
    return(False)

def nouveauTrajet(point): # Prends un array 1D ou tuple. Returns a trajectory with one point.
    newSt = newState(point)
    newTrajet =[newSt]
    return(newTrajet)

def addPointToTraj(point,trajectory): #takes in a 5-tuple and a trajectory. Returns a trajectory
    x = trajectory[-1][1]
    y = trajectory[-1][2]
    (meanVx, meanVy) = trajectory[-1][6]
    newX = point[1]
    newY = point[2]
    n = len(trajectory)
    dt = point[0] - trajectory[-1][0]
    newVelocity = ((newX - x)/dt, (newY - y)/dt)
    newMeanVelocity = ((n*meanVx + newVelocity[0])/(n+1), (n*meanVy + newVelocity[1])/(n+1))
    newState = list(point)
    newState = newState + [newVelocity, newMeanVelocity] #here I assume that "point" is a 5-tuple
    newState = tuple(newState)
    newTrajectory = trajectory + [newState]
    return(newTrajectory)

#------------------------------------------------------------------------------#
#                     Functions applied on frame/image/nuage/points
#------------------------------------------------------------------------------#
#totally arbitrary condition for now

def isNoise(point, noiseList, noiseTol):#Prends un np.array ou un tuple, et une liste de tuples. returns a Boolean.
    #Something smarter can be done easily.
    return(tupleIsIn(point,noiseList, equalFuncNoise, noiseTol))

def advanceOneFrameState(state, stationnary): # Prends un tuple . Returns a tuple
    if stationnary:
        frame = state[0]
        newPoint = [(frame+1)]
        n = len(state)
        for i in range(n-1):
            newPoint.append(state[i+1])
        newPoint = tuple(newPoint)
        return(newPoint)
    else:
        frame = state[0]
        newState = [(frame+1)]
        x = state[1]
        y = state[2]
        (vx,vy) = state[5]
        (meanVx,meanVy) = state[6]
        z = state[3]
        h = state[4]
        newX = x + vx
        newY = y + vy
        newState = newState + [newX,newY,z,h,(vx,vy),(meanVx,meanVy)]
        newState = tuple(newState)
        return(newState)
#------------------------------------------------------------------------------#
#                           Functions applied on trajectories
#------------------------------------------------------------------------------#

def advanceOneFrameTraj(trajectory, stationnary):#takes a trajectory and a Bool. Returns a trajectory
    lastState = trajectory[-1]
    lastState = advanceOneFrameState(lastState,stationnary)
    trajectory.append(lastState)
    return(trajectory)

def leavesDomain(trajectoire, xMin,xMax,yMin,yMax, speedTolerance): #prends une liste de states, des bounds, une tolerance. Returns a Boolean
    lastState  = trajectoire[-1]
    expectedState = advanceOneFrameState(lastState, False)
    expectedX = expectedState[1]
    expectedY = expectedState[2]
    if (expectedX < (xMax - speedTolerance)) & (expectedX > (xMin + speedTolerance)) & (expectedY < (yMax - speedTolerance)) & (expectedY > (yMin + speedTolerance)):
        return(False)
    return(True)

def remainsTooLong(trajectoire, trajectoiresMortes, lifeTimeTolerance, baseLifeTime): #Prend une liste de tuples, liste de trajectoires, et deux reel. Returns a Boolean.
    nbMortes = len(trajectoiresMortes)
    if nbMortes < 1:
        meanLifeTime = 0
    else:
        meanLifeTime = tempsDeVieMoyenne(trajectoiresMortes, nbMortes)
    if meanLifeTime == 0:
        return(tempsDeVie(trajectoire) > baseLifeTime)
    else:
        return(tempsDeVie(trajectoire) > min(baseLifeTime, meanLifeTime + lifeTimeTolerance))

def changesDirectionTooMuch(trajectoire, trajectoiresMortes, DirectionChangeTolerance):
    vs = [state[5] for state in trajectoire]
    startVs = vs[:-1]
    endVs = vs[1:]
    changes = []
    for i in range(len(startVs)):
        change = dirChange(startVs[i],endVs[i])
        changes.append(change)
    changeCount = 0
    for ch in changes:
        if ch:
            changeCount = changeCount+1
    tooMuch = changeCount > DirectionChangeTolerance
    return(tooMuch)
