#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
from copy import deepcopy


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def greedy( self,time_allowance=60.0 ):
		startingTime = time.time()

		allCities = self._scenario.getCities()

		cityCount = len(allCities)
		# First, pick a random node to start the path
		path = [allCities[random.randrange(cityCount)]]

		unvisitedCities = set(allCities)
		unvisitedCities.remove(path[0])
		# While we still have cities that haven't been visited yet...
		while unvisitedCities:
			# Find/append the nearest unvisited city
			nearest_city = min(unvisitedCities, key = lambda city: path[-1].costTo(city))
			path.append(nearest_city)
			unvisitedCities.remove(nearest_city)

		solution = TSPSolution(path)

		results = {}

		results['cost'] = solution.cost
		results['time'] = time.time() - startingTime
		results['count'] = 1
		results['soln'] = solution
		results['max'] = None
		results['total'] = None
		results['pruned'] = None

		return results


	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''
	
	# This is the main branch and bound logic. It runs in O(n!) time and O(n!) space. See 
	#   Write-Up for details
	def branchAndBound( self, time_allowance=60.0 ):
		startingTime = time.time()

		mainScenario = self._scenario
		initialWorkingArray = self.initializeWorkingArray(mainScenario)
		initialWorkingArray, initialLowerBound = self.computeFullReducedCostMatrix(initialWorkingArray, -1, -1)

		greedyResults = self.greedy()
		bssf = greedyResults['cost'] + 1

		maxQueueSize = 0
		currentQueueSize = 0
		totalCreatedStates = 0
		totalPrunedStates = 0

		iterationPriorityQueue = []
		# Priority Queue will contain tuples with the following format: (PriorityQueue Placement 
		#  Value, LowerBound Cost, Y index (Source Node Value), X index (Destination Node Value), Working Array, [] of all visited nodes)

		solutions = []
		# Priority Queue will contain tuples with the following format: (solution cost, ordered 
		#  list showing path for that solution)

		self.initializeHeap(initialWorkingArray, initialLowerBound, iterationPriorityQueue, bssf, maxQueueSize, totalCreatedStates, totalPrunedStates)

		# MAIN OPERATION - Run a while loop that runs while the heap is not empty and while we 
		#  still have time. 

		while (len(iterationPriorityQueue) != 0 and (time.time() - startingTime) < time_allowance): # This while loop is run at worst n! times
			nextTupleToExplore = heapq.heappop(iterationPriorityQueue)

			# If the currently stored item in the queue is worth exploring...
			if (nextTupleToExplore[1] < bssf):

				# Get all of the nodes we have already visited to avoid running those in our matrix
				alreadyVisitedNodes = nextTupleToExplore[5]

				# Set all of the nodes we need to actually work on in a list
				nodesToVisit = list(set([i for i in range(len(initialWorkingArray))]) - 
					set(alreadyVisitedNodes))
				
				# For all of the nodes in our matrix that we need to test out, make a new branch 
				# 	in our tree...
				for targetDestinationNode in nodesToVisit:
					# Copy the array to keep from modifying the parent branch
					temporaryNewArray = deepcopy(nextTupleToExplore[4])
					
					# Since we are making this new matrix, update our created States counter
					totalCreatedStates += 1

					# Come up with the reduced cost matrix for this specific tree branch
					temporaryNewArray, addedCost = self.computeIteratedReducedCostMatrix(
						temporaryNewArray, nextTupleToExplore[3], targetDestinationNode)
					
					# After calculating our next branch, see if that branch is worth diving into...
					if ((nextTupleToExplore[1] + addedCost) < bssf):
						# Update the already visited nodes list because we will declare this 
						# 	current node as visited
						updatedAlreadyVisitedNodes = deepcopy(alreadyVisitedNodes)
						updatedAlreadyVisitedNodes.append(targetDestinationNode)

						# If the length of that new list is equal to the amount of nodes we need 
						# 	to explore, we found a solution!
						if (len(updatedAlreadyVisitedNodes) == len(initialWorkingArray)):
							if (temporaryNewArray[targetDestinationNode][0] != float('inf')):
								solutionCost = nextTupleToExplore[1] + addedCost 
								solutionCost += temporaryNewArray[targetDestinationNode][0]
								# updatedAlreadyVisitedNodes stores the traversal through the 
								# 	nodes in order
								heapq.heappush(solutions, (solutionCost, updatedAlreadyVisitedNodes))
								if (solutionCost < bssf):
									# Update a new lowest cost to beat
									bssf = solutionCost
						else:	
							# If we have more nodes to go, push this new branch into our tree
							newLowestCost = nextTupleToExplore[1] + addedCost
							heapq.heappush(iterationPriorityQueue, (newLowestCost / 
					       		len(updatedAlreadyVisitedNodes), newLowestCost, 
									nextTupleToExplore[3], targetDestinationNode, 
										temporaryNewArray, updatedAlreadyVisitedNodes))

							currentQueueSize = len(iterationPriorityQueue)
							if (maxQueueSize < currentQueueSize):
								maxQueueSize = currentQueueSize
					else:
						totalCreatedStates += 1
						totalPrunedStates += 1

			# If the node that was once added to the queue is actually now more than the bssf, we 
			# 	prune it and don't explore it.
			else:
				totalPrunedStates += 1

		solutionCount = len(solutions)
		if (solutionCount != 0):
			bestSolution = heapq.heappop(solutions)
			solutionOutput = self.computeOutputSolutionArray(bestSolution[1])
		else:
			solutionOutput = []
			bestSolution = [float('inf')]

		# Now we print out the result of our operations!

		results = {}
		endingTime = time.time()
		
		results['cost'] = bestSolution[0]
		results['time'] = endingTime - startingTime
		results['count'] = solutionCount
		results['soln'] = TSPSolution(solutionOutput) if (bestSolution[0] != float('inf')) else None
		results['max'] = maxQueueSize
		results['total'] = totalCreatedStates
		results['pruned'] = totalPrunedStates

		return results


	# This function handles initializing a 2-D array used for making the branch and bound tree by 
	# 	setting all unreachable possibilities to inf. For the inf thing, use the 
	# 	scenario.edge_exists 2-d array with false/true statements.
	# This function runs in O(n^2) time and O(n^2) space
	def initializeWorkingArray( self, mainScenario):

		arraySize = len(mainScenario._cities)
		startupWorkingArray = [[0 for i in range(arraySize)] for j in range(arraySize)]

		for y in range(arraySize):
			for x in range(arraySize):
				if (mainScenario._edge_exists[y][x]):
					# Here we set the current index equal to the length of the line between the 
					# 	two points.

					newValue = mainScenario._cities[y].costTo(mainScenario._cities[x])
					
					startupWorkingArray[y][x] = newValue
				else:
					startupWorkingArray[y][x] = float('inf')

		return startupWorkingArray

		
	# This function will minimize and set up a row in the array. It can choose to "strikeThrough", 
	# 	or set an entire row to infinity, or just minimize a row numerically. 
	# This function runs in O(n) time and O(1) space. 
	def minimizeRow( self, targetRowIndex, targetColumnIndex, workingArray, strikeThrough):
		if (strikeThrough):
			minimumValue = workingArray[targetRowIndex][targetColumnIndex]
		else:
			minimumValue = float('inf')

		for x in range(len(workingArray[targetRowIndex])):
			if (strikeThrough):
				workingArray[targetRowIndex][x] = float('inf')
			else:
				if (workingArray[targetRowIndex][x] < minimumValue):
					minimumValue = workingArray[targetRowIndex][x]
		if (not strikeThrough):
			if (minimumValue > 0 and minimumValue != float('inf')):
				for x in range(len(workingArray[targetRowIndex])):
					workingArray[targetRowIndex][x] -= minimumValue
			elif (minimumValue == float('inf')):
				minimumValue = 0

		return workingArray, minimumValue
		

	# This function will minimize and set up a column in the array, similarily to the row function 
	# 	above
	# This function also runs in O(n) time and O(1) space.
	def minimizeCol( self, targetColumnIndex, targetRowIndex, workingArray, strikeThrough):
		if (strikeThrough):
			minimumValue = workingArray[targetRowIndex][targetColumnIndex]
		else:
			minimumValue = float('inf')
				
		for y in range(len(workingArray)):
			if (strikeThrough):
				workingArray[y][targetColumnIndex] = float('inf')
			else:
				if (workingArray[y][targetColumnIndex] < minimumValue):
					minimumValue = workingArray[y][targetColumnIndex]
		
		if (not strikeThrough):
			if (minimumValue > 0 and minimumValue != float('inf')):
				for y in range(len(workingArray)):
					workingArray[y][targetColumnIndex] -= minimumValue
			elif (minimumValue == float('inf')):
				minimumValue = 0

		return workingArray, minimumValue
	

	# This function computes the reduced cost matrix for a given matrix by making sure all columns
	#  and rows have at least one 0 in them.
	# This function runs in O(n^2) and O(1) space. 
	def computeFullReducedCostMatrix( self, workingArray, excludeRow, excludeColumn):
		# Note: excludeRow and excludeColumn ignore rows and columns that are set to all infinity 
		# 	or that will be
		
		costToReduce = 0

		for row in range(len(workingArray)):
			if (row != excludeRow):
				workingArray, addingCost = self.minimizeRow(row, 0, workingArray, False)
				costToReduce += addingCost

		for column in range(len(workingArray[0])):
			if (column != excludeColumn):
				workingArray, addingCost = self.minimizeCol(column, 0, workingArray, False)
				costToReduce += addingCost
		
		return workingArray, costToReduce
	

	# This function computes and returns a reduced cost matrix given a target position in the array
	# This function runs in O(n^2) time and O(1) space. 
	def computeIteratedReducedCostMatrix( self, workingArray, targetY, targetX):
		costToReduce = workingArray[targetY][targetX]

		self.minimizeRow(targetY, targetX, workingArray, True) # O(n) time

		self.minimizeCol(targetX, targetY, workingArray, True) # O(n) time

		workingArray, addingCost = self.computeFullReducedCostMatrix(workingArray, targetY, 
			targetX) # O(n^2) time
		costToReduce += addingCost

		return workingArray, costToReduce
	

	# This function makes the first branches of potential solutions by going from node 0 to all 
	# 	other nodes. 
	# This function runs in O(n^3) time, O(n^3) space
	def initializeHeap( self, initialArray, initialCost, workingHeap, bssf, maxQueueSize, 
		    totalCreatedStates, totalPrunedStates):

		temporaryArray = deepcopy(initialArray) # O(n^2) time

		for destinationNode in range(1, len(temporaryArray)): # O(n) time for each node
			newPossibleArray, newArrayCost = self.computeIteratedReducedCostMatrix(temporaryArray, 
				0, destinationNode) # O(n^2) time
			if ((initialCost + newArrayCost) < bssf):
				# O(log n) time, O(n^2) space
				heapq.heappush(workingHeap, (newArrayCost + initialCost, newArrayCost + 
					initialCost, 0, destinationNode, newPossibleArray, [0, destinationNode]))
				totalCreatedStates += 1
				maxQueueSize += 1
			else:
				totalPrunedStates += 1

			temporaryArray = deepcopy(initialArray) # O(n^2) time
			

	# This function converts a list of indexes into a solution the GUI can read
	# This runs in O(n) time and O(n) space
	def computeOutputSolutionArray( self, inputSolutionArray):
		resultList = []

		for targetIndex in inputSolutionArray:
			resultList.append(self._scenario._cities[targetIndex])

		return resultList


	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	def fancy( self,time_allowance=60.0 ):
		initialCostMatrix = self.initializeWorkingArray(self._scenario)

		inputMSTPointerList, costKeys = self.primMST(initialCostMatrix)

		minimumSpanningTree = self.outputMSTToMatrix(inputMSTPointerList, costKeys, initialCostMatrix)

		return 


	def primMST( self, inputCostMatrix):
		workingCostMatrix = deepcopy(inputCostMatrix)

		nodeParents = [-1 for i in range(len(workingCostMatrix))]
		nodesVisited = [False for i in range(len(workingCostMatrix))]
		keys = [float('inf') for i in range(len(workingCostMatrix))]

		# Start by choosing node 0
		keys[0] = 0
		nodeParents[0] = -1

		# For each of the vertices we need to add to our mst...
		for i in range(len(workingCostMatrix)) :

			targetNode = self.findMinInList(keys, nodesVisited)[1]
			
			nodesVisited[targetNode] = True

			for v in range(len(workingCostMatrix)) :
				if (workingCostMatrix[v][targetNode] is not float('inf')) and (nodesVisited[v] == 
					False) and (workingCostMatrix[v][targetNode] < keys[v]) :

					nodeParents[v] = targetNode
					keys[v] = workingCostMatrix[v][targetNode]

		return nodeParents, keys


	def findMinInList( self, keyList, inputAlreadyVisitedList):
		minData = (float('inf'), -1)

		for i in range(len(inputAlreadyVisitedList)) :
			if inputAlreadyVisitedList[i] == False and keyList[i] < minData[0] :
				minData = (keyList[i], i)
		
		return minData


	def outputMSTToMatrix( self, inputMSTPointerList, costKeys, actualCostMatrix):
		workingMatrix = [[float('inf') for i in range(len(inputMSTPointerList))] for j in range(len(inputMSTPointerList))]

		for targetToIndex in range(len(inputMSTPointerList)):
			if (inputMSTPointerList[targetToIndex] != -1):
				# This will add in both directions between nodes
				targetFromIndex = inputMSTPointerList[targetToIndex]
				workingMatrix[targetFromIndex][targetToIndex] = actualCostMatrix[targetFromIndex][targetToIndex]
				workingMatrix[targetToIndex][targetFromIndex] = actualCostMatrix[targetToIndex][targetFromIndex]

		return workingMatrix