# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        closestFood = float("inf") 
        
        for food in foodList:
            closestFood = min(closestFood, manhattanDistance(newPos, food))

        for ghost in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 2):
                return -float("inf")

        return successorGameState.getScore() + 1.0/closestFood

def scoreEvaluationFunction(currentGameState):
    
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.maxVal(gameState, 0, 0)[0]

    def miniMax(self, gameState, agentIndex, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex == 0: 
            return self.maxVal(gameState, agentIndex, depth)[1]
        else:
            return self.minVal(gameState, agentIndex, depth)[1]

    def maxVal(self, gameState, agentIndex, depth):
        bestAction = None
        bestValue = -float("inf")
        
        for action in gameState.getLegalActions(agentIndex):
            succValue = self.miniMax(gameState.generateSuccessor(agentIndex, action), (depth + 1) % gameState.getNumAgents(), depth + 1)
            if succValue > bestValue:
                bestValue = succValue
                bestAction = action
                
        return bestAction, bestValue

    def minVal(self, gameState, agentIndex, depth):
        bestAction = None
        bestValue = float("inf")
        
        for action in gameState.getLegalActions(agentIndex):
            succValue = self.miniMax(gameState.generateSuccessor(agentIndex, action), (depth + 1) % gameState.getNumAgents(), depth + 1)
            
            if succValue < bestValue:
                bestValue = succValue
                bestAction = action
                
        return bestAction, bestValue

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxVal(gameState, 0, 0, -float("inf"), float("inf"))[0]

    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta):
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxVal(gameState, agentIndex, depth, alpha, beta)[1]
        else:
            return self.minVal(gameState, agentIndex, depth, alpha, beta)[1]

    def maxVal(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = ("max", -float("inf"))
        
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action, self.alphaBeta(gameState.generateSuccessor(agentIndex,action), (depth + 1) % gameState.getNumAgents(), depth + 1, alpha, beta))
            
            if succAction[1] > bestAction[1]:
                bestAction = succAction

            if bestAction[1] > beta: 
                return bestAction
            else: 
                alpha = max(alpha, bestAction[1])

        return bestAction

    def minVal(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = ("min", float("inf"))
        
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action, self.alphaBeta(gameState.generateSuccessor(agentIndex, action), (depth + 1) % gameState.getNumAgents(), depth + 1, alpha, beta))
            
            if succAction[1] < bestAction[1]:
                bestAction = succAction

            if bestAction[1] < alpha: 
                return bestAction
            else: 
                beta = min(beta, bestAction[1])

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        maxDepth = self.depth * gameState.getNumAgents()
        return self.expectiMax(gameState, "expect", maxDepth, 0)[0]
    
    def expectiMax(self, gameState, action, depth, agentIndex):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return (action, self.evaluationFunction(gameState))
        if agentIndex == 0:
            return self.maxVal(gameState, action, depth, agentIndex)
        else:
            return self.expVal(gameState, action, depth, agentIndex)
    
    def maxVal(self, gameState, action, depth, agentIndex):
        bestAction = ("max", -float("inf"))
        
        for legalAction in gameState.getLegalActions(agentIndex):
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            succAction = None
            
            if depth != self.depth * gameState.getNumAgents():
                succAction = action
            else:
                succAction = legalAction
                
            succVal = self.expectiMax(gameState.generateSuccessor(agentIndex, legalAction), succAction, depth - 1, nextAgent)
            
            if succVal[1] > bestAction[1]:
                bestAction = succVal

        return bestAction
        
    def expVal(self, gameState, action, depth, agentIndex):
        legalActions = gameState.getLegalActions(agentIndex)
        avgScore = 0;
        chance = 1.0 / len(legalActions)
        
        for legalAction in legalActions:
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            bestAction = self.expectiMax(gameState.generateSuccessor(agentIndex, legalAction), action, depth - 1, nextAgent)
            avgScore += bestAction[1] * chance
            
        return (action, avgScore)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    
    This evaluation function considers multiple factors to determine the desirability
    of a given game state. It calculates scores based on Pac-Man's position, distances
    to ghosts, remaining food and capsules, and the overall game state outcome.

    Factors considered:
    1. Pac-Man's proximity to remaining food.
    2. Avoidance of ghosts within a certain distance.
    3. Remaining food and capsules count, with respective multipliers.
    4. Distance to the nearest food pellet.
    5. Game outcome (win/loss) penalties and rewards.

    Parameters:
    - currentGameState: GameState object representing the current state of the game.

    Returns:
    - Score representing the desirability of the given game state.
    
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    foodList = newFood.asList()
    minFood = float('inf')
    
    for food in foodList:
        minFood = min(minFood, manhattanDistance(newPos, food))

    ghostDist = 0
    ghostList = currentGameState.getGhostPositions()
    
    for ghost in ghostList:
        ghostDist = manhattanDistance(newPos, ghost)
        
        if (ghostDist < 2):
            return -float('inf')

    foodLeft = currentGameState.getNumFood()
    capsLeft = len(currentGameState.getCapsules())
    foodLeftMulti = 100000
    capsLeftMulti = 10000
    foodDistMulti = 1000
    factors = 0
    
    if currentGameState.isLose():
        factors -= 9999999
    elif currentGameState.isWin():
        factors += 9999999

    return 1.0/(foodLeft + 1) * foodLeftMulti + ghostDist + 1.0/(minFood + 1) * foodDistMulti + 1.0/(capsLeft + 1) * capsLeftMulti + factors
           
# Abbreviation
better = betterEvaluationFunction
