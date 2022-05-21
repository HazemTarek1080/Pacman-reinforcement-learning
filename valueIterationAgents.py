# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            vcurr = self.values.copy()
            for state in self.mdp.getStates():
                all_actions = self.mdp.getPossibleActions(state)
                transitions = []
                value_list = []
                if self.mdp.isTerminal(state):
                    self.values[state] = 0
                else:
                    for action in all_actions:
                        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                        value = 0
                        for transition in transitions:
                            value += transition[1] * (self.mdp.getReward(state, action, transition[0]) + self.discount * vcurr[transition[0]])
                        value_list.append(value)
                    self.values[state] = max(value_list)


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for transition in transitions:
            value += transition[1] * (self.mdp.getReward(state, action, transition[0]) + self.discount * self.values[transition[0]])
        return value


        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        from math import inf
        if self.mdp.isTerminal(state):
            return None
        else:
            bestval = -inf
            bestaction = 0
            all_actions = self.mdp.getPossibleActions(state)
            for action in all_actions:
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                value = 0
                for transition in transitions:
                    value += transition[1] * (self.mdp.getReward(state, action, transition[0]) + self.discount * self.values[transition[0]])
                if value > bestval:
                    bestaction = action
                    bestval = value
            return bestaction

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

        for k in range(self.iterations):
            state = self.mdp.getStates()[k % len(self.mdp.getStates())]   #computing the index of the desired state.
            best = self.computeActionFromValues(state)
            if best is None:
                V = 0
            else:
                V = self.computeQValueFromValues(state, best)
            self.values[state] = V


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Compute predecessors
        states = self.mdp.getStates()
        predecessors = {}
        for state in states:
            curr_pred = set()
            for st in states:
                for action in self.mdp.getPossibleActions(st):
                    for next, prob in self.mdp.getTransitionStatesAndProbs(st, action):
                        if (next == state and prob != 0):
                            curr_pred.add(st)
            predecessors[state] = curr_pred

        # Initialize an empty priority queue
        priority = util.PriorityQueue()


        for state in states:
            if not self.mdp.isTerminal(state):
                curr_val = self.values[state]
                q_val = []
                for action in self.mdp.getPossibleActions(state):
                    q_val += [self.computeQValueFromValues(state, action)]
                max_q_val = max(q_val)
                diff = abs((curr_val - max_q_val))
                priority.push(state, -diff)


        for i in range(0, self.iterations):
            if priority.isEmpty():
                break
            s = priority.pop()
            if not self.mdp.isTerminal(s):
                vals = []
                for action in self.mdp.getPossibleActions(s):
                    val = 0
                    for next, prob in self.mdp.getTransitionStatesAndProbs(s, action):
                        val += prob * (self.mdp.getReward(s, action, next) + (self.discount * self.values[next]))
                    vals.append(val)
                self.values[s] = max(vals)


            for pred in predecessors[s]:
                curr_val = self.values[pred]
                q_val = []
                for action in self.mdp.getPossibleActions(pred):
                    q_val += [self.computeQValueFromValues(pred, action)]
                max_q_val = max(q_val)
                diff = abs((curr_val - max_q_val ))
                if (diff > self.theta):
                    priority.update(pred, -diff)