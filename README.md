# The Circle Of Life
The purpose of this project is to build a probabilistic model of an environment in the presence of uncertainty, and use it to inform and direct decision making.

## Environment Setup
The environment for this project is a graph of nodes connected by edges. The agent, the prey, and the predator, can move between the nodes along the edges. There are 50 nodes, numbered 1 to 50, connected in a large circle. Additionally, add edges at random to increase connectivity across the circle, in the following way:

- Picking a random node with degree less than 3.
- Add an edge between it and one node within 5 steps forward or backward along the primary loop.
- Do this until no more edges can be added.

## The Problem : Predator, Prey and Agent
This environment is occupied by three entities, the Predator, the Prey, and the Agent, who can move from node to node along the edges. The Agent wants to catch the Prey; the Predator wants to catch the Agent. If the Agent and the Prey occupy the same node, the Agent wins. If the Agent and the Predator occupy the same node, the Agent loses.The three players move in rounds, starting with the Agent, followed by the Prey, followed by the Predator.

### Movement of Prey
The rules for the Prey are simple - every time the Prey moves, it selects among its neighbors or its current cell, uniformly at random (i.e., if it has 3 neighbors, there is a 1/4 probability of it staying where it is.) This continues, regardless of the actions or locations of the others, until the game concludes.

### Movement of Predator
Every time the Predator moves, it looks at its available neighbors, and calculates the shortest distance to the Agent for each neighbor it can move to. It then moves to the neighbor with shortest distance remaining to the Agent. If multiple neighbors have the same distance to the Agent, the Predator selects uniformly at random between them.

### Movement of Agent
The motion of the Agent is dictated by the specific strategy of the Agent. In partial information settings, the Agent may choose to survey a node at a distance to determine what if anything is in that node before deciding where to move. (Imagine, for instance, the Agent sending out a drone to spy on a given location.) Note, the Agent is aware of how the Predator and Prey choose the actions they are going to take, though is unaware of the specific actions chosen.

## Agents and Environments
The different kinds of conditions for the agent to pursue the prey are - 

### The Complete Information Setting -
In this setting, the Agent always knows exactly where the Predator is and where the Prey is.

AGENT 1: Whenever it is this Agent’s turn to move, it will examine each of its available neighbors and select from them in the following order (breaking ties at random). The rules for agent 1 movement are :

- Neighbors that are closer to the Prey and farther from the Predator.
- Neighbors that are closer to the Prey and not closer to the Predator.
- Neighbors that are not farther from the Prey and farther from the Predator.
- Neighbors that are not farther from the Prey and not closer to the Predator.
- Neighbors that are farther from the Predator.
- Neighbors that are not closer to the Predator.
- Sit still and pray.

AGENT 2: This agent is of my own design. This agent can look up and check if the neighbors within 5 moves away of current position have predator then it moves away from the predator, otherwise it moves to close the distance to the prey.

### The Partial Information Setting -
In this setting, the Agent always knows where the Predator is, but does not necessarily know where the Prey is. Every time the Agent moves, the Agent can first choose a node to survey (anywhere in the graph) to determine whether or not the Prey is there. Additionally, the Agent gains information about where the Prey isn’t every time it enters a node and the Prey isn’t there. In this setting, the Agent needs to track a belief state for where the Prey is, a collection of probabilities for each node that the Prey is there. Every time the Agent learns something about the Prey, these probabilities need to be updated. Every time the Prey is known to move, these probabilities need to be updated.

AGENT 3: Whenever it is this Agent’s turn to move, if it is not currently certain where the Prey is, it will survey the node with the highest probability of containing the Prey (breaking ties at random), and update the probabilities based on the result. After this, it will assume that the Prey is located in the node with the now highest probability of containing the Prey (breaking ties at random), and will act in accordance with the rules for Agent 1.

AGENT 4: Implemented with similar logic as Agent 2 but with partial information about the prey position.

### The Partial Predator Information Setting -
In this setting, the Agent always knows where the Prey is, but does not necessarily know where the Predator is. Every time the Agent moves, the Agent can first choose a node to survey (anywhere in the graph) to determine whether or not the Predator is there. Additionally, the Agent gains information about where the Predator isn’t every time it enters a node and the Predator isn’t there. In this setting, the Agent needs to track a belief state for where the Predator is, a collection of probabilities for each node that the Predator is there. Every time the Agent learns something about the Predator, these probabilities need to be updated. Every time the Predator is known to move, these probabilities need to be updated.

The predator moves 60% of the times to close the distance to agent and 40% of the times randomly to any neighbor of the current position of the predator.

AGENT 5: 
Whenever it is this Agent’s turn to move, if it is not currently certain where the Predator is, it will survey the node with the highest probability of containing the Predator (breaking ties first based on proximity to the Agent, then at random), and update the probabilities based on the result. After this, it will assume the Predator is located at the node with the now highest probability of containing the Predator (breaking ties first based on proximity to the Agent, then at random), and will act in accordance with the rules for Agent 1.

AGENT 6: Implemented with similar logic as Agent 2 but with partial information about the predator position.

### The Combined Partial Infromation Setting -
In this setting, the Agent does not necessarily know where the Predator or Prey are. Every time the Agent moves, the Agent can first choose a node to survey (anywhere in the graph) to determine who occupies that node. In this setting, the Agent needs to keep track of belief states for both the Predator and the Prey, updating them based on information collected and knowledge of the actions of the two players.

Here also the predator moves 60% of the times to close the distance to agent and 40% of the times randomly to any neighbor of the current position of the predator.

AGENT 7: Whenever it is this Agent’s turn to move, if it is not currently certain where the predator is, it will survey in accordance with Agent 5. If it knows where the Predator is, but not the Prey, it will survey in accordance with Agent 3. As before, however, this Agent only surveys once per round. Once probabilities are updated based on the results of the survey, the Agent acts by assuming the Prey is at the node of highest probability of containing the Prey (breaking ties at random) and assuming the Predator is at the node of highest probability of containing the Predator (breaking ties by proximity to the Agent, then at random). It then applies the actions of Agent 1.

AGENT 8: Implemented with similar logic as Agent 2 but with partial prey and predator information.





