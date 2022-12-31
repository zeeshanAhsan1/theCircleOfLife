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




