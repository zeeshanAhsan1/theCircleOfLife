# The Circle Of Life
The purpose of this project is to build a probabilistic model of an environment in the presence of uncertainty, and use it to inform and direct decision making.

## Environment Setup
The environment for this project is a graph of nodes connected by edges. The agent, the prey, and the predator, can move between the nodes along the edges. There are 50 nodes, numbered 1 to 50, connected in a large circle. Additionally, add edges at random to increase connectivity across the circle, in the following way:

- Picking a random node with degree less than 3.
- Add an edge between it and one node within 5 steps forward or backward along the primary loop.
- Do this until no more edges can be added.


