# ga_for_graph_coloring

# General
The graph coloring problem is kind of curious problem. We have to allocate colors in graph (in my case undirected) to minimze connection between nodes which have the same color. To solved this problem I used a genetic algorithm. To use genetic algorithm I had to specify how to present the individual. In my case I setup a simple vector, which describe a number of nodes and a number on i-th position describe color. For example for three nodes and three colors a solution individual could be described as [0, 1, 2] -> first node got first color, second node got second color etc. So if I can write my problem in vector I can use genetic alogrithm. As we know genetic algorithms have a lot of hyperparameters. I used Optuna with MlFlow to search parameters spaces. For testing sake I generated five big adjency matricies (to describe graph) - it had from 50 to 100 nodes. As a error I counted a number of connections between nodes of the same colors (we want to minimize it).
# Technologies
- Deap
- Optuna
- Mlflow
- Numpy
- Networkx
# Status
The project has been completed
