# ga_for_graph_coloring

# General
The graph coloring problem is kind of curious problem. We have to allocate colors in graph (in my case undirected) to minimze connection between nodes which have the same color. To solved this problem I used a genetic algorithm. To use genetic algorithm I had to specify how to present the individual. In my case I setup a simple vector, which describe a number of nodes and a number on i-th position describe color. For example for three nodes and three colors a solution individual could be described as [0, 1, 2] -> first node got first color, second node got second color etc. So if I can write my problem in vector I can use genetic alogrithm. As we know genetic algorithms have a lot of hyperparameters. I used Optuna with MlFlow to search parameters spaces. For testing sake I generated five big adjency matricies (to describe graph) - it had from 50 to 100 nodes. As a error I counted a number of connections between nodes of the same colors (we want to minimize it).

# Experiments
To optimally use genetic alogirthms I wanted to optimize its hyperparameters. To do it I used an Optuna and a MLflow. I defined searching space (crossover type, selection type etc.) and then run 30 trails of optuna optimization. The best five set of params:
![image](https://github.com/user-attachments/assets/0d317e3f-9f4a-4272-8724-f7e36c1903a8)


The best params on comparing plot:
![image](https://github.com/user-attachments/assets/1432aa1e-7dc7-47a3-99a6-ab55980ef5b4)

As we can see in most cases the tournament selection and two point crossover was the best. The increasing number of generations and magnitude of population not always influence error in good way, but always increaced time spent to train. The crossover rate was kept high (about 0.90) and mutation rate was kept moderate (about 0.40).

Example colored graph:
![image](https://github.com/user-attachments/assets/8acb0266-c539-42c5-a449-3c755f647146)

# Technologies
- Deap
- Optuna
- Mlflow
- Numpy
- Networkx
# Status
The project has been completed
