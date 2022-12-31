import random
import sys
import numpy as np
import matplotlib.pyplot as plt

no_of_nodes=50
probability_table=[]
pred_probab_table=[]
threshold=500
prob_count_ag3 = []
prob_count_ag5 = []
prob_count_prey_ag7 =[]
prob_count_pred_ag7 = []
prob_count_prey_fault_drone_ag7 = []
prob_count_pred_fault_drone_ag7 = []

prob_count_prey_fault_drone_ag8 = []
prob_count_pred_fault_drone_ag8 = []


class Prey:
  def __init__(self, position, parent, distance):
    self.position = position
    self.distance = distance
    self.parent = parent

class Predator:
  def __init__(self, position, parent, distance_Agent):
    self.position = position
    self.distance_Agent = distance_Agent
    self.parent = parent 

class Agent:
  def __init__(self, position, parent, distance_Pred, distance_Prey):
    self.position = position
    self.distance_Pred = distance_Pred
    self.distance_Prey = distance_Prey
    self.parent = parent       

def djikstra(graph,start,end):

    unvisited = {}
    curVertex = start
    for i in range(1,no_of_nodes+1):
        unvisited[i] = float('inf')

    unvisited[start] = 0
    visited = {}

    curVertex = min(unvisited, key=unvisited.get)

    while unvisited:

        curVertex = min(unvisited, key=unvisited.get)
        visited[curVertex] = unvisited[curVertex]

        if curVertex == end:
            return visited[end]

        for nbr in graph.get(curVertex):
            if nbr in visited:
                continue
            tempDist = unvisited[curVertex] + 1
            if(tempDist < unvisited[nbr]):
                unvisited[nbr] = tempDist

        unvisited.pop(curVertex)

def Agent1(adjacency_list):
    index=list(range(1,no_of_nodes+1))
    Agent_pos = np.random.choice(index)
    index_pred=list(range(1,no_of_nodes+1))
    index_pred.remove(Agent_pos)
    predator_pos = np.random.choice(index_pred)
    prey_pos = np.random.choice(index_pred)
    Agent1 = Agent(Agent_pos,0,0,0)
    Predator1 = Predator(predator_pos,0,0)
    Prey1 = Prey(prey_pos,0,0)
    count=0
    flag=0
    while(True):
        count=count+1
        if(count==threshold):
            break
        print("Agent1: ",Agent1.position," Predator: ",Predator1.position," Prey: ",Prey1.position)
        dist={}
        prey_list=[]
        for i in adjacency_list[Agent1.position]:
            temp_prey=djikstra(adjacency_list,i,Prey1.position)
            temp_pred=djikstra(adjacency_list,i,Predator1.position)
            dist[i]=[temp_prey,temp_pred]
        sorted_list_dist = sorted(dist.items(), key=lambda x:x[1])
        dist=dict(sorted_list_dist) 
        print("dist ",dist) 
        var=list(dist.keys())[0]  
        agent_next_pos=0                                                 
        for i in dist:
            if(dist[i][0] == dist[var][0] and dist[i][1] >= 2):
                agent_next_pos=i
        
        maximus=0
        if(agent_next_pos==0):
            for i in dist:
                if(dist[i][1] != 0):          
                    if(i>maximus):
                        agent_next_pos=i
                        maximus=i
        
        if(agent_next_pos!=0):                                            
            Agent1.position=agent_next_pos
        print("Agent new position ",Agent1.position)
        
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break
        if(Agent1.position == Prey1.position):
            flag=1
            print("Agent won")
            break
        
        l_prey=[]
        for j in adjacency_list[Prey1.position]:
            l_prey.append(j)
        l_prey.append(Prey1.position)    
        prey_next_point=np.random.choice(l_prey)
        Prey1.position=prey_next_point

        if(Agent1.position == Prey1.position):
            flag=1
            print("Agent won")
            break

        min_dist_agent=100000
        for i in adjacency_list[Predator1.position]:
            temp=djikstra(adjacency_list,i,Agent1.position)
            print("Predator1 temp ",temp," i ", i)
            if(temp<min_dist_agent):
                pred_next_pos=i
                min_dist_agent=temp
        Predator1.position=pred_next_pos  
        print("Predator1 new position ",Predator1.position) 
        
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break
        

             
    if(flag==1):
        return 1
    if(flag==2):
        return 2
    if(flag==0):
        return 0    

def Agent2(adjacency_list):
    index=list(range(1,no_of_nodes+1))
    Agent_pos = np.random.choice(index)
    index_pred=list(range(1,no_of_nodes+1))
    index_pred.remove(Agent_pos)
    predator_pos = np.random.choice(index_pred)
    prey_pos = np.random.choice(index_pred)
    #print(Agent_pos,predator_pos,prey_pos)
    Agent1 = Agent(Agent_pos,0,0,0)
    Predator1 = Predator(predator_pos,0,0)
    Prey1 = Prey(prey_pos,0,0)
    count=0
    flag=0
    while(True):
        count=count+1
        if(count==threshold):
            break
        print("Agent1: ",Agent1.position," Predator: ",Predator1.position," Prey: ",Prey1.position)
        dist_prey={}
        dist_pred={}
        dist={}
        prey_list=[]
        for i in adjacency_list[Agent1.position]:
            temp_prey=djikstra(adjacency_list,i,Prey1.position)
            temp_pred=djikstra(adjacency_list,i,Predator1.position)
            if(temp_prey<5):
                dist_prey[i]=[temp_prey]
            if(temp_pred<5):
                dist_pred[i]=[temp_pred]
            if(temp_prey>=5 and temp_pred>=5):
                dist[i]=[temp_prey,temp_pred]  
        if(len(dist_pred)>0):
            sorted_list_dist = sorted(dist_pred.items(), key=lambda x:x[1])
            dist_pred=dict(sorted_list_dist)
            print("dist_pred ",dist_pred)
            agent_next_pos=list(dist_pred.keys())[len(dist_pred)-1]
        elif(len(dist_prey)>0):
            sorted_list_dist = sorted(dist_prey.items(), key=lambda x:x[1])
            dist_prey=dict(sorted_list_dist)
            print("dist_prey ",dist_prey)
            agent_next_pos=list(dist_prey.keys())[0]    
        else:    
            sorted_list_dist = sorted(dist.items(), key=lambda x:x[1])
            dist=dict(sorted_list_dist) 
            print("dist ",dist) 
            var=list(dist.keys())[0]  
            agent_next_pos=0                                                 
            for i in dist:
                if(dist[i][0] == dist[var][0] and dist[i][1] >2):
                    agent_next_pos=i
            
            maximus=0
            if(agent_next_pos==0):
                for i in dist:
                    if(dist[i][1] != 0):          
                        if(i>maximus):
                            agent_next_pos=i
                            maximus=i
        
        if(agent_next_pos!=0):                                            
            Agent1.position=agent_next_pos
        print("Agent new position ",Agent1.position)
        
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break
        if(Agent1.position == Prey1.position):
            flag=1
            print("Agent won")
            break
        
        l_prey=[]
        for j in adjacency_list[Prey1.position]:
            l_prey.append(j)
        l_prey.append(Prey1.position)    
        prey_next_point=np.random.choice(l_prey)
        Prey1.position=prey_next_point

        if(Agent1.position == Prey1.position):
            flag=1
            print("Agent won")
            break

        min_dist_agent=100000
        for i in adjacency_list[Predator1.position]:
            temp=djikstra(adjacency_list,i,Agent1.position)
            print("Predator1 temp ",temp," i ", i)
            if(temp<min_dist_agent):
                pred_next_pos=i
                min_dist_agent=temp
        Predator1.position=pred_next_pos  
        print("Predator1 new position ",Predator1.position) 
        
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break
        

             
    if(flag==1):
        return 1
    if(flag==2):
        return 2
    if(flag==0):
        return 0 

def Survey(index,prey):
    if(index==prey):
        return 1
    else:
        return 0    
    
def Survey_pred(index,prey):
    if(index==prey):
        return 1
    else:
        return 0



def probability_distribution(agentPos,surveyNode,surveyRes,graph):

    copy_prob_table = [None] * len(probability_table)
    for i in range(len(probability_table)):
        copy_prob_table[i] = probability_table[i]
    

    #Indices of table are from 0 to 49.
    probability_table[agentPos-1] = 0
    # if(surveyRes == 0):
    #     probability_table[surveyNode-1] = 0

    if(surveyRes == 1):
        #probability_table[surveyNode-1] = 1
        for i in range(len(probability_table)):
            probability_table[i] = 0
        probability_table[surveyNode-1] = 1
        return


    elif(surveyRes == 0):

        probability_table[surveyNode-1] = 0

        for i in range(len(probability_table)):
            if(i == agentPos-1):
                continue
            if(i == surveyNode-1):
                continue
            else:
                probability_res_i = 0
                for j in range(len(probability_table)):
                    if((i+1) in graph[j+1]):
                        fact2 = 1/(len(graph[j+1])+1)
                    else:
                        fact2 = 0

                    probability_res_i += copy_prob_table[j] * fact2
            
                probability_table[i] = probability_res_i

def probability_distribution2(graph):

    #Previous Probabilities
    copy_prob_table = [None] * len(probability_table)
    for i in range(len(probability_table)):
        copy_prob_table[i] = probability_table[i]
    
    #Calculating and populating next probabilities after the prey has moved
    for i in range(len(probability_table)):
        prob_res_i = 0
        for j in range(len(probability_table)):
            if((i+1) in graph[j+1] or (i+1) == (j+1)):
                fact2 = 1/(len(graph[j+1]) + 1)
            else:
                fact2 = 0
            prob_res_i += copy_prob_table[j] * fact2

        probability_table[i] = prob_res_i

def probab_pred_distribution(graph,agentPos):

    #Previous Probabilities
    copy_prob_table = [None] * len(pred_probab_table)
    for i in range(len(pred_probab_table)):
        copy_prob_table[i] = pred_probab_table[i]
    
    #Calculating and populating next probabilities after the prey has moved
    for i in range(len(copy_prob_table)):
        prob_res_i = 0
        for j in range(len(copy_prob_table)):
            if((i+1) in graph[j+1]):
                min_pred_dist = 0
                pred_Arr = []
                min_d = 100000
                min_d_i = []
                for x in graph[j+1]:
                    tempdist = djikstra(graph, x, agentPos)
                    tupEl = (x,tempdist)
                    pred_Arr.append(tupEl)
                for g in range(len(pred_Arr)):
                    if(pred_Arr[g][1] <= min_d):
                        min_d = pred_Arr[g][1]
                        min_d_i.append(pred_Arr[g][0])
                    

                options = 0
                for g in pred_Arr:
                    if(min_d == g[1]):
                        options +=1
                if(options == 1):
                    co_or = min_d_i[0] - 1
                    for t in range(len(pred_probab_table)):
                        pred_probab_table[t] = 0
                        pred_probab_table[co_or] = 1


                elif(options == 2):
                    co_or1 = min_d_i[0] - 1   
                    co_or2 = min_d_i[1] - 1
                    for t in range(len(pred_probab_table)):
                        pred_probab_table[t] = 0
                    pred_probab_table[co_or1] = 1/2
                    pred_probab_table[co_or2] = 1/2
                elif(options == 3):
                    co_or1 = min_d_i[0] - 1
                    co_or2 = min_d_i[1] - 1
                    co_or3 = min_d_i[2] - 1
                    
                    for t in range(len(pred_probab_table)):
                        pred_probab_table[t] = 0
                    pred_probab_table[co_or1] = 1/3
                    pred_probab_table[co_or2] = 1/3
                    pred_probab_table[co_or3] = 1/3

                else:
                    for t in range(len(pred_probab_table)):
                        pred_probab_table[t] = 0

            else:
                for t in range(len(pred_probab_table)):
                    pred_probab_table[t] = copy_prob_table[t]
                

def prob_distracted(graph):
    #Previous Probabilities
    copy_prob_table = [None] * len(pred_probab_table)
    for i in range(len(pred_probab_table)):
        copy_prob_table[i] = pred_probab_table[i]
    
    #Calculating and populating next probabilities after the prey has moved

    for i in range(len(pred_probab_table)):
        prob_res_i = 0
        for j in range(len(pred_probab_table)):
            #print("J = ", j)
            if((i+1) in graph[j+1]):
                fact2 = 1/(len(graph[j+1]))
            else:
                fact2 = 0
            prob_res_i += copy_prob_table[j] * fact2

        pred_probab_table[i] = prob_res_i

def Ag3(adjacency_list):
    count_prey_pos = 0
    index=list(range(1,no_of_nodes+1))
    Agent_pos = np.random.choice(index)
    index_pred=list(range(1,no_of_nodes+1))
    index_pred.remove(Agent_pos)
    predator_pos = np.random.choice(index_pred)
    prey_pos = np.random.choice(index_pred)

    Agent1 = Agent(Agent_pos,0,0,0)
    Predator1 = Predator(predator_pos,0,0)
    Prey1 = Prey(prey_pos,0,0)

    #Clearing probability table
    probability_table.clear()
    #Inserting Raw probabilities
    for x in range(no_of_nodes):
        probability_table.append(1/no_of_nodes)
    
    probability_table[Agent1.position-1]=0

    for y in range(no_of_nodes):
        if(y != Agent1.position-1):
            probability_table[y]=(1/(no_of_nodes-1))

    #Initial Survey
    surveyList = list(range(1,no_of_nodes+1))
    surveyList.remove(Agent_pos)
    surveyNode = np.random.choice(surveyList)

    surveyRes = Survey(surveyNode, Prey1.position)
    count=1
    flag=0
    loopCounter = 0
    while(True):
        loopCounter += 1
        if(count==threshold):                                         # VISH
            break
        count=count+1
        
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break
        
        if(surveyRes == 1):

            #Update the count for Agent knowing the exact position of prey
            count_prey_pos +=1 

            #Prey Found -> Update Probab table
            for i in range(len(probability_table)):
                probability_table[i] = 0
            probability_table[surveyNode-1] = 1
            
            #print("Probability Table after survey prey found :", probability_table)
            print("Sum of prob table after survey prey found :", sum(probability_table))

            # Prey is at survey Node -> Agent has to move towards Survey Node
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                dist[i]=[temp_prey,temp_pred]
            
            sorted_list_dist = sorted(dist.items(), key=lambda x:x[1])
            dist=dict(sorted_list_dist) # Sorted dict of distances
            print("dist ",dist) 
            var=list(dist.keys())[0]  
            agent_next_pos=0                                                 # Today
            for i in dist:
                if(dist[i][0] == dist[var][0] and dist[i][1] != 0):          # Today
                    agent_next_pos=i
            # TODAY
            if(agent_next_pos==0):
                print("HERE HERE HERE HERE HERE HERE")
                for i in dist:
                    if(dist[i][1] != 0):          
                        agent_next_pos=i
            # TODAY
            if(agent_next_pos != 0):                                            # TODAY
                Agent1.position=agent_next_pos


        else:
            #Prey Not Found -> Update Probab table
            print("Agent Position", Agent1.position)
            for i in range(len(probability_table)):
                probability_table[i] = 1/(no_of_nodes-2)
            probability_table[surveyNode-1] = 0
            probability_table[Agent1.position-1] = 0
            #print("Probability Table after survey no prey found :", probability_table)
            print("Sum of prob table after survey no prey found :", sum(probability_table))

            # As all probabilities are equal -> Calculate farthest movement from Predator
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                dist[i] = temp_pred
            
            maxDistance = 0
            maxNbr = 0
            for key,val in dist.items():
                if (val > maxDistance):
                    maxNbr = key
                    maxDistance = val
            
            # Move the agent

            Agent1.position = maxNbr

        # Check if Agent Won /died                                 #VISH 
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break
        if(Agent1.position == Prey1.position):
            flag=1
            print("Agent won")
            break
        

        # As agent has moved update the probability table with new information
        for x in range(len(probability_table)):
            probability_table[x] = 1/(no_of_nodes-2)
        probability_table[Agent1.position - 1] = 0
        probability_table[surveyNode - 1] = 0
        #print("Probability Table after survey no prey found and agent move:", probability_table)
        print("Sum of prob table after survey no prey found and agent move:", sum(probability_table))

        #Now moving the prey
        l_prey=[]
        for j in adjacency_list[Prey1.position]:
            l_prey.append(j)
        l_prey.append(Prey1.position)    
        prey_next_point=np.random.choice(l_prey)
        Prey1.position=prey_next_point

        #Check if Agent Won
        # if(Agent1.position == Prey1.position):
        #     flag=1
        #     print("Agent won")
        #     break

        #Update belief Probablity Table of Agent
        probability_distribution2(adjacency_list)
        if (1 in probability_table):
            count_prey_pos += 1
        #print("Probability Table after prey move :", probability_table)
        print("Sum of prob table after prey moves :", sum(probability_table))

        #Move predator
        min_dist_agent=100000
        for i in adjacency_list[Predator1.position]:
            temp=djikstra(adjacency_list,i,Agent1.position)
            #print("Predator1 temp ",temp," i ", i)
            if(temp<min_dist_agent):
                pred_next_pos=i
                min_dist_agent=temp
        Predator1.position=pred_next_pos

        #Check if predator won
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break

        # Take the max of the probability table for position of the new survey
        maxProbab = max(probability_table)
        #maxPosition = probability_table.index(maxProbab) + 1 #Adding 1 to compensate for the indices starting from 0 rather than 1

        tempL = []
        for f in range(len(probability_table)):
            if(probability_table[f] == maxProbab):
                tempL.append(f)

        maxPosition = np.random.choice(tempL) + 1
        if(maxPosition == Prey1.position):
            count_prey_pos += 1

        
        # Do a survey on this maxPosition
        #prey_pos = Prey1.position
        surveyNode = maxPosition
        #print("Survey Node:", surveyNode)
        surveyRes = Survey(surveyNode, Prey1.position)
    
    
    percentCalc = (count_prey_pos/loopCounter) * 100
    prob_count_ag3.append(percentCalc)
    if(flag==1):
        return 1
    if(flag==2):
        return 2
    if(flag==0):
        return 0

def Agent4(adjacency_list):
    index=list(range(1,no_of_nodes+1))
    Agent_pos = np.random.choice(index)
    index_pred=list(range(1,no_of_nodes+1))
    index_pred.remove(Agent_pos)
    predator_pos = np.random.choice(index_pred)
    prey_pos = np.random.choice(index_pred)

    Agent1 = Agent(Agent_pos,0,0,0)
    Predator1 = Predator(predator_pos,0,0)
    Prey1 = Prey(prey_pos,0,0)

    #Clearing probability table
    probability_table.clear()
    #Inserting Raw probabilities
    for x in range(no_of_nodes):
        probability_table.append(1/no_of_nodes)
    
    probability_table[Agent1.position-1]=0

    for y in range(no_of_nodes):
        if(y != Agent1.position-1):
            probability_table[y]=(1/(no_of_nodes-1))

    #Initial Survey
    surveyList = list(range(1,no_of_nodes+1))
    surveyList.remove(Agent_pos)
    surveyNode = np.random.choice(surveyList)

    surveyRes = Survey(surveyNode, Prey1.position)
    count=1
    flag=0
    while(True):
        if(count==threshold):                                         # VISH
            break
        count=count+1
        
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break
        
        if(surveyRes == 1):
            #Prey Found -> Update Probab table
            for i in range(len(probability_table)):
                probability_table[i] = 0
            probability_table[surveyNode-1] = 1
            
            print("Probability Table after survey prey found :", probability_table)
            print("Sum of prob table after survey prey found :", sum(probability_table))

            # Prey is at survey Node -> Agent has to move towards Survey Node
            dist_prey={}
            dist_pred={}
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                if(temp_prey<5):
                    dist_prey[i]=[temp_prey]
                if(temp_pred<5):
                    dist_pred[i]=[temp_pred]
                if(temp_prey>=5 and temp_pred>=5):
                    dist[i]=[temp_prey,temp_pred]  
            if(len(dist_pred)>0):
                sorted_list_dist = sorted(dist_pred.items(), key=lambda x:x[1])
                dist_pred=dict(sorted_list_dist)
                print("dist_pred ",dist_pred)
                agent_next_pos=list(dist_pred.keys())[len(dist_pred)-1]
            elif(len(dist_prey)>0):
                sorted_list_dist = sorted(dist_prey.items(), key=lambda x:x[1])
                dist_prey=dict(sorted_list_dist)
                print("dist_prey ",dist_prey)
                agent_next_pos=list(dist_prey.keys())[0]    
            else:    
                sorted_list_dist = sorted(dist.items(), key=lambda x:x[1])
                dist=dict(sorted_list_dist) 
                print("dist ",dist) 
                var=list(dist.keys())[0]  
                agent_next_pos=0                                                 
                for i in dist:
                    if(dist[i][0] == dist[var][0] and dist[i][1] >2):
                        agent_next_pos=i
                
                maximus=0
                if(agent_next_pos==0):
                    print("HERE HERE HERE HERE HERE HERE")
                    for i in dist:
                        if(dist[i][1] != 0):          
                            if(i>maximus):
                                agent_next_pos=i
                                maximus=i
            
            if(agent_next_pos!=0):                                            
                Agent1.position=agent_next_pos
            print("Agent new position ",Agent1.position)


        else:
            #Prey Not Found -> Update Probab table
            print("Agent Position", Agent1.position)
            for i in range(len(probability_table)):
                probability_table[i] = 1/(no_of_nodes-2)
            probability_table[surveyNode-1] = 0
            probability_table[Agent1.position-1] = 0
            print("Probability Table after survey no prey found :", probability_table)
            print("Sum of prob table after survey no prey found :", sum(probability_table))

            # As all probabilities are equal -> Calculate farthest movement from Predator
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                dist[i] = temp_pred
            
            maxDistance = 0
            maxNbr = 0
            for key,val in dist.items():
                if (val > maxDistance):
                    maxNbr = key
                    maxDistance = val
            
            # Move the agent

            Agent1.position = maxNbr

        #Check if Agent Won /died                                 #VISH 
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break
        if(Agent1.position == Prey1.position):
            flag=1
            print("Agent won")
            break
        

        # As agent has moved update the probability table with new information
        for x in range(len(probability_table)):
            probability_table[x] = 1/(no_of_nodes-2)
        probability_table[Agent1.position - 1] = 0
        probability_table[surveyNode - 1] = 0
        print("Probability Table after survey no prey found and agent move:", probability_table)
        print("Sum of prob table after survey no prey found and agent move:", sum(probability_table))

        #Now moving the prey
        l_prey=[]
        for j in adjacency_list[Prey1.position]:
            l_prey.append(j)
        l_prey.append(Prey1.position)    
        prey_next_point=np.random.choice(l_prey)
        Prey1.position=prey_next_point

        #Check if Agent Won
        if(Agent1.position == Prey1.position):
            flag=1
            print("Agent won")
            break

        #Update belief Probablity Table of Agent
        probability_distribution2(adjacency_list)
        print("Probability Table after prey move :", probability_table)
        print("Sum of prob table after prey moves :", sum(probability_table))

        #Move predator
        min_dist_agent=100000
        for i in adjacency_list[Predator1.position]:
            temp=djikstra(adjacency_list,i,Agent1.position)
            print("Predator1 temp ",temp," i ", i)
            if(temp<min_dist_agent):
                pred_next_pos=i
                min_dist_agent=temp
        Predator1.position=pred_next_pos

        #Check if predator won
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break

        # Take the max of the probability table for position of the new survey
        maxProbab = max(probability_table)
        maxPosition = probability_table.index(maxProbab) + 1 #Adding 1 to compensate for the indices starting from 0 rather than 1

        # Do a survey on this maxPosition
        #prey_pos = Prey1.position
        surveyNode = maxPosition
        print("Survey Node:", surveyNode)
        surveyRes = Survey(surveyNode, Prey1.position)

    if(flag==1):
        return 1
    if(flag==2):
        return 2
    if(flag==0):
        return 0

def Ag5(adjacency_list):
    count_pred_pos = 0
    index=list(range(1,no_of_nodes+1))
    Agent_pos = np.random.choice(index)
    index_pred=list(range(1,no_of_nodes+1))
    index_pred.remove(Agent_pos)
    predator_pos = np.random.choice(index_pred)
    prey_pos = np.random.choice(index_pred)

    Agent1 = Agent(Agent_pos,0,0,0)
    Predator1 = Predator(predator_pos,0,0)
    Prey1 = Prey(prey_pos,0,0)

    #Clearing probability table
    pred_probab_table.clear()
    #Inserting Raw probabilities
    for x in range(no_of_nodes):
        pred_probab_table.append(1/no_of_nodes)
    
    pred_probab_table[Agent1.position-1]=0

    for y in range(no_of_nodes):
        if(y != Agent1.position-1):
            pred_probab_table[y]=(1/(no_of_nodes-1))

    #Initial Survey
    surveyList = list(range(1,no_of_nodes+1))
    surveyList.remove(Agent_pos)
    surveyNode = Predator1.position

    surveyRes = Survey(surveyNode, Predator1.position)
    count=1
    flag=0
    loopCounter = 0
    while(True):
        loopCounter += 1
        if(count==threshold):                                         # VISH
            break
        count=count+1
        """
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break
        """
        if(surveyRes == 1):
            #Update the counter for Agent knowing the predator position
            count_pred_pos += 1 
            #Pred Found -> Update Probab table
            for i in range(len(pred_probab_table)):
                pred_probab_table[i] = 0
            pred_probab_table[surveyNode-1] = 1
            
            print("Probability Table after survey pred found :", pred_probab_table)
            print("Sum of prob table after survey pred found :", sum(pred_probab_table))

            
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                dist[i]=[temp_prey,temp_pred]
            
            sorted_list_dist = sorted(dist.items(), key=lambda x:x[1])
            dist=dict(sorted_list_dist) # Sorted dict of distances
            print("dist ",dist) 
            var=list(dist.keys())[0]  
            agent_next_pos=0                                                 
            for i in dist:
                if(dist[i][0] == dist[var][0] and dist[i][1] != 0):          
                    agent_next_pos=i
            
            if(agent_next_pos==0):
                print("HERE HERE HERE HERE HERE HERE")
                for i in dist:
                    if(dist[i][1] != 0):          
                        agent_next_pos=i

            if(agent_next_pos != 0):                                            
                Agent1.position=agent_next_pos


        else:
            #Pred Not Found -> Update Probab table
            print("Agent Position", Agent1.position)
            for i in range(len(pred_probab_table)):
                pred_probab_table[i] = 1/(no_of_nodes-2)
            pred_probab_table[surveyNode-1] = 0
            pred_probab_table[Agent1.position-1] = 0
            print("Probability Table after survey no Pred found :", pred_probab_table)
            print("Sum of prob table after survey no Pred found :", sum(pred_probab_table))

            # As all probabilities are equal -> Calculate farthest movement from Predator
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                dist[i] = temp_prey
            
            minDistance = 10000000
            minNbr = 0
            for key,val in dist.items():
                if (val < minDistance):
                    minNbr = key
                    minDistance = val
            
            # Move the agent

            Agent1.position = minNbr

        #Check if Agent Won /died                                 
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break
        if(Agent1.position == Prey1.position):
            flag=1
            print("Agent won")
            break
        

        # As agent has moved update the probability table with new information
        for x in range(len(pred_probab_table)):
            pred_probab_table[x] = 1/(no_of_nodes-2)
        pred_probab_table[Agent1.position - 1] = 0
        pred_probab_table[surveyNode - 1] = 0
        if (1 in pred_probab_table):
            count_pred_pos += 1
        print("Probability Table after survey no pred found and agent move:", pred_probab_table)
        print("Sum of prob table after survey no pred found and agent move:", sum(pred_probab_table))

        #Now moving the prey
        l_prey=[]
        for j in adjacency_list[Prey1.position]:
            l_prey.append(j)
        l_prey.append(Prey1.position)    
        prey_next_point=np.random.choice(l_prey)
        Prey1.position=prey_next_point

        #Check if Agent Won
        if(Agent1.position == Prey1.position):
            flag=1
            print("Agent won")
            break

        #Move predator
        pred_choice=np.random.choice([0,1],p=[0.6,0.4])
        if(pred_choice==1):
            min_dist_agent=100000
            for i in adjacency_list[Predator1.position]:
                temp=djikstra(adjacency_list,i,Agent1.position)
                print("Predator1 temp ",temp," i ", i)
                if(temp<min_dist_agent):
                    pred_next_pos=i
                    min_dist_agent=temp
            Predator1.position=pred_next_pos  
            #Update belief Probablity Table of Agent
            probab_pred_distribution(adjacency_list, Agent1.position)
            if ( 1 in pred_probab_table):
                  count_pred_pos +=1
        elif(pred_choice==0):
            Predator1.position=np.random.choice(adjacency_list[Predator1.position])
            prob_distracted(adjacency_list)  
            if ( 1 in pred_probab_table):
                  count_pred_pos +=1



        #Check if predator won
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break

        # Take the max of the probability table for position of the new survey
        maxProbab = max(pred_probab_table)
        #maxPosition = pred_probab_table.index(maxProbab) + 1 #Adding 1 to compensate for the indices starting from 0 rather than 1

        if (pred_probab_table[Predator1.position - 1] == maxProbab):
            count_pred_pos += 1

        tempL = []
        for f in range(len(pred_probab_table)):
            if(pred_probab_table[f] == maxProbab):
                tempL.append(f)
        
        maxPosition = np.random.choice(tempL) + 1
        
        # Do a survey on this maxPosition
        #prey_pos = Prey1.position
        surveyNode = maxPosition
        print("Survey Node:", surveyNode)
        surveyRes = Survey(surveyNode, Predator1.position)


    percentCalc = (count_pred_pos/loopCounter) * 100
    prob_count_ag5.append(percentCalc)
    if(flag==1):
        return 1
    if(flag==2):
        return 2
    if(flag==0):
        return 0         

def Agent6(adjacency_list):
    index=list(range(1,no_of_nodes+1))
    Agent_pos = np.random.choice(index)
    index_pred=list(range(1,no_of_nodes+1))
    index_pred.remove(Agent_pos)
    predator_pos = np.random.choice(index_pred)
    prey_pos = np.random.choice(index_pred)

    Agent1 = Agent(Agent_pos,0,0,0)
    Predator1 = Predator(predator_pos,0,0)
    Prey1 = Prey(prey_pos,0,0)

    #Clearing probability table
    pred_probab_table.clear()
    #Inserting Raw probabilities
    for x in range(no_of_nodes):
        pred_probab_table.append(1/no_of_nodes)
    
    pred_probab_table[Agent1.position-1]=0

    for y in range(no_of_nodes):
        if(y != Agent1.position-1):
            pred_probab_table[y]=(1/(no_of_nodes-1))

    #Initial Survey
    surveyList = list(range(1,no_of_nodes+1))
    surveyList.remove(Agent_pos)
    surveyNode = np.random.choice(surveyList)

    surveyRes = Survey(surveyNode, Predator1.position)
    count=1
    flag=0
    while(True):
        if(count==threshold):                                         # VISH
            break
        count=count+1
        """
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break
        """
        if(surveyRes == 1):
            #Prey Found -> Update Probab table
            for i in range(len(pred_probab_table)):
                pred_probab_table[i] = 0
            pred_probab_table[surveyNode-1] = 1
            
            print("Probability Table after survey pred found :", pred_probab_table)
            print("Sum of prob table after survey pred found :", sum(pred_probab_table))

            # Prey is at survey Node -> Agent has to move towards Survey Node
            dist_prey={}
            dist_pred={}
            dist={}
            prey_list=[]
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                if(temp_prey<5):
                    dist_prey[i]=[temp_prey]
                if(temp_pred<5):
                    dist_pred[i]=[temp_pred]
                if(temp_prey>=5 and temp_pred>=5):
                    dist[i]=[temp_prey,temp_pred]  
            if(len(dist_pred)>0):
                sorted_list_dist = sorted(dist_pred.items(), key=lambda x:x[1])
                dist_pred=dict(sorted_list_dist)
                print("dist_pred ",dist_pred)
                agent_next_pos=list(dist_pred.keys())[len(dist_pred)-1]
            elif(len(dist_prey)>0):
                sorted_list_dist = sorted(dist_prey.items(), key=lambda x:x[1])
                dist_prey=dict(sorted_list_dist)
                print("dist_prey ",dist_prey)
                agent_next_pos=list(dist_prey.keys())[0]    
            else:    
                sorted_list_dist = sorted(dist.items(), key=lambda x:x[1])
                dist=dict(sorted_list_dist) 
                print("dist ",dist) 
                var=list(dist.keys())[0]  
                agent_next_pos=0                                                 
                for i in dist:
                    if(dist[i][0] == dist[var][0] and dist[i][1] >2):
                        agent_next_pos=i
                
                maximus=0
                if(agent_next_pos==0):
                    print("HERE HERE HERE HERE HERE HERE")
                    for i in dist:
                        if(dist[i][1] != 0):          
                            if(i>maximus):
                                agent_next_pos=i
                                maximus=i
            
            if(agent_next_pos!=0):                                            
                Agent1.position=agent_next_pos
            print("Agent new position ",Agent1.position)


        else:
            #Pred Not Found -> Update Probab table
            print("Agent Position", Agent1.position)
            for i in range(len(pred_probab_table)):
                pred_probab_table[i] = 1/(no_of_nodes-2)
            pred_probab_table[surveyNode-1] = 0
            pred_probab_table[Agent1.position-1] = 0
            print("Probability Table after survey no Pred found :", pred_probab_table)
            print("Sum of prob table after survey no Pred found :", sum(pred_probab_table))

            # As all probabilities are equal -> Calculate farthest movement from Predator
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                dist[i] = temp_prey
            
            minDistance = 10000000
            minNbr = 0
            for key,val in dist.items():
                if (val < minDistance):
                    minNbr = key
                    minDistance = val
            
            # Move the agent

            Agent1.position = minNbr

        #Check if Agent Won /died                                 
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break
        if(Agent1.position == Prey1.position):
            flag=1
            print("Agent won")
            break
        

        # As agent has moved update the probability table with new information
        for x in range(len(pred_probab_table)):
            pred_probab_table[x] = 1/(no_of_nodes-2)
        pred_probab_table[Agent1.position - 1] = 0
        pred_probab_table[surveyNode - 1] = 0
        print("Probability Table after survey no pred found and agent move:", pred_probab_table)
        print("Sum of prob table after survey no pred found and agent move:", sum(pred_probab_table))

        #Now moving the prey
        l_prey=[]
        for j in adjacency_list[Prey1.position]:
            l_prey.append(j)
        l_prey.append(Prey1.position)    
        prey_next_point=np.random.choice(l_prey)
        Prey1.position=prey_next_point

        #Check if Agent Won
        if(Agent1.position == Prey1.position):
            flag=1
            print("Agent won")
            break

        #Move predator
        #VISH
        pred_choice=np.random.choice([0,1],p=[0.6,0.4])
        if(pred_choice==1):
            min_dist_agent=100000
            for i in adjacency_list[Predator1.position]:
                temp=djikstra(adjacency_list,i,Agent1.position)
                print("Predator1 temp ",temp," i ", i)
                if(temp<min_dist_agent):
                    pred_next_pos=i
                    min_dist_agent=temp
            Predator1.position=pred_next_pos  
            #Update belief Probablity Table of Agent
            probab_pred_distribution(adjacency_list, Agent1.position)
            print("Probability Table after pred move :", pred_probab_table)
            print("Len  = ", len(pred_probab_table))
            print("Sum of prob table after pred moves :", sum(pred_probab_table))   
            print("On Top")   
        elif(pred_choice==0):
            Predator1.position=np.random.choice(adjacency_list[Predator1.position])
            prob_distracted(adjacency_list)  
            print("Probability Table after pred move :", pred_probab_table)
            print("Sum of prob table after pred moves :", sum(pred_probab_table))   
            print("In Bottom")      
        #Predator1.position=pred_next_pos


        #Check if predator won
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break

        # Take the max of the probability table for position of the new survey
        maxProbab = max(pred_probab_table)
        maxPosition = pred_probab_table.index(maxProbab) + 1 #Adding 1 to compensate for the indices starting from 0 rather than 1

        # Do a survey on this maxPosition
        #prey_pos = Prey1.position
        surveyNode = maxPosition
        print("Survey Node:", surveyNode)
        surveyRes = Survey(surveyNode, Predator1.position)

    if(flag==1):
        return 1
    if(flag==2):
        return 2
    if(flag==0):
        return 0


def Ag7(adjacency_list):

    count_prey_pos = 0
    count_pred_pos = 0


    flag_preySurvey = 0 #Set when we know about prey position
    flag_predSurvey = 1 #Set when we know about pred position

    index=list(range(1,no_of_nodes+1))
    Agent_pos = np.random.choice(index)
    index_pred=list(range(1,no_of_nodes+1))
    index_pred.remove(Agent_pos)
    predator_pos = np.random.choice(index_pred)
    prey_pos = np.random.choice(index_pred)

    Agent1 = Agent(Agent_pos,0,0,0)
    Predator1 = Predator(predator_pos,0,0)
    Prey1 = Prey(prey_pos,0,0)

    #Clearing probability table
    pred_probab_table.clear()
    probability_table.clear()

    #Initially agent knows the predator pos -> Update Probab Table for Pred
    for i in range(no_of_nodes):
        pred_probab_table.append(0)
    pred_probab_table[Predator1.position - 1] = 1
    # count_pred_pos += 1

    #Agent doesn't know Prey's position -> Update Probab Table for Prey
    for i in range(no_of_nodes):
        probability_table.append(1/(no_of_nodes - 1))
    probability_table[Agent1.position - 1] = 0

    #Initial Survey
    surveyList = list(range(1,no_of_nodes+1))
    surveyList.remove(Agent_pos)
    surveyNode = np.random.choice(surveyList)

    surveyRes = Survey(surveyNode, Prey1.position)

    count = 0
    flag = 0
    loopCounter = 0
    while(True):
        loopCounter += 1
        if(count==threshold):                                      # VISH
            break
        count=count+1
        
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break

        if(flag_predSurvey == 1 and surveyRes == 1): #Agent knows PredPos and found prey in survey
            
            #Prey Found -> Update Probab table
            count_prey_pos += 1
            for i in range(len(probability_table)):
                probability_table[i] = 0
            probability_table[surveyNode-1] = 1
            
            #print("Probability Table after survey prey found :", probability_table)
            print("Sum of prob table after survey prey found :", sum(probability_table))

            
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                dist[i]=[temp_prey,temp_pred]
            
            sorted_list_dist = sorted(dist.items(), key=lambda x:x[1])
            dist=dict(sorted_list_dist) # Sorted dict of distances
            print("dist ",dist) 
            var=list(dist.keys())[0]  
            agent_next_pos=0                                                 # Today
            for i in dist:
                if(dist[i][0] == dist[var][0] and dist[i][1] != 0):          # Today
                    agent_next_pos=i
            # TODAY
            if(agent_next_pos==0):
                print("HERE HERE HERE HERE HERE HERE")
                for i in dist:
                    if(dist[i][1] != 0):          
                        agent_next_pos=i
            # TODAY
            if(agent_next_pos != 0):                                            # TODAY
                Agent1.position=agent_next_pos

            # As agent has moved update the probability table with new information
            for x in range(len(probability_table)):
                probability_table[x] = 1/(no_of_nodes-2)
            probability_table[Agent1.position - 1] = 0
            probability_table[surveyNode - 1] = 0
            #print("Probability Table after survey no prey found and agent move:", probability_table)
            print("Sum of prob table after survey no prey found and agent move:", sum(probability_table))

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            if(1 in probability_table):
                count_prey_pos += 1
            #print("Probability Table after prey move :", probability_table)
            print("Sum of prob table after prey moves :", sum(probability_table))


            #VISH
            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
                if( 1 in pred_probab_table):
                    count_pred_pos += 1
                #print("Probability Table after pred move :", pred_probab_table)
                #print("Len  = ", len(pred_probab_table))
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("On Top")   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list)  
                if( 1 in pred_probab_table):
                    count_pred_pos += 1
                #print("Probability Table after pred move :", pred_probab_table)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("In Bottom")      
            #Predator1.position=pred_next_pos


            #Check if predator won
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        elif(flag_predSurvey == 1 and surveyRes == 0): # Agent knows PredPos and doesnt know PreyPos
            #Prey Not Found -> Update Probab table
            
            print("Agent Position", Agent1.position)
            for i in range(len(probability_table)):
                probability_table[i] = 1/(no_of_nodes-2)
            probability_table[surveyNode-1] = 0
            probability_table[Agent1.position-1] = 0
            #print("Probability Table after survey no prey found :", probability_table)
            print("Sum of prob table after survey no prey found :", sum(probability_table))

            # As all probabilities are equal -> Calculate farthest movement from Predator
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                dist[i] = temp_pred
            
            maxDistance = -1
            maxNbr = 0
            for key,val in dist.items():
                if (val > maxDistance):
                    maxNbr = key
                    maxDistance = val
            
            # Move the agent
            Agent1.position = maxNbr

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

            # As agent has moved update the probability table with new information
            for x in range(len(probability_table)):
                probability_table[x] = 1/(no_of_nodes-2)
            probability_table[Agent1.position - 1] = 0
            probability_table[surveyNode - 1] = 0
            #print("Probability Table after survey no prey found and agent move:", probability_table)
            print("Sum of prob table after survey no prey found and agent move:", sum(probability_table))

            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            if( 1 in probability_table):
                count_prey_pos +=1
            #print("Probability Table after prey move :", probability_table)
            print("Sum of prob table after prey moves :", sum(probability_table))

            #VISH
            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
                if(1 in pred_probab_table):
                    count_pred_pos += 1
                #print("Probability Table after pred move :", pred_probab_table)
                #print("Len  = ", len(pred_probab_table))
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("On Top")   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list) 
                if(1 in pred_probab_table):
                    count_pred_pos += 1 
                #print("Probability Table after pred move :", pred_probab_table)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("In Bottom")      
            #Predator1.position=pred_next_pos


            #Check if predator won
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        elif(flag_preySurvey == 1 and surveyRes == 1): # Agent knows prey pos and doesnt know pred pos but predFound in survey
            count_pred_pos += 1
            #Pred Found -> Update Probab table
            for i in range(len(pred_probab_table)):
                pred_probab_table[i] = 0
            pred_probab_table[surveyNode-1] = 1
            
            #print("Probability Table after survey pred found :", pred_probab_table)
            print("Sum of prob table after survey pred found :", sum(pred_probab_table))

            
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                dist[i]=[temp_prey,temp_pred]

            sorted_list_dist = sorted(dist.items(), key=lambda x:x[1])
            dist=dict(sorted_list_dist) # Sorted dict of distances
            print("dist ",dist) 
            var=list(dist.keys())[0]  
            agent_next_pos=0                                                 
            for i in dist:
                if(dist[i][0] == dist[var][0] and dist[i][1] != 0):          
                    agent_next_pos=i
            
            if(agent_next_pos==0):
                print("HERE HERE HERE HERE HERE HERE")
                for i in dist:
                    if(dist[i][1] != 0):          
                        agent_next_pos=i

            if(agent_next_pos != 0):                                            
                Agent1.position=agent_next_pos
            
            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break
            
            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            if( 1 in probability_table):
                count_prey_pos += 1
            #print("Probability Table after prey move :", probability_table)
            print("Sum of prob table after prey moves :", sum(probability_table))

            #VISH
            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
                if ( 1 in pred_probab_table):
                    count_pred_pos +=1 
                #print("Probability Table after pred move :", pred_probab_table)
                #print("Len  = ", len(pred_probab_table))
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("On Top")   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list)
                if ( 1 in pred_probab_table):
                    count_pred_pos +=1   
                #print("Probability Table after pred move :", pred_probab_table)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("In Bottom")

            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        elif(flag_preySurvey == 1 and surveyRes == 0):# Agent knows prey pos and doesnt know pred pos but pred not in survey
            
            #Pred Not Found -> Update Probab table
            print("Agent Position", Agent1.position)
            for i in range(len(pred_probab_table)):
                pred_probab_table[i] = 1/(no_of_nodes-2)
            pred_probab_table[surveyNode-1] = 0
            pred_probab_table[Agent1.position-1] = 0
            #print("Probability Table after survey no Pred found :", pred_probab_table)
            print("Sum of prob table after survey no Pred found :", sum(pred_probab_table))

            # As all probabilities are equal -> Calculate farthest movement from Predator
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                dist[i] = temp_prey
            
            minDistance = 10000000
            minNbr = 0
            for key,val in dist.items():
                if (val < minDistance):
                    minNbr = key
                    minDistance = val
            
            # Move the agent

            Agent1.position = minNbr

            #Check if Agent Won /died                                 
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            if( 1 in probability_table):
                count_prey_pos += 1
            #print("Probability Table after prey move :", probability_table)
            print("Sum of prob table after prey moves :", sum(probability_table))

            # if(Agent1.position == Prey1.position):
            #     flag=1
            #     print("Agent won")
            #     break

            #VISH
            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
                if ( 1 in pred_probab_table):
                    count_pred_pos +=1
            #    print("Probability Table after pred move :", pred_probab_table)
            #    print("Len  = ", len(pred_probab_table))
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
            #    print("On Top")   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list)
                if ( 1 in pred_probab_table):
                    count_pred_pos +=1  
            #    print("Probability Table after pred move :", pred_probab_table)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
            #    print("In Bottom")

            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        # Checking for next survey
        if( 1 in pred_probab_table ): # Agent knows where pred is -> Survey for prey in next
            flag_preySurvey = 0
            flag_predSurvey = 1
            count_pred_pos += 1

            # Take the max of the probability table for position of the new survey
            maxProbab = max(probability_table)
            maxPosition = probability_table.index(maxProbab) + 1 #Adding 1 to compensate for the indices starting from 0 rather than 1

            # Do a survey on this maxPosition
            #prey_pos = Prey1.position
            surveyNode = maxPosition
            print("Survey Node:", surveyNode)
            surveyRes = Survey(surveyNode, Prey1.position)
        
        else: # Agent doesn't know where pred is -> Survey for pred

            flag_predSurvey = 0
            flag_preySurvey = 1
            
            maxProbab = max(pred_probab_table)
            maxPosition = pred_probab_table.index(maxProbab) + 1 #Adding 1 to compensate for the indices starting from 0 rather than 1
            if(maxProbab == 1):
                count_prey_pos +=1 
            
            surveyNode = maxPosition
            print("Survey Node:", surveyNode)
            surveyRes = Survey(surveyNode, Predator1.position)


    percentCalc = (count_prey_pos/loopCounter) * 100
    prob_count_prey_ag7.append(percentCalc)
    percentCalc = (count_pred_pos/loopCounter) * 100
    prob_count_pred_ag7.append(percentCalc)
    if(flag==1):
        return 1
    if(flag==2):
        return 2
    if(flag==0):
        return 0

def Agent8(adjacency_list):
    flag_preySurvey = 0 #Set when we know about prey position
    flag_predSurvey = 1 #Set when we know about pred position

    index=list(range(1,no_of_nodes+1))
    Agent_pos = np.random.choice(index)
    index_pred=list(range(1,no_of_nodes+1))
    index_pred.remove(Agent_pos)
    predator_pos = np.random.choice(index_pred)
    prey_pos = np.random.choice(index_pred)

    Agent1 = Agent(Agent_pos,0,0,0)
    Predator1 = Predator(predator_pos,0,0)
    Prey1 = Prey(prey_pos,0,0)

    #Clearing probability table
    pred_probab_table.clear()
    probability_table.clear()

    #Initially agent knows the predator pos -> Update Probab Table for Pred
    for i in range(no_of_nodes):
        pred_probab_table.append(0)
    pred_probab_table[Predator1.position - 1] = 1

    #Agent doesn't know Prey's position -> Update Probab Table for Prey
    for i in range(no_of_nodes):
        probability_table.append(1/(no_of_nodes - 1))
    probability_table[Agent1.position - 1] = 0

    #Initial Survey
    surveyList = list(range(1,no_of_nodes+1))
    surveyList.remove(Agent_pos)
    surveyNode = np.random.choice(surveyList)

    surveyRes = Survey(surveyNode, Prey1.position)

    count = 0
    flag = 0
    while(True):

        if(count==threshold):                                      
            break
        count=count+1
        
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break
        

        if(flag_predSurvey == 1 and surveyRes == 1): 

            #Prey Found -> Update Probab table
            for i in range(len(probability_table)):
                probability_table[i] = 0
            probability_table[surveyNode-1] = 1
            print("Sum of prob table after survey prey found :", sum(probability_table))

            # Vish Today
            dist_prey={}
            dist_pred={}
            dist={}
            prey_list=[]
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                if(temp_prey<5):
                    dist_prey[i]=[temp_prey]
                if(temp_pred<5):
                    dist_pred[i]=[temp_pred]
                if(temp_prey>=5 and temp_pred>=5):
                    dist[i]=[temp_prey,temp_pred]  
            if(len(dist_pred)>0):
                sorted_list_dist = sorted(dist_pred.items(), key=lambda x:x[1])
                dist_pred=dict(sorted_list_dist)
                print("dist_pred ",dist_pred)
                agent_next_pos=list(dist_pred.keys())[len(dist_pred)-1]
            elif(len(dist_prey)>0):
                sorted_list_dist = sorted(dist_prey.items(), key=lambda x:x[1])
                dist_prey=dict(sorted_list_dist)
                print("dist_prey ",dist_prey)
                agent_next_pos=list(dist_prey.keys())[0]    
            else:    
                sorted_list_dist = sorted(dist.items(), key=lambda x:x[1])
                dist=dict(sorted_list_dist) 
                print("dist ",dist) 
                var=list(dist.keys())[0]  
                agent_next_pos=0                                                 
                for i in dist:
                    if(dist[i][0] == dist[var][0] and dist[i][1] >2):
                        agent_next_pos=i
                
                maximus=0
                if(agent_next_pos==0):
                    print("HERE HERE HERE HERE HERE HERE")
                    for i in dist:
                        if(dist[i][1] != 0):          
                            if(i>maximus):
                                agent_next_pos=i
                                maximus=i
            # Vish Today
            if(agent_next_pos != 0):                                            
                Agent1.position=agent_next_pos

            # As agent has moved update the probability table with new information
            for x in range(len(probability_table)):
                probability_table[x] = 1/(no_of_nodes-2)
            probability_table[Agent1.position - 1] = 0
            probability_table[surveyNode - 1] = 0
            #print("Probability Table after survey no prey found and agent move:", probability_table)
            print("Sum of prob table after survey no prey found and agent move:", sum(probability_table))

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            print("Sum of prob table after prey moves :", sum(probability_table))

            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list)  
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
            #Predator1.position=pred_next_pos


            #Check if predator won
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        elif(flag_predSurvey == 1 and surveyRes == 0): # Agent knows PredPos and doesnt know PreyPos
            #Prey Not Found -> Update Probab table
            print("Agent Position", Agent1.position)
            for i in range(len(probability_table)):
                probability_table[i] = 1/(no_of_nodes-2)
            probability_table[surveyNode-1] = 0
            probability_table[Agent1.position-1] = 0
            print("Sum of prob table after survey no prey found :", sum(probability_table))

            # As all probabilities are equal -> Calculate farthest movement from Predator
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                dist[i] = temp_pred
            
            maxDistance = -1
            maxNbr = 0
            for key,val in dist.items():
                if (val > maxDistance):
                    maxNbr = key
                    maxDistance = val
            
            # Move the agent
            Agent1.position = maxNbr

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

            # As agent has moved update the probability table with new information
            for x in range(len(probability_table)):
                probability_table[x] = 1/(no_of_nodes-2)
            probability_table[Agent1.position - 1] = 0
            probability_table[surveyNode - 1] = 0
            print("Sum of prob table after survey no prey found and agent move:", sum(probability_table))

            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            print("Sum of prob table after prey moves :", sum(probability_table))

            #VISH
            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("On Top")   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list)  
                print("Sum of prob table after pred moves :", sum(pred_probab_table)) 
            #Predator1.position=pred_next_pos


            #Check if predator won
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        elif(flag_preySurvey == 1 and surveyRes == 1): # Agent knows prey pos and doesnt know pred pos but predFound in survey

            #Pred Found -> Update Probab table
            for i in range(len(pred_probab_table)):
                pred_probab_table[i] = 0
            pred_probab_table[surveyNode-1] = 1
            
            #print("Probability Table after survey pred found :", pred_probab_table)
            print("Sum of prob table after survey pred found :", sum(pred_probab_table))

            # Vish Today
            dist_prey={}
            dist_pred={}
            dist={}
            prey_list=[]
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                if(temp_prey<5):
                    dist_prey[i]=[temp_prey]
                if(temp_pred<5):
                    dist_pred[i]=[temp_pred]
                if(temp_prey>=5 and temp_pred>=5):
                    dist[i]=[temp_prey,temp_pred]  
            if(len(dist_pred)>0):
                sorted_list_dist = sorted(dist_pred.items(), key=lambda x:x[1])
                dist_pred=dict(sorted_list_dist)
                print("dist_pred ",dist_pred)
                agent_next_pos=list(dist_pred.keys())[len(dist_pred)-1]
            elif(len(dist_prey)>0):
                sorted_list_dist = sorted(dist_prey.items(), key=lambda x:x[1])
                dist_prey=dict(sorted_list_dist)
                print("dist_prey ",dist_prey)
                agent_next_pos=list(dist_prey.keys())[0]    
            else:    
                sorted_list_dist = sorted(dist.items(), key=lambda x:x[1])
                dist=dict(sorted_list_dist) 
                print("dist ",dist) 
                var=list(dist.keys())[0]  
                agent_next_pos=0                                                 
                for i in dist:
                    if(dist[i][0] == dist[var][0] and dist[i][1] >2):
                        agent_next_pos=i
                
                maximus=0
                if(agent_next_pos==0):
                    print("HERE HERE HERE HERE HERE HERE")
                    for i in dist:
                        if(dist[i][1] != 0):          
                            if(i>maximus):
                                agent_next_pos=i
                                maximus=i
            # Vish Today
            if(agent_next_pos != 0):                                            
                Agent1.position=agent_next_pos
            
            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break
            
            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            #print("Probability Table after prey move :", probability_table)
            print("Sum of prob table after prey moves :", sum(probability_table))

            #VISH
            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
                #print("Probability Table after pred move :", pred_probab_table)
                #print("Len  = ", len(pred_probab_table))
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("On Top")   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list)  
                #print("Probability Table after pred move :", pred_probab_table)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("In Bottom")

            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        elif(flag_preySurvey == 1 and surveyRes == 0):# Agent knows prey pos and doesnt know pred pos but pred not in survey

            #Pred Not Found -> Update Probab table
            print("Agent Position", Agent1.position)
            for i in range(len(pred_probab_table)):
                pred_probab_table[i] = 1/(no_of_nodes-2)
            pred_probab_table[surveyNode-1] = 0
            pred_probab_table[Agent1.position-1] = 0
            #print("Probability Table after survey no Pred found :", pred_probab_table)
            print("Sum of prob table after survey no Pred found :", sum(pred_probab_table))

            # As all probabilities are equal -> Calculate farthest movement from Predator
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                dist[i] = temp_prey
            
            minDistance = 10000000
            minNbr = 0
            for key,val in dist.items():
                if (val < minDistance):
                    minNbr = key
                    minDistance = val
            
            # Move the agent

            Agent1.position = minNbr

            #Check if Agent Won /died                                 
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            #print("Probability Table after prey move :", probability_table)
            print("Sum of prob table after prey moves :", sum(probability_table))

            # if(Agent1.position == Prey1.position):
            #     flag=1
            #     print("Agent won")
            #     break

            #VISH
            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
            #    print("Probability Table after pred move :", pred_probab_table)
            #    print("Len  = ", len(pred_probab_table))
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
            #    print("On Top")   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list)  
            #    print("Probability Table after pred move :", pred_probab_table)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
            #    print("In Bottom")

            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        # Checking for next survey
        if( 1 in pred_probab_table ): # Agent knows where pred is -> Survey for prey in next
            flag_preySurvey = 0
            flag_predSurvey = 1

            # Take the max of the probability table for position of the new survey
            maxProbab = max(probability_table)
            maxPosition = probability_table.index(maxProbab) + 1 #Adding 1 to compensate for the indices starting from 0 rather than 1

            # Do a survey on this maxPosition
            #prey_pos = Prey1.position
            surveyNode = maxPosition
            print("Survey Node:", surveyNode)
            surveyRes = Survey(surveyNode, Prey1.position)
        
        else: # Agent doesn't know where pred is -> Survey for pred

            flag_predSurvey = 0
            flag_preySurvey = 1

             # Take the max of the probability table for position of the new survey
            maxProbab = max(pred_probab_table)
            maxPosition = pred_probab_table.index(maxProbab) + 1 #Adding 1 to compensate for the indices starting from 0 rather than 1

            # Do a survey on this maxPosition
            #prey_pos = Prey1.position
            surveyNode = maxPosition
            print("Survey Node:", surveyNode)
            surveyRes = Survey(surveyNode, Predator1.position)

    if(flag==1):
        return 1
    if(flag==2):
        return 2
    if(flag==0):
        return 0

def fault_survey(index,location):

    fault_surv = np.random.choice([0,1],p=[0.9,0.1])

    if(index == location and fault_survey == 0):
        return 1
    else:
        return 0


def faulty_drone_ag7(adjacency_list):

    count_prey_pos = 0
    count_pred_pos = 0


    flag_preySurvey = 0 #Set when we know about prey position
    flag_predSurvey = 1 #Set when we know about pred position

    index=list(range(1,no_of_nodes+1))
    Agent_pos = np.random.choice(index)
    index_pred=list(range(1,no_of_nodes+1))
    index_pred.remove(Agent_pos)
    predator_pos = np.random.choice(index_pred)
    prey_pos = np.random.choice(index_pred)

    Agent1 = Agent(Agent_pos,0,0,0)
    Predator1 = Predator(predator_pos,0,0)
    Prey1 = Prey(prey_pos,0,0)

    #Clearing probability table
    pred_probab_table.clear()
    probability_table.clear()

    #Initially agent knows the predator pos -> Update Probab Table for Pred
    for i in range(no_of_nodes):
        pred_probab_table.append(0)
    pred_probab_table[Predator1.position - 1] = 1
    # count_pred_pos += 1

    #Agent doesn't know Prey's position -> Update Probab Table for Prey
    for i in range(no_of_nodes):
        probability_table.append(1/(no_of_nodes - 1))
    probability_table[Agent1.position - 1] = 0

    #Initial Survey
    surveyList = list(range(1,no_of_nodes+1))
    surveyList.remove(Agent_pos)
    surveyNode = np.random.choice(surveyList)

    surveyRes = fault_survey(surveyNode, Prey1.position)

    count = 0
    flag = 0
    loopCounter = 0
    while(True):
        loopCounter += 1
        if(count==threshold):                                      # VISH
            break
        count=count+1
        
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break

        if(flag_predSurvey == 1 and surveyRes == 1): #Agent knows PredPos and found prey in survey
            
            #Prey Found -> Update Probab table
            count_prey_pos += 1
            for i in range(len(probability_table)):
                probability_table[i] = 0
            probability_table[surveyNode-1] = 1
            
            #print("Probability Table after survey prey found :", probability_table)
            print("Sum of prob table after survey prey found :", sum(probability_table))

            
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                dist[i]=[temp_prey,temp_pred]
            
            sorted_list_dist = sorted(dist.items(), key=lambda x:x[1])
            dist=dict(sorted_list_dist) # Sorted dict of distances
            print("dist ",dist) 
            var=list(dist.keys())[0]  
            agent_next_pos=0                                                 # Today
            for i in dist:
                if(dist[i][0] == dist[var][0] and dist[i][1] != 0):          # Today
                    agent_next_pos=i
            # TODAY
            if(agent_next_pos==0):
                print("HERE HERE HERE HERE HERE HERE")
                for i in dist:
                    if(dist[i][1] != 0):          
                        agent_next_pos=i
            # TODAY
            if(agent_next_pos != 0):                                            # TODAY
                Agent1.position=agent_next_pos

            # As agent has moved update the probability table with new information
            for x in range(len(probability_table)):
                probability_table[x] = 1/(no_of_nodes-2)
            probability_table[Agent1.position - 1] = 0
            probability_table[surveyNode - 1] = 0
            #print("Probability Table after survey no prey found and agent move:", probability_table)
            print("Sum of prob table after survey no prey found and agent move:", sum(probability_table))

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            if(1 in probability_table):
                count_prey_pos += 1
            #print("Probability Table after prey move :", probability_table)
            print("Sum of prob table after prey moves :", sum(probability_table))


            #VISH
            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
                if( 1 in pred_probab_table):
                    count_pred_pos += 1
                #print("Probability Table after pred move :", pred_probab_table)
                #print("Len  = ", len(pred_probab_table))
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("On Top")   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list)  
                if( 1 in pred_probab_table):
                    count_pred_pos += 1
                #print("Probability Table after pred move :", pred_probab_table)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("In Bottom")      
            #Predator1.position=pred_next_pos


            #Check if predator won
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        elif(flag_predSurvey == 1 and surveyRes == 0): # Agent knows PredPos and doesnt know PreyPos
            #Prey Not Found -> Update Probab table
            
            print("Agent Position", Agent1.position)
            for i in range(len(probability_table)):
                probability_table[i] = 1/(no_of_nodes-2)
            probability_table[surveyNode-1] = 0
            probability_table[Agent1.position-1] = 0
            #print("Probability Table after survey no prey found :", probability_table)
            print("Sum of prob table after survey no prey found :", sum(probability_table))

            # As all probabilities are equal -> Calculate farthest movement from Predator
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                dist[i] = temp_pred
            
            maxDistance = -1
            maxNbr = 0
            for key,val in dist.items():
                if (val > maxDistance):
                    maxNbr = key
                    maxDistance = val
            
            # Move the agent
            Agent1.position = maxNbr

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

            # As agent has moved update the probability table with new information
            for x in range(len(probability_table)):
                probability_table[x] = 1/(no_of_nodes-2)
            probability_table[Agent1.position - 1] = 0
            probability_table[surveyNode - 1] = 0
            #print("Probability Table after survey no prey found and agent move:", probability_table)
            print("Sum of prob table after survey no prey found and agent move:", sum(probability_table))

            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            if( 1 in probability_table):
                count_prey_pos +=1
            #print("Probability Table after prey move :", probability_table)
            print("Sum of prob table after prey moves :", sum(probability_table))

            #VISH
            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
                if(1 in pred_probab_table):
                    count_pred_pos += 1
                #print("Probability Table after pred move :", pred_probab_table)
                #print("Len  = ", len(pred_probab_table))
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("On Top")   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list) 
                if(1 in pred_probab_table):
                    count_pred_pos += 1 
                #print("Probability Table after pred move :", pred_probab_table)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("In Bottom")      
            #Predator1.position=pred_next_pos


            #Check if predator won
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        elif(flag_preySurvey == 1 and surveyRes == 1): # Agent knows prey pos and doesnt know pred pos but predFound in survey
            count_pred_pos += 1
            #Pred Found -> Update Probab table
            for i in range(len(pred_probab_table)):
                pred_probab_table[i] = 0
            pred_probab_table[surveyNode-1] = 1
            
            #print("Probability Table after survey pred found :", pred_probab_table)
            print("Sum of prob table after survey pred found :", sum(pred_probab_table))

            
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                dist[i]=[temp_prey,temp_pred]

            sorted_list_dist = sorted(dist.items(), key=lambda x:x[1])
            dist=dict(sorted_list_dist) # Sorted dict of distances
            print("dist ",dist) 
            var=list(dist.keys())[0]  
            agent_next_pos=0                                                 
            for i in dist:
                if(dist[i][0] == dist[var][0] and dist[i][1] != 0):          
                    agent_next_pos=i
            
            if(agent_next_pos==0):
                print("HERE HERE HERE HERE HERE HERE")
                for i in dist:
                    if(dist[i][1] != 0):          
                        agent_next_pos=i

            if(agent_next_pos != 0):                                            
                Agent1.position=agent_next_pos
            
            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break
            
            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            if( 1 in probability_table):
                count_prey_pos += 1
            #print("Probability Table after prey move :", probability_table)
            print("Sum of prob table after prey moves :", sum(probability_table))

            #VISH
            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
                if ( 1 in pred_probab_table):
                    count_pred_pos +=1 
                #print("Probability Table after pred move :", pred_probab_table)
                #print("Len  = ", len(pred_probab_table))
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("On Top")   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list)
                if ( 1 in pred_probab_table):
                    count_pred_pos +=1   
                #print("Probability Table after pred move :", pred_probab_table)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("In Bottom")

            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        elif(flag_preySurvey == 1 and surveyRes == 0):# Agent knows prey pos and doesnt know pred pos but pred not in survey
            
            #Pred Not Found -> Update Probab table
            print("Agent Position", Agent1.position)
            for i in range(len(pred_probab_table)):
                pred_probab_table[i] = 1/(no_of_nodes-2)
            pred_probab_table[surveyNode-1] = 0
            pred_probab_table[Agent1.position-1] = 0
            #print("Probability Table after survey no Pred found :", pred_probab_table)
            print("Sum of prob table after survey no Pred found :", sum(pred_probab_table))

            # As all probabilities are equal -> Calculate farthest movement from Predator
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                dist[i] = temp_prey
            
            minDistance = 10000000
            minNbr = 0
            for key,val in dist.items():
                if (val < minDistance):
                    minNbr = key
                    minDistance = val
            
            # Move the agent

            Agent1.position = minNbr

            #Check if Agent Won /died                                 
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            if( 1 in probability_table):
                count_prey_pos += 1
            #print("Probability Table after prey move :", probability_table)
            print("Sum of prob table after prey moves :", sum(probability_table))

            # if(Agent1.position == Prey1.position):
            #     flag=1
            #     print("Agent won")
            #     break

            #VISH
            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
                if ( 1 in pred_probab_table):
                    count_pred_pos +=1
            #    print("Probability Table after pred move :", pred_probab_table)
            #    print("Len  = ", len(pred_probab_table))
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
            #    print("On Top")   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list)
                if ( 1 in pred_probab_table):
                    count_pred_pos +=1  
            #    print("Probability Table after pred move :", pred_probab_table)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
            #    print("In Bottom")

            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break
        
            



        # Checking for next survey
        if( 1 in pred_probab_table ): # Agent knows where pred is -> Survey for prey in next
            flag_preySurvey = 0
            flag_predSurvey = 1
            count_pred_pos += 1

            # Take the max of the probability table for position of the new survey
            maxProbab = max(probability_table)
            maxPosition = probability_table.index(maxProbab) + 1 #Adding 1 to compensate for the indices starting from 0 rather than 1

            # Do a survey on this maxPosition
            #prey_pos = Prey1.position
            surveyNode = maxPosition
            print("Survey Node:", surveyNode)
            surveyRes = fault_survey(surveyNode, Prey1.position)
        
        else: # Agent doesn't know where pred is -> Survey for pred

            flag_predSurvey = 0
            flag_preySurvey = 1
            # if(1 in probability_table): # Agent knows where prey is exactly
            #     flag_preySurvey = 1
            # else:
            #     flag_preySurvey = 0


             # Take the max of the probability table for position of the new survey
            maxProbab = max(pred_probab_table)
            maxPosition = pred_probab_table.index(maxProbab) + 1 #Adding 1 to compensate for the indices starting from 0 rather than 1
            if(maxProbab == 1):
                count_prey_pos +=1 
            # Do a survey on this maxPosition
            #prey_pos = Prey1.position
            surveyNode = maxPosition
            print("Survey Node:", surveyNode)
            surveyRes = fault_survey(surveyNode, Predator1.position)


    percentCalc = (count_prey_pos/loopCounter) * 100
    
    prob_count_prey_fault_drone_ag7.append(percentCalc)
    percentCalc = (count_pred_pos/loopCounter) * 100
    prob_count_pred_fault_drone_ag7.append(percentCalc)
    if(flag==1):
        return 1
    if(flag==2):
        return 2
    if(flag==0):
        return 0


def faulty_drone_agent8(adjacency_list):
    flag_preySurvey = 0 #Set when we know about prey position
    flag_predSurvey = 1 #Set when we know about pred position

    index=list(range(1,no_of_nodes+1))
    Agent_pos = np.random.choice(index)
    index_pred=list(range(1,no_of_nodes+1))
    index_pred.remove(Agent_pos)
    predator_pos = np.random.choice(index_pred)
    prey_pos = np.random.choice(index_pred)

    Agent1 = Agent(Agent_pos,0,0,0)
    Predator1 = Predator(predator_pos,0,0)
    Prey1 = Prey(prey_pos,0,0)

    #Clearing probability table
    pred_probab_table.clear()
    probability_table.clear()

    #Initially agent knows the predator pos -> Update Probab Table for Pred
    for i in range(no_of_nodes):
        pred_probab_table.append(0)
    pred_probab_table[Predator1.position - 1] = 1

    #Agent doesn't know Prey's position -> Update Probab Table for Prey
    for i in range(no_of_nodes):
        probability_table.append(1/(no_of_nodes - 1))
    probability_table[Agent1.position - 1] = 0

    #Initial Survey
    surveyList = list(range(1,no_of_nodes+1))
    surveyList.remove(Agent_pos)
    surveyNode = np.random.choice(surveyList)

    surveyRes = fault_survey(surveyNode, Prey1.position)

    count = 0
    flag = 0
    loopCounter = 0
    count_prey_pos = 0
    count_pred_pos = 0
    while(True):

        loopCounter +=1

        if(count==threshold):                                      
            break
        count=count+1
        
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator Won")
            break
        

        if(flag_predSurvey == 1 and surveyRes == 1): 

            #Prey Found -> Update Probab table
            for i in range(len(probability_table)):
                probability_table[i] = 0
            probability_table[surveyNode-1] = 1
            print("Sum of prob table after survey prey found :", sum(probability_table))

            # Vish Today
            dist_prey={}
            dist_pred={}
            dist={}
            prey_list=[]
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                if(temp_prey<5):
                    dist_prey[i]=[temp_prey]
                if(temp_pred<5):
                    dist_pred[i]=[temp_pred]
                if(temp_prey>=5 and temp_pred>=5):
                    dist[i]=[temp_prey,temp_pred]  
            if(len(dist_pred)>0):
                sorted_list_dist = sorted(dist_pred.items(), key=lambda x:x[1])
                dist_pred=dict(sorted_list_dist)
                print("dist_pred ",dist_pred)
                agent_next_pos=list(dist_pred.keys())[len(dist_pred)-1]
            elif(len(dist_prey)>0):
                sorted_list_dist = sorted(dist_prey.items(), key=lambda x:x[1])
                dist_prey=dict(sorted_list_dist)
                print("dist_prey ",dist_prey)
                agent_next_pos=list(dist_prey.keys())[0]    
            else:    
                sorted_list_dist = sorted(dist.items(), key=lambda x:x[1])
                dist=dict(sorted_list_dist) 
                print("dist ",dist) 
                var=list(dist.keys())[0]  
                agent_next_pos=0                                                 
                for i in dist:
                    if(dist[i][0] == dist[var][0] and dist[i][1] >2):
                        agent_next_pos=i
                
                maximus=0
                if(agent_next_pos==0):
                    print("HERE HERE HERE HERE HERE HERE")
                    for i in dist:
                        if(dist[i][1] != 0):          
                            if(i>maximus):
                                agent_next_pos=i
                                maximus=i
            # Vish Today
            if(agent_next_pos != 0):                                            
                Agent1.position=agent_next_pos

            # As agent has moved update the probability table with new information
            for x in range(len(probability_table)):
                probability_table[x] = 1/(no_of_nodes-2)
            probability_table[Agent1.position - 1] = 0
            probability_table[surveyNode - 1] = 0
            #print("Probability Table after survey no prey found and agent move:", probability_table)
            print("Sum of prob table after survey no prey found and agent move:", sum(probability_table))

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            print("Sum of prob table after prey moves :", sum(probability_table))

            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list)  
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
            #Predator1.position=pred_next_pos


            #Check if predator won
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        elif(flag_predSurvey == 1 and surveyRes == 0): # Agent knows PredPos and doesnt know PreyPos
            #Prey Not Found -> Update Probab table
            print("Agent Position", Agent1.position)
            for i in range(len(probability_table)):
                probability_table[i] = 1/(no_of_nodes-2)
            probability_table[surveyNode-1] = 0
            probability_table[Agent1.position-1] = 0
            print("Sum of prob table after survey no prey found :", sum(probability_table))

            # As all probabilities are equal -> Calculate farthest movement from Predator
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                dist[i] = temp_pred
            
            maxDistance = -1
            maxNbr = 0
            for key,val in dist.items():
                if (val > maxDistance):
                    maxNbr = key
                    maxDistance = val
            
            # Move the agent
            Agent1.position = maxNbr

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

            # As agent has moved update the probability table with new information
            for x in range(len(probability_table)):
                probability_table[x] = 1/(no_of_nodes-2)
            probability_table[Agent1.position - 1] = 0
            probability_table[surveyNode - 1] = 0
            print("Sum of prob table after survey no prey found and agent move:", sum(probability_table))

            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            print("Sum of prob table after prey moves :", sum(probability_table))

            #VISH
            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("On Top")   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list)  
                print("Sum of prob table after pred moves :", sum(pred_probab_table)) 
            #Predator1.position=pred_next_pos


            #Check if predator won
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        elif(flag_preySurvey == 1 and surveyRes == 1): # Agent knows prey pos and doesnt know pred pos but predFound in survey

            #Pred Found -> Update Probab table
            for i in range(len(pred_probab_table)):
                pred_probab_table[i] = 0
            pred_probab_table[surveyNode-1] = 1
            
            #print("Probability Table after survey pred found :", pred_probab_table)
            print("Sum of prob table after survey pred found :", sum(pred_probab_table))

            # Vish Today
            dist_prey={}
            dist_pred={}
            dist={}
            prey_list=[]
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                temp_pred=djikstra(adjacency_list,i,Predator1.position)
                if(temp_prey<5):
                    dist_prey[i]=[temp_prey]
                if(temp_pred<5):
                    dist_pred[i]=[temp_pred]
                if(temp_prey>=5 and temp_pred>=5):
                    dist[i]=[temp_prey,temp_pred]  
            if(len(dist_pred)>0):
                sorted_list_dist = sorted(dist_pred.items(), key=lambda x:x[1])
                dist_pred=dict(sorted_list_dist)
                print("dist_pred ",dist_pred)
                agent_next_pos=list(dist_pred.keys())[len(dist_pred)-1]
            elif(len(dist_prey)>0):
                sorted_list_dist = sorted(dist_prey.items(), key=lambda x:x[1])
                dist_prey=dict(sorted_list_dist)
                print("dist_prey ",dist_prey)
                agent_next_pos=list(dist_prey.keys())[0]    
            else:    
                sorted_list_dist = sorted(dist.items(), key=lambda x:x[1])
                dist=dict(sorted_list_dist) 
                print("dist ",dist) 
                var=list(dist.keys())[0]  
                agent_next_pos=0                                                 
                for i in dist:
                    if(dist[i][0] == dist[var][0] and dist[i][1] >2):
                        agent_next_pos=i
                
                maximus=0
                if(agent_next_pos==0):
                    print("HERE HERE HERE HERE HERE HERE")
                    for i in dist:
                        if(dist[i][1] != 0):          
                            if(i>maximus):
                                agent_next_pos=i
                                maximus=i
            # Vish Today
            if(agent_next_pos != 0):                                            
                Agent1.position=agent_next_pos
            
            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break
            
            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            #print("Probability Table after prey move :", probability_table)
            print("Sum of prob table after prey moves :", sum(probability_table))

            #VISH
            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
                #print("Probability Table after pred move :", pred_probab_table)
                #print("Len  = ", len(pred_probab_table))
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("On Top")   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list)  
                #print("Probability Table after pred move :", pred_probab_table)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
                #print("In Bottom")

            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        elif(flag_preySurvey == 1 and surveyRes == 0):# Agent knows prey pos and doesnt know pred pos but pred not in survey

            #Pred Not Found -> Update Probab table
            print("Agent Position", Agent1.position)
            for i in range(len(pred_probab_table)):
                pred_probab_table[i] = 1/(no_of_nodes-2)
            pred_probab_table[surveyNode-1] = 0
            pred_probab_table[Agent1.position-1] = 0
            #print("Probability Table after survey no Pred found :", pred_probab_table)
            print("Sum of prob table after survey no Pred found :", sum(pred_probab_table))

            # As all probabilities are equal -> Calculate farthest movement from Predator
            dist={}
            for i in adjacency_list[Agent1.position]:
                temp_prey=djikstra(adjacency_list,i,Prey1.position)
                dist[i] = temp_prey
            
            minDistance = 10000000
            minNbr = 0
            for key,val in dist.items():
                if (val < minDistance):
                    minNbr = key
                    minDistance = val
            
            # Move the agent

            Agent1.position = minNbr

            #Check if Agent Won /died                                 
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

            #Now moving the prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)
            #print("Probability Table after prey move :", probability_table)
            print("Sum of prob table after prey moves :", sum(probability_table))

            # if(Agent1.position == Prey1.position):
            #     flag=1
            #     print("Agent won")
            #     break

            #VISH
            pred_choice=np.random.choice([0,1],p=[0.6,0.4])
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
                #Update belief Probablity Table of Agent
                probab_pred_distribution(adjacency_list, Agent1.position)
            #    print("Probability Table after pred move :", pred_probab_table)
            #    print("Len  = ", len(pred_probab_table))
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
            #    print("On Top")   
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                prob_distracted(adjacency_list)  
            #    print("Probability Table after pred move :", pred_probab_table)
                print("Sum of prob table after pred moves :", sum(pred_probab_table))   
            #    print("In Bottom")

            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        # Checking for next survey
        if( 1 in pred_probab_table ): # Agent knows where pred is -> Survey for prey in next
            flag_preySurvey = 0
            flag_predSurvey = 1

            # Take the max of the probability table for position of the new survey
            maxProbab = max(probability_table)
            maxPosition = probability_table.index(maxProbab) + 1 #Adding 1 to compensate for the indices starting from 0 rather than 1

            # Do a survey on this maxPosition
            #prey_pos = Prey1.position
            surveyNode = maxPosition
            print("Survey Node:", surveyNode)
            surveyRes = Survey(surveyNode, Prey1.position)
        
        else: # Agent doesn't know where pred is -> Survey for pred

            flag_predSurvey = 0
            flag_preySurvey = 1

             # Take the max of the probability table for position of the new survey
            maxProbab = max(pred_probab_table)
            maxPosition = pred_probab_table.index(maxProbab) + 1 #Adding 1 to compensate for the indices starting from 0 rather than 1

            # Do a survey on this maxPosition
            #prey_pos = Prey1.position
            surveyNode = maxPosition
            print("Survey Node:", surveyNode)
            surveyRes = fault_survey(surveyNode, Predator1.position)


    percentCalc = (count_prey_pos/loopCounter) * 100
    
    prob_count_prey_fault_drone_ag8.append(percentCalc)
    percentCalc = (count_pred_pos/loopCounter) * 100
    prob_count_pred_fault_drone_ag8.append(percentCalc)

    if(flag==1):
        return 1
    if(flag==2):
        return 2
    if(flag==0):
        return 0



count_agent_1=0
count_pred_1=0
count_agent_2=0
count_pred_2=0
count_agent_3=0
count_pred_3=0
count_agent_4=0
count_pred_4=0
count_agent_5=0
count_pred_5=0
count_agent_6=0
count_pred_6=0
count_agent_7=0
count_pred_7=0
count_agent_8=0
count_pred_8=0
count_agent_9=0
count_pred_9=0
count_agent_10=0
count_pred_10=0



tie=0
tie2=0
tie3=0
tie4=0
tie5=0
tie6=0
tie7=0
tie8=0
tie9=0
tie10=0

for x in range(0,100):
    temp=1
    adjacency_list={}
    for i in range(1,no_of_nodes+1):
        if(i==1):
            adjacency_list[temp]=[no_of_nodes,temp+1]
        elif(i==no_of_nodes):    
            adjacency_list[temp]=[temp-1,1]
        else:    
            adjacency_list[temp]=[temp-1,temp+1]
        temp=temp+1
    print(adjacency_list)
    print("----------------------")
    temp=1
    degree_check={}
    for i in range(1,no_of_nodes+1):
        if(i==1):
            degree_check[temp]=[no_of_nodes,temp+1]
        elif(i==no_of_nodes):    
            degree_check[temp]=[temp-1,1]
        else:    
            degree_check[temp]=[temp-1,temp+1]
        temp=temp+1
    index=list(range(1,no_of_nodes+1))
    count=0
    flag=0
    while(degree_check):
        choice = np.random.choice(index)
        l= list(range(choice-5,choice+5+1))
        list_of_possible_edges=[]
        for x in l:
            if(x==0):
                continue
            if(x<1):
                if(no_of_nodes+1+x in index):
                    list_of_possible_edges.append(no_of_nodes+1+x)
                else:
                    continue    
            elif(x>no_of_nodes):
                if(x-no_of_nodes in index):
                    list_of_possible_edges.append(x-no_of_nodes) 
                else:
                    continue    
            else:
                list_of_possible_edges.append(x) 
        list_of_possible_edges = list(dict.fromkeys(list_of_possible_edges))  
        list_of_possible_edges = [ x for x in list_of_possible_edges if len(adjacency_list[x])==2]
        list_of_possible_edges.remove(choice)
        if(choice-1>0 and choice-1 in list_of_possible_edges):
            if(choice==1):
                list_of_possible_edges.remove(no_of_nodes)
            else:
                list_of_possible_edges.remove(choice-1)   
        if(choice+1<=no_of_nodes and choice+1 in list_of_possible_edges):
            if(choice==no_of_nodes):
                list_of_possible_edges.remove(1)
            else:
                list_of_possible_edges.remove(choice+1)    
        #print("list_of_possible_edges ",list_of_possible_edges)
        
        if(len(list_of_possible_edges)==0):
            index.remove(choice)
            del degree_check[choice] 
        else:
            chosen_edge=np.random.choice(list_of_possible_edges)
            #print("chosen_edge ",chosen_edge)
            if(len(adjacency_list[chosen_edge])==2):
                adjacency_list[chosen_edge].append(choice)
                adjacency_list[choice].append(chosen_edge)
                count=count+1
                
                flag=1
        
        if(flag==1):
            index.remove(choice)
            index.remove(chosen_edge)   
            flag=0 
            del degree_check[choice]   
            del degree_check[chosen_edge] 

    print(adjacency_list)
    print(count)

    for w in range(0,30):
        var=Agent1(adjacency_list)
        if(var==1):
            count_agent_1=count_agent_1+1
        if(var==2):
            count_pred_1=count_pred_1+1  
        if(var==0):
            tie=tie+1 
    
    for w in range(0,30):
        var=Agent2(adjacency_list)
        if(var==1):
            count_agent_2=count_agent_2+1
        if(var==2):
            count_pred_2=count_pred_2+1  
        if(var==0):
            tie2=tie2+1        
    
    for w in range(0,30):
        var=Ag3(adjacency_list)
        if(var==1):
            count_agent_3=count_agent_3+1
        if(var==2):
            count_pred_3=count_pred_3+1  
        if(var==0):
            tie3=tie3+1
    
    for w in range(0,30):
        var=Agent4(adjacency_list)
        if(var==1):
            count_agent_4=count_agent_4+1
        if(var==2):
            count_pred_4=count_pred_4+1  
        if(var==0):
            tie4=tie4+1        
    
    for w in range(0,30):
        var=Ag5(adjacency_list)
        if(var==1):
            count_agent_5=count_agent_5+1
        if(var==2):
            count_pred_5=count_pred_5+1  
        if(var==0):
            tie5=tie5+1                
    
    for w in range(0,30):
        var=Agent6(adjacency_list)
        if(var==1):
            count_agent_6=count_agent_6+1
        if(var==2):
            count_pred_6=count_pred_6+1  
        if(var==0):
            tie6=tie6+1


    for w in range(0,30):
        var=Ag7(adjacency_list)
        if(var==1):
            count_agent_7=count_agent_7+1
        if(var==2):
            count_pred_7=count_pred_7+1  
        if(var==0):
            tie7=tie7+1 
    
    for w in range(0,30):
        var=Agent8(adjacency_list)
        if(var==1):
            count_agent_8=count_agent_8+1
        if(var==2):
            count_pred_8=count_pred_8+1  
        if(var==0):
            tie8=tie8+1
        
    for w in range(0,30):
        var=faulty_drone_ag7(adjacency_list)
        if(var==1):
            count_agent_9=count_agent_9+1
        if(var==2):
            count_pred_9=count_pred_9+1  
        if(var==0):
            tie9=tie9+1
    
    for w in range(0,30):
        var=faulty_drone_agent8(adjacency_list)
        if(var==1):
            count_agent_10=count_agent_10+1
        if(var==2):
            count_pred_10=count_pred_10+1  
        if(var==0):
            tie10=tie10+1
        
    


x =["Agent1","Agent2","Agent3","Agent4","Agent5","Agent6","Agent7","Agent8"]
y = [(count_agent_1/3000)*100, (count_agent_2/3000)*100, (count_agent_3/3000)*100,(count_agent_4/3000)*100,(count_agent_5/3000)*100,(count_agent_6/3000)*100,(count_agent_7/3000)*100,(count_agent_8/3000)*100]
plt.plot(x,y)
plt.title("Agent Winning Percentage")
plt.xlabel("Agent")
plt.ylabel("Winning Percentage")
leg=plt.legend()
plt.show()


print("count_agent_1 ",(count_agent_1/3000)*100," count_pred_1 ",(count_pred_1/3000)*100, " tie ",tie)
print("count_agent_2 ",(count_agent_2/3000)*100," count_pred_2 ",(count_pred_2/3000)*100, " tie2 ",tie2)
print("count_agent_3 ",(count_agent_3/3000)*100," count_pred_3 ",(count_pred_3/3000)*100, " tie3 ",tie3)
print("Average Value Ag3 Prey pos: ", ((sum(prob_count_ag3)/3000)))
print("count_agent_4 ",(count_agent_4/3000)*100," count_pred_4 ",(count_pred_4/3000)*100, " tie4 ",tie4)
print("count_agent_5 ",(count_agent_5/3000)*100," count_pred_5 ",(count_pred_5/3000)*100, " tie5 ",tie5)
print("Average Value Ag5 Pred Pos: ", ((sum(prob_count_ag5)/3000)))
print("count_agent_6 ",(count_agent_6/3000)*100," count_pred_6 ",(count_pred_6/3000)*100, " tie6 ",tie6)
print("count_agent_7 ",(count_agent_7/3000)*100," count_pred_6 ",(count_pred_7/3000)*100, " tie6 ",tie7)
print("Average Value prey Ag7: ", ((sum(prob_count_prey_ag7)/3000)))
print("Average Value pred Ag7: ", ((sum(prob_count_pred_ag7)/3000)))
print("count_agent_8 ",(count_agent_8/3000)*100," count_pred_8 ",(count_pred_8/3000)*100, " tie8 ",tie8)                                  

print("count_agent_9 ",(count_agent_9/3000)*100," count_pred_6 ",(count_pred_9/3000)*100, " tie9 ",tie9)
print("Average Value prey Ag7: ", ((sum(prob_count_prey_fault_drone_ag7)/3000)))
print("Average Value pred Ag7: ", ((sum(prob_count_pred_fault_drone_ag7)/3000)))

print("count_agent_9 ",(count_agent_10/3000)*100," count_pred_6 ",(count_pred_10/3000)*100, " tie9 ",tie9)
print("Average Value prey Ag7: ", ((sum(prob_count_prey_fault_drone_ag8)/3000)))
print("Average Value pred Ag7: ", ((sum(prob_count_pred_fault_drone_ag8)/3000)))



 