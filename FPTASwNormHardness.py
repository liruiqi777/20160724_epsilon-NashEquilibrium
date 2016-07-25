#Code for generating an illustration for the paper
#7 node tree with hard-coded payoff matrices

import networkx as nx
import matplotlib.pyplot as plt
import time
import random
#import pickle
from networkx.drawing.nx_agraph import *

#import pygraphviz

#Payoff values-- dictionary
#Key : (source, target)
#Value: [ [source plays 0], [source plays 1] ]

M1 = {
    (0, 1): [ [0.4, 0], [0.1, 0.5] ], #how much 0 is getting from 1

    (1, 0): [ [0.3, 0], [0, 0.5] ],
    (1, 2): [ [0, 0.3], [0.4, 0.1] ],

    (2, 1): [ [0, 0.2], [0.3, 0.1] ],
    (2, 3): [ [0, 0.4], [0.3, 0] ],

    (3, 2): [ [0.1, 0.2], [0.4, 0] ],
    (3, 4): [ [0, 0.4], [0.2, 0] ],

    (4, 3): [ [0.2, 0.4], [0.3, 0] ],
    (4, 5): [ [1, 0.4], [0, 0.8] ],

    (5, 4): [ [1, 0], [0, 0.5] ],
    (5, 6): [ [0.8, 0], [0.2, 1] ],

    (6, 5): [ [1, 0], [0.2, 0.8] ],
    }


#M2 = {
#    (0, 1): [ [0.5, 0], [0, 0.5] ], #how much 0 is getting from 1
#    (0, 2): [ [0.5, 0], [0, 0.5] ],
#
#    (1, 0): [ [0.33, 0], [0, 0.33] ],
#    (1, 3): [ [0.33, 0], [0, 0.33] ],
#    (1, 4): [ [0.33, 0], [0, 0.33] ],
#
#    (2, 0): [ [0.33, 0], [0, 0.33] ],
#    (2, 5): [ [0.33, 0], [0, 0.33] ],
#    (2, 6): [ [0.33, 0], [0, 0.33] ],
#
#    (3, 1): [ [1.0, 0], [0, 1.0] ],
#
#    (4, 1): [ [1.0, 0], [0, 1.0] ],
#    
#    (5, 2): [ [1.0, 0], [0, 1.0] ],
#
#    (6, 2): [ [1.0, 0], [0, 1.0] ],    
#    }



file = open("20160712 FPTAS data1.csv", "w")
file.write("game#,difficulty1,difficulty2,s1,s2,esp,upstreamT,0,1,2,3,4,5,6\n")


file2 = open("20160712 game collection1.txt", "wt")


#Create a balanced r-ary tree of height h 
#Node 0 is the root
def create_tree(r, h):
    G = nx.balanced_tree(r, h)

    return G


    
    #nx.draw_networkx(G, pos = nx.spring_layout(G))
    #nx.draw_spring(G)

    for i in range(7):
        G.node[i]['color'] = 'white'

    

    #######  Visualization with graphviz  ########

    #write_dot(G,'test.dot')

    # same layout using matplotlib with no labels
    #plt.title("draw_networkx")
    pos=graphviz_layout(G,prog='dot')
    nx.draw(G,pos,with_labels=True,arrows=False, node_color = 'lightgray')


    plt.show()
    return G

inf = float('inf')
m = 2 #Number of actions
s1 = 11 #probability grid size
s2 = 11 #payoff grid size
eps = 0.1
actions = [0, 1] #set of actions for each player
err = 1e-10
print_st = ""

ansCount = 0


M = M1
currHardness = (0,0)
gameNum = 0
neighbSize = {0:1, 1:2, 2:2, 3:2, 4:2, 5:2, 6:1}
neighborhood = {0:[1], 1:[0, 2], 2:[1, 3], 3:[2, 4], 4:[3, 5], 5:[4, 6], 6:[5]}

#T_{i->j}(pi, pj)
T = {} # dictionary with keys as 4-tuples
W = {} # dictionary with keys as 4-tuples
B = {} #Key: 5-tuple
R = {} #Key: 3+1-tuple. Extra: i 
arg_R = {} #arg max of R: 4 -> 2
S = {} #Key: 2+1-tuple. Extra: i
F = {} #Key: 5+1-tuple. Extra: i

strategy = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
ch = {0:[1], 1:[2], 2:[3], 3:[4], 4:[5], 5:[6], 6:[]}
pa = {0:None, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5}

#Note: j's probability is given, not action
def M_prime(i, j, a_i, p_j):
    total = 0.0 #expected payoff of i

    for a_j in range(0,m):
        total += p_j[a_j] * M[(i,j)][a_i][a_j]

    return round((s2-1) * total) / (s2-1)

def init_boundary():
    pass

'''
def get_B_leaf(i, j, p_i, p_j):
    for a_i in range(0,m):
        for a_i_prime in range(0,m):
            if p_i[a_i_prime] * M_prime(i, j, a_i_prime, p_j) < M_prime(i, j, a_i, p_j) - eps:
                return -inf
    return 0.0
'''


def get_T(i, j, p_i, p_j):
    global print_st, T, W

    if (i, j, p_i, p_j) in T:
        return T[(i, j, p_i, p_j)]

    if len(ch[i]) == 0: #i is a leaf
        #import pdb; pdb.set_trace()
        if get_B(i, j, p_i, p_j) == 0:
            #print i, "->", j, "p_i=", p_i, "p_j=", p_j, "\tWitness S = ", S_last 
            T[(i, j, p_i, p_j)] = 0
            W[(i, j, p_i, p_j)] = (0.0, 0.0)
            print_st += str(i) + " -> " + str(j) + \
                            " p_i= " + str(p_i) + " p_j= " + str(p_j)
            print_st += "\n"
            return 0
    else:   #i is an internal node or root
        for S_last in [ (float(x)/(s2-1), float(y)/(s2-1)) for x in range(0,s2) for y in range(0,s2)]:
            if  get_B(i, j, p_i, p_j, S_last) == 0 and \
               get_R(i, ch[i][-1], p_i, S_last) == 0:

                T[(i, j, p_i, p_j)] = 0
                W[(i, j, p_i, p_j)] = S_last
                print_st += str(i) + " -> " + str(j) + \
                                " p_i= " + str(p_i) + " p_j= " + str(p_j) +\
                                " Witness S = " + str(S_last)
                print_st += "\n"
                return 0
        
    #print "Returning -inf for: ", i, "->", j, "p_i=", p_i, "p_j=", p_j, "\tWitness S = ", S_last 
    T[(i, j, p_i, p_j)] = -inf
    W[(i, j, p_i, p_j)] = None
    return -inf
            
#Default argument: S_last is all 0's (for leaf node i)
def get_B(i, j, p_i, p_j, S_last = [0.0] * m):
    #if j == None: #i is the root
    #    return 0
    
    global B
    S_last = tuple(S_last)
    
    if (i, j, p_i, p_j, S_last) in B:
        return B[(i, j, p_i, p_j, S_last)]
    
    for a_i in range(0,m):
        if j != None:
            total = 0.0 #i's current total payoff for playing p_i
            for a_i_prime in range(0,m):
                total += p_i[a_i_prime] * (M_prime(i, j, a_i_prime, p_j) + S_last[a_i_prime])
            if total < M_prime(i, j, a_i, p_j) + S_last[a_i] - eps:
                B[(i, j, p_i, p_j, S_last)] = -inf
                return -inf
        else:
            total = 0.0
            for a_i_prime in range(0,m):
                total += p_i[a_i_prime] * S_last[a_i_prime]
            if total < S_last[a_i] - eps:
                B[(i, j, p_i, p_j, S_last)] = -inf
                return -inf

    B[(i, j, p_i, p_j, S_last)] = 0
    return 0


def get_F(i, o_l, p_i, S_o_l, p_o_l, S_o_prev):
    global F
    
    if (i, o_l, p_i, S_o_l, p_o_l, S_o_prev) in F:
        return F[(i, o_l, p_i, S_o_l, p_o_l, S_o_prev)]
    
    #print "i = ", i, "o_l = ", o_l
    
    #TO DO: More efficient: make sure that irrelevant values of S don't come in
    '''
    for a_i in range(0,m):
        if abs ( S_o_l[a_i] - (M_prime(i, o_l, a_i, p_o_l) + S_o_prev[a_i]) ) > err:
            return -inf
    '''
    ''' the above can be deleted by uncommenting the part in get_R '''
    
    F[(i, o_l, p_i, S_o_l, p_o_l, S_o_prev)] = get_T(o_l, i, p_o_l, p_i) + get_R(i, o_l - 1, p_i, S_o_prev)
    return F[(i, o_l, p_i, S_o_l, p_o_l, S_o_prev)]

def get_R(i, o_l, p_i, S_o_l):
    if o_l not in ch[i]:
        #if o_l == 2:
        #    import pdb; pdb.set_trace()
        if S_o_l[0] < err and S_o_l[1] < err: #Boundary condition: S_o_0 = 0
            return 0
        else:
            return -inf
    global R, argR
    if (i, o_l, p_i, S_o_l) in R:
        return R[(i, o_l, p_i, S_o_l)]

    #Only for 2-action case
    R[(i, o_l, p_i, S_o_l)] = -inf
    arg_R[(i, o_l, p_i, S_o_l)] = ( None, None )
    for p_o_l_0 in [float(x)/(s1-1) for x in range(0,s1)]:
        p_o_l = (p_o_l_0, 1 - p_o_l_0)        
        S_o_prev = ( S_o_l[0] - M_prime(i, o_l, 0, p_o_l),
                    S_o_l[1] - M_prime(i, o_l, 1, p_o_l) )

        if S_o_prev[0] < 0 or S_o_prev[1] < 0:
            continue
        
        #for S_o_prev in [ [x*1.0/(s2-1), ...] for x in range(0,s2) for y in range(0,s2)]:
        f = get_F(i, o_l, p_i, S_o_l, p_o_l, S_o_prev)
        if f == 0:
            R[(i, o_l, p_i, S_o_l)] = 0
            arg_R[(i, o_l, p_i, S_o_l)] = ( p_o_l, S_o_prev )            
            return 0

    return -inf

#S_partial is the sum of i's payoffs up to i's last child
def downstream(i, p_i, S_partial):
    
    print (i, " plays ", p_i, " (S_partial = ", S_partial, ")")
    strategy[i] = p_i[0]
    
    if len(ch[i]) == 0:
        return

    for ch_index in range(len(ch[i]) - 1, -1, -1):
        ch_i = ch[i][ch_index]
        p_ch_i, S_prev = arg_R[i, ch_i, p_i, S_partial]

        S_next_level = None
        for key, val in T.items():
            #Check: key[1] needed?
            if key[0] == ch_i and key[1] == i and \
               key[2] == p_ch_i and key[3] == p_i and val == 0:
               
                S_next_level = W[key]

        if S_next_level == None:
            print ("No epsilon-MSNE!")
            return
        downstream(ch_i, p_ch_i, S_next_level)

        S_partial = S_prev


def randPayoff():
    global M
    global neighborhood
    global neighbSize
    
    # Randomizing Payoff
    for x in range(0, len(ch)):
        playingZeroMax = 0
        playingOneMax = 0
        playingZeroMin = 0
        playingOneMin = 0
        
        # Randomizing Payoff
        for y in range(0, neighbSize[x]):
            M[(x, neighborhood[x][y])][0][0] = round(random.uniform(0.0, 1.0), 3)
            M[(x, neighborhood[x][y])][0][1] = round(random.uniform(0.0, 1.0), 3)
            playingZeroMax += max(M[(x, neighborhood[x][y])][0][0], M[(x, neighborhood[x][y])][0][1])
            playingZeroMin += min(M[(x, neighborhood[x][y])][0][0], M[(x, neighborhood[x][y])][0][1])
            
            M[(x, neighborhood[x][y])][1][0] = round(random.uniform(0.0, 1.0), 3)
            M[(x, neighborhood[x][y])][1][1] = round(random.uniform(0.0, 1.0), 3)
            playingOneMax += max(M[(x, neighborhood[x][y])][1][0], M[(x, neighborhood[x][y])][1][1])
            playingOneMin += min(M[(x, neighborhood[x][y])][1][0], M[(x, neighborhood[x][y])][1][1])
        
        playingMax = max(playingOneMax, playingZeroMax)
        playingMin = min(playingOneMin, playingZeroMin)
        
        # Normalizing Payoff Matrix
        for y in range(0, neighbSize[x]):
            M[(x, neighborhood[x][y])][0][0] = round((M[(x, neighborhood[x][y])][0][0]-playingMin/neighbSize[x])/(playingMax-playingMin/neighbSize[x]), 3)
            M[(x, neighborhood[x][y])][0][1] = round((M[(x, neighborhood[x][y])][0][1]-playingMin/neighbSize[x])/(playingMax-playingMin/neighbSize[x]), 3)
            M[(x, neighborhood[x][y])][1][0] = round((M[(x, neighborhood[x][y])][1][0]-playingMin/neighbSize[x])/(playingMax-playingMin/neighbSize[x]), 3)
            M[(x, neighborhood[x][y])][1][1] = round((M[(x, neighborhood[x][y])][1][1]-playingMin/neighbSize[x])/(playingMax-playingMin/neighbSize[x]), 3)



def hardness():
    global M
    global ch
    global pa
    global err
    global neighbSize
    global neighborhood
    
    numAgent = len(ch)
    ##    neighbSize = {}
    ##    neighborhood = {}
    ##
    ##    # initializing the dictionaries for the neighborhood size
    ##    # and neighborhood members of each player
    ##    for x in range(0, numAgent):
    ##        if pa[x] == None:
    ##            neighbSize[x] = len(ch[x])
    ##            neighborhood[x] = ch[x]
    ##        else:
    ##            neighbSize[x] = len(ch[x]) + 1
    ##            neighborhood[x] = ch[x]
    ##            neighborhood[x].append(pa[x])
    
    # computing a difference in payoff matrix to help simplify
    # the hardness computation
    diffDict = M.copy()
    for x in diffDict.keys():
        diffDict[x] = [M[x][0][0] - M[x][1][0],\
                       M[x][0][1] - M[x][1][1]]
    #    edgeList = diffDict.keys()
    #    for x in range(0, len(edgeList)):
    #        myKey = edgeList[x]
    #        diffDict[myKey] = [M[myKey][0][0] - M[myKey][1][0],\
    #                         M[myKey][0][1] - M[myKey][1][1]]
    
    # assuming that totalHardness = sum of hardness at each agent
    totalHardness = 0.0
    totalHardness1 = 0.0
    
    # calculating the hardness at each agent
    for x in range(0, numAgent):
        decisionDict= {0:0, 1:0}
        currNeighbSize = neighbSize[x]
        totalConfig = pow(2, currNeighbSize)
        for y in range(0, totalConfig):
            binaryRep = bin(y)[2:].zfill(currNeighbSize)
            payoffDiff = 0.0
            
            for z in range(0, currNeighbSize):
                payoffDiff += diffDict[(x, neighborhood[x][z])][int(binaryRep[z])]
            
            if abs(payoffDiff) <= err:
                decisionDict[0] += 1
                decisionDict[1] += 1
            elif payoffDiff > 0:
                decisionDict[0] += 1
            else :
                decisionDict[1] += 1
        
        # calculate the hardness at each agent and sum into totalHardness
        totalHardness += float(decisionDict[0]*decisionDict[1])/float(totalConfig*totalConfig)

        if decisionDict[0] < totalConfig/2:
            totalHardness1 += float(totalConfig - 2*decisionDict[0])/totalConfig
        elif decisionDict[1] < totalConfig/2:
            totalHardness1 += float(totalConfig - 2*decisionDict[1])/totalConfig


#        if decisionDict[0] < totalConfig/2:
#            totalHardness += float(totalConfig - 2*decisionDict[0])/totalConfig
#        elif decisionDict[1] < totalConfig/2:
#            totalHardness += float(totalConfig - 2*decisionDict[1])/totalConfig
    # normalizing the total hardness to a number between 0 and 1
    return (totalHardness, totalHardness1)



def fptas():
    time1 = time.time()
    
    global print_st
    global s1
    global s2
    global eps
    global ansCount
    global strategy
    global currHardness
    global gameNum
    
    G = create_tree(6, 1)
    node_list = nx.nodes(G)

    #Need to initialize the parent dict pa from the graph
    
    node_list.reverse() #get ordering: leaves to root
    #print node_list
    for i in node_list:
        j = pa[i]
        print_st += str(i) + " -> " + str(j)
        print_st += "\n"
        for p_i_0 in [float(x)/(s1-1) for x in range(0,s1)]:
            p_i = (p_i_0, 1 - p_i_0)
            if j == None:
                get_T(i, j, p_i, None)
            else:
                for p_j_0 in [float(x)/(s1-1) for x in range(0,s1)]:
                    p_j = (p_j_0, 1 - p_j_0)
                    get_T(i, j, p_i, p_j)

        if j == None: #i is the root
            time2 = time.time()
            max_T = -inf
            p_arg_max_T = None #Check
            S_last = None #Check
            
            f = open("output.txt", "wt")
            f.write(print_st)
            f.close()

            for key, val in T.items():
                if key[0] == i and val == 0:
                    p_arg_max_T = key[2]
                    S_last = W[key]
                    downstream(i, p_arg_max_T, S_last)
                    file.write(str(gameNum))
                    file.write(",")
                    file.write(str(currHardness[0])[0:])
                    file.write(",")
                    file.write(str(currHardness[1])[0:])
                    file.write(",")
                    file.write(str(s1))
                    file.write(",")
                    file.write(str(s2))
                    file.write(",")
                    file.write(str(eps))
                    file.write(",")
                    file.write(str(time2-time1))
                    file.write(",")
                    for x in range (0, 7):
                        file.write(str(strategy[x]))
                        file.write(",")
                    file.write("\n")
                    print ("---------------------------------------------")
                    strategy = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0}
            # ansCount = ansCount + 1
            # return
            return

#    time2 = time.time()
#    file.write(str(gameNum))
#    file.write(",")
#    file.write(str(currHardness[0])[0:])
#    file.write(",")
#    file.write(str(currHardness[1])[0:])
#    file.write(",")
#    file.write(str(s1))
#    file.write(",")
#    file.write(str(s2))
#    file.write(",")
#    file.write(str(eps))
#    file.write(",")
#    file.write(str(time2-time1))
#    file.write(",")
#    for x in range (0, 7):
#        file.write(str(strategy[x]))
#        file.write(",")
#    file.write("\n")
#
#    f = open("output.txt", "wt")
#    f.write(print_st)
#    f.close()



for game in range(0, 20):

    global M
    
    randPayoff()
#    if game == 1:
#        M = M2
#    elif not game == 0:
#        randPayoff()
    global currHardness
    currHardness = hardness()
#    global gameNum
    gameNum = game+1

    file2.write(str(gameNum))
    file2.write("\n")
    file2.write(str(M))
    file2.write("\n\n")
    
    for x in range (8, 9):
#        global s1
#        global s2
#        global print_st
#        global T
#        global W
#        global B
#        global R
#        global arg_R
#        global S
#        global F
#        global strategy
#        global file

        s1 = 5*x+1
        s2 = 5*x+1
        fptas()
        
        print_st = ""
        T = {} # dictionary with keys as 4-tuples
        W = {} # dictionary with keys as 4-tuples
        B = {} #Key: 5-tuple
        R = {} #Key: 3+1-tuple. Extra: i
        arg_R = {} #arg max of R: 4 -> 2
        S = {} #Key: 2+1-tuple. Extra: i
        F = {} #Key: 5+1-tuple. Extra: i
        strategy = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0}
        print ("Done!")
        #create_tree(2,2)

file.close()
file2.close()




'''
GAMES THAT CAN BE SOLVED!!!

M3 = {(0, 1): [[0.61, 0.89], [0.29, 0.68]], (1, 3): [[0.37, 0.33], [0.47, 0.47]], (4, 1): [[0.05, 0.64], [0.36, 0.65]], (3, 1): [[0.16, 0.68], [0.39, 0.85]], (1, 4): [[0.5, 0.5], [0.37, 0.07]], (0, 2): [[0.21, 0.72], [0.66, 0.65]], (2, 0): [[0.28, 0.66], [0.27, 0.99]], (2, 6): [[0.17, 0.37], [0.01, 0.37]], (6, 2): [[0.01, 0.98], [0.68, 0.88]], (2, 5): [[0.35, 0.18], [0.11, 0.19]], (5, 2): [[0.9, 0.52], [0.16, 0.31]], (1, 0): [[0.36, 0.84], [0.32, 0.88]]}

M4 = {(0, 1): [[0.73, 0.7], [0.53, 0.24]], (1, 3): [[0.27, 0.46], [0.68, 0.46]], (4, 1): [[0.85, 0.69], [0.89, 0.7]], (3, 1): [[0.73, 0.97], [0.09, 0.16]], (1, 4): [[0.89, 0.7], [0.02, 0.29]], (0, 2): [[0.39, 0.11], [0.31, 0.59]], (2, 0): [[0.11, 0.02], [0.93, 0.81]], (2, 6): [[0.43, 0.56], [0.27, 0.63]], (6, 2): [[0.44, 0.48], [0.06, 0.49]], (2, 5): [[0.88, 0.29], [0.54, 0.7]], (5, 2): [[0.39, 0.01], [0.58, 0.17]], (1, 0): [[0.07, 0.93], [0.42, 0.34]]}

M5 = {(0, 1): [[0.66, 0.4], [0.16, 0.12]], (1, 3): [[0.03, 0.54], [0.64, 0.87]], (4, 1): [[0.55, 0.36], [0.27, 0.54]], (3, 1): [[0.2, 0.92], [0.76, 0.05]], (1, 4): [[0.92, 0.44], [0.03, 0.56]], (0, 2): [[0.14, 0.02], [0.2, 0.57]], (2, 0): [[0.33, 0.03], [0.84, 0.19]], (2, 6): [[0.38, 0.91], [0.23, 0.07]], (6, 2): [[0.29, 0.78], [0.26, 0.62]], (2, 5): [[0.68, 0.52], [0.27, 0.69]], (5, 2): [[0.07, 0.47], [0.76, 0.6]], (1, 0): [[0.24, 0.73], [0.15, 0.8]]}

M6 = {(0, 1): [[0.73, 0.42], [0.88, 0.09]], (1, 3): [[0.73, 0.01], [0.44, 0.47]], (4, 1): [[0.44, 0.65], [0.91, 0.95]], (3, 1): [[0.61, 0.34], [0.78, 0.21]], (1, 4): [[0.68, 0.52], [0.99, 0.23]], (0, 2): [[0.94, 0.06], [0.89, 0.43]], (2, 0): [[0.65, 0.22], [0.23, 0.15]], (2, 6): [[0.83, 0.05], [0.71, 0.42]], (6, 2): [[0.61, 0.08], [0.85, 0.86]], (2, 5): [[0.36, 0.7], [0.69, 0.4]], (5, 2): [[0.26, 0.22], [0.23, 0.19]], (1, 0): [[0.02, 0.51], [0.49, 0.82]]}

M7 = {(0, 1): [[0.17, 0.7], [0.01, 0.5]], (1, 3): [[0.33, 0.64], [0.27, 0.86]], (4, 1): [[0.47, 0.16], [0.16, 0.95]], (3, 1): [[0.09, 0.29], [0.38, 0.76]], (1, 4): [[0.21, 0.6], [0.33, 0.07]], (0, 2): [[0.12, 0.13], [0.01, 0.66]], (2, 0): [[0.74, 0.86], [0.72, 0.92]], (2, 6): [[0.01, 0.57], [0.18, 0.47]], (6, 2): [[0.86, 0.43], [0.5, 0.53]], (2, 5): [[0.66, 0.87], [0.71, 0.3]], (5, 2): [[0.06, 0.76], [0.35, 0.76]], (1, 0): [[0.08, 0.15], [0.1, 0.83]]}

M8 = {(0, 1): [[0.83, 0.54], [0.91, 0.06]], (1, 3): [[0.09, 0.11], [0.35, 0.5]], (4, 1): [[1.0, 0.65], [0.18, 0.8]], (3, 1): [[0.77, 0.92], [0.17, 0.54]], (1, 4): [[0.26, 0.49], [0.35, 0.6]], (0, 2): [[0.47, 0.83], [0.32, 0.78]], (2, 0): [[0.94, 0.95], [0.9, 0.88]], (2, 6): [[0.11, 0.41], [0.28, 0.28]], (6, 2): [[0.81, 0.08], [0.68, 0.33]], (2, 5): [[0.5, 0.92], [0.89, 0.47]], (5, 2): [[0.22, 0.16], [0.43, 0.9]], (1, 0): [[0.95, 0.17], [0.44, 0.95]]}

M9_6_2.25_414105_128.226 = {(0, 1): [[0.34, 0.52], [0.39, 0.78]], (1, 3): [[0.06, 0.78], [0.48, 0.46]], (4, 1): [[0.36, 0.86], [0.78, 0.8]], (3, 1): [[0.45, 0.42], [0.1, 0.4]], (1, 4): [[0.84, 0.99], [0.56, 0.34]], (0, 2): [[0.58, 0.88], [0.23, 0.11]], (2, 0): [[0.91, 0.67], [0.46, 0.98]], (2, 6): [[0.24, 0.98], [0.44, 0.36]], (6, 2): [[0.73, 0.75], [0.63, 0.97]], (2, 5): [[0.11, 0.04], [0.13, 0.59]], (5, 2): [[0.03, 0.68], [0.69, 0.13]], (1, 0): [[0.37, 0.59], [0.73, 0.54]]}

M10_6_4.25_.796875_414105_132.175 = {(0, 1): [[0.67, 0.78], [0.83, 0.03]], (1, 3): [[0.24, 0.19], [0.89, 0.23]], (4, 1): [[0.49, 0.68], [0.67, 0.65]], (3, 1): [[0.43, 0.4], [0.93, 0.57]], (1, 4): [[0.27, 0.35], [0.83, 0.22]], (0, 2): [[0.13, 0.61], [0.23, 0.86]], (2, 0): [[0.57, 0.4], [0.46, 0.17]], (2, 6): [[0.25, 0.49], [0.73, 0.81]], (6, 2): [[0.86, 0.58], [0.24, 0.18]], (2, 5): [[0.75, 0.35], [0.62, 0.17]], (5, 2): [[0.49, 0.99], [0.43, 0.75]], (1, 0): [[0.33, 0.89], [0.73, 0.44]]}

M11_7_4_.75_414105_127.629 = {(0, 1): [[0.77, 0.13], [0.83, 0.67]], (1, 3): [[0.43, 0.19], [0.32, 0.18]], (4, 1): [[0.55, 0.03], [0.09, 0.68]], (3, 1): [[0.31, 0.37], [0.39, 0.8]], (1, 4): [[0.43, 0.07], [0.93, 0.57]], (0, 2): [[0.04, 0.93], [0.33, 0.38]], (2, 0): [[0.61, 0.83], [0.45, 0.41]], (2, 6): [[0.32, 0.46], [0.37, 0.64]], (6, 2): [[0.94, 0.56], [0.77, 0.42]], (2, 5): [[0.97, 0.24], [0.79, 0.08]], (5, 2): [[0.34, 0.85], [0.95, 0.83]], (1, 0): [[0.24, 0.19], [0.75, 1.0]]}

M12_8_3.5_.9375_414105_117.520 = {(0, 1): [[0.94, 0.68], [0.09, 0.01]], (1, 3): [[0.8, 0.18], [0.18, 0.49]], (4, 1): [[0.13, 0.4], [0.1, 0.45]], (3, 1): [[0.81, 0.19], [0.82, 0.75]], (1, 4): [[0.82, 0.0], [0.07, 0.91]], (0, 2): [[0.07, 0.33], [0.68, 0.31]], (2, 0): [[0.37, 0.52], [0.72, 0.07]], (2, 6): [[0.14, 0.08], [0.85, 0.57]], (6, 2): [[0.24, 0.88], [0.9, 0.94]], (2, 5): [[0.02, 0.58], [0.83, 0.1]], (5, 2): [[0.65, 0.53], [0.69, 0.04]], (1, 0): [[0.47, 0.07], [0.0, 0.65]]}

M13_10_3.25_1.0468_414105_155.364 = {(0, 1): [[0.19, 0.07], [0.64, 0.9]], (1, 3): [[0.03, 0.19], [0.36, 0.43]], (4, 1): [[0.19, 0.64], [0.61, 0.35]], (3, 1): [[0.35, 0.92], [0.81, 0.61]], (1, 4): [[0.86, 0.35], [0.25, 0.16]], (0, 2): [[0.92, 0.17], [0.04, 0.18]], (2, 0): [[0.55, 0.97], [0.65, 0.27]], (2, 6): [[0.45, 0.06], [0.05, 0.22]], (6, 2): [[0.33, 0.56], [0.37, 0.67]], (2, 5): [[0.84, 0.6], [0.76, 0.02]], (5, 2): [[0.34, 0.6], [0.24, 0.41]], (1, 0): [[0.42, 0.53], [0.46, 0.14]]}

M14_23_0_1.75_414105_2143.147 = {(0, 1): [[0.28, 0.64], [0.1, 0.34]], (1, 3): [[0.42, 0.65], [0.91, 0.28]], (4, 1): [[0.03, 0.8], [0.41, 0.1]], (3, 1): [[0.14, 0.9], [0.73, 0.66]], (1, 4): [[0.86, 0.0], [0.21, 0.5]], (0, 2): [[0.24, 0.14], [0.59, 0.15]], (2, 0): [[0.66, 0.88], [0.89, 0.46]], (2, 6): [[0.06, 0.15], [0.51, 0.26]], (6, 2): [[0.13, 0.14], [0.07, 0.64]], (2, 5): [[0.97, 0.78], [0.33, 0.93]], (5, 2): [[0.33, 0.29], [0.65, 0.25]], (1, 0): [[0.81, 0.02], [0.11, 0.63]]}

M15_24_4.25_.796875_414105_1667.894 = {(0, 1): [[0.23, 0.04], [0.81, 0.44]], (1, 3): [[0.58, 0.34], [0.36, 0.2]], (4, 1): [[0.79, 0.89], [0.77, 0.57]], (3, 1): [[0.11, 0.28], [0.46, 0.84]], (1, 4): [[0.82, 0.15], [0.05, 0.37]], (0, 2): [[0.7, 0.27], [0.35, 0.09]], (2, 0): [[0.21, 0.42], [0.99, 0.86]], (2, 6): [[0.16, 0.07], [0.06, 0.21]], (6, 2): [[0.5, 0.42], [0.31, 0.97]], (2, 5): [[0.94, 0.85], [0.92, 0.33]], (5, 2): [[0.49, 0.86], [0.26, 0.96]], (1, 0): [[0.41, 0.11], [0.22, 0.85]]}

M16_25_0_414105_2711.501 = {(0, 1): [[0.62, 0.49], [0.1, 0.08]], (1, 3): [[0.6, 0.4], [0.51, 0.21]], (4, 1): [[0.13, 0.48], [0.8, 0.13]], (3, 1): [[0.77, 0.31], [0.95, 0.84]], (1, 4): [[0.27, 0.16], [0.12, 0.3]], (0, 2): [[0.57, 0.11], [0.55, 0.04]], (2, 0): [[0.22, 0.93], [0.65, 0.46]], (2, 6): [[0.48, 0.52], [0.54, 0.55]], (6, 2): [[0.46, 0.93], [0.25, 0.63]], (2, 5): [[0.58, 0.49], [0.54, 0.24]], (5, 2): [[0.27, 0.57], [0.44, 0.79]], (1, 0): [[0.15, 0.88], [0.4, 0.87]]}

M17_31_2.5_1.21875_414105_144.279 = {(0, 1): [[0.34, 0.44], [0.77, 0.48]], (1, 3): [[0.67, 0.79], [0.69, 0.34]], (4, 1): [[0.65, 0.63], [0.29, 0.51]], (3, 1): [[0.14, 0.52], [0.08, 0.7]], (1, 4): [[0.34, 0.18], [0.15, 0.48]], (0, 2): [[0.63, 0.63], [0.06, 0.62]], (2, 0): [[0.52, 0.18], [0.09, 0.07]], (2, 6): [[0.02, 0.08], [0.73, 0.52]], (6, 2): [[0.08, 0.94], [0.51, 0.75]], (2, 5): [[0.48, 0.99], [0.16, 0.57]], (5, 2): [[0.19, 0.33], [0.31, 0.82]], (1, 0): [[0.41, 0.89], [0.78, 0.15]]}

M18_65_3.5_1.03125_414105_318.290 = {(0, 1): [[0.38, 0.29], [0.37, 0.9]], (1, 3): [[0.55, 0.49], [0.88, 0.22]], (4, 1): [[0.77, 0.17], [0.95, 0.8]], (3, 1): [[0.56, 0.91], [0.84, 0.45]], (1, 4): [[0.55, 0.12], [0.48, 0.55]], (0, 2): [[0.24, 0.48], [0.83, 0.16]], (2, 0): [[0.16, 0.57], [0.96, 0.59]], (2, 6): [[0.11, 0.9], [0.57, 0.82]], (6, 2): [[0.26, 0.08], [0.62, 0.21]], (2, 5): [[0.2, 0.27], [0.59, 0.24]], (5, 2): [[0.31, 0.13], [0.24, 0.98]], (1, 0): [[0.83, 0.53], [0.22, 0.29]]}

M19_73_2.25_1.2343_414105_244.459 = {(0, 1): [[0.0, 0.71], [0.37, 0.36]], (1, 3): [[0.84, 0.03], [0.25, 0.77]], (4, 1): [[0.41, 0.99], [0.02, 0.35]], (3, 1): [[0.4, 0.31], [0.85, 0.53]], (1, 4): [[0.29, 0.34], [0.17, 0.19]], (0, 2): [[0.53, 0.32], [0.58, 0.24]], (2, 0): [[0.67, 0.42], [0.73, 0.62]], (2, 6): [[0.91, 0.69], [0.54, 0.43]], (6, 2): [[0.62, 0.31], [0.29, 0.92]], (2, 5): [[0.24, 0.46], [0.53, 0.11]], (5, 2): [[0.11, 0.94], [0.47, 0.46]], (1, 0): [[0.54, 0.37], [0.16, 0.57]]}

M20_96_3.75_.984375_414105_181.922 = {(0, 1): [[0.96, 0.29], [0.12, 0.55]], (1, 3): [[0.32, 0.76], [0.19, 0.87]], (4, 1): [[0.34, 0.47], [0.27, 0.13]], (3, 1): [[0.75, 0.57], [0.16, 0.08]], (1, 4): [[0.26, 0.03], [0.73, 0.38]], (0, 2): [[0.07, 0.92], [0.1, 0.64]], (2, 0): [[0.2, 0.1], [0.04, 0.49]], (2, 6): [[0.23, 0.57], [0.76, 0.47]], (6, 2): [[0.46, 0.08], [0.29, 0.41]], (2, 5): [[0.81, 0.24], [0.55, 0.41]], (5, 2): [[0.46, 0.81], [0.3, 0.97]], (1, 0): [[0.9, 0.93], [0.76, 0.64]]}

M21_13_3.25_.9843_414105_99.951 = {(0, 1): [[0.23, 0.92], [0.1, 0.62]], (1, 3): [[0.47, 0.31], [0.12, 0.12]], (4, 1): [[0.49, 0.09], [0.69, 0.88]], (3, 1): [[0.89, 0.52], [0.41, 0.28]], (1, 4): [[0.94, 0.51], [0.95, 0.75]], (0, 2): [[0.36, 0.76], [0.71, 0.61]], (2, 0): [[0.94, 0.13], [0.82, 0.63]], (2, 6): [[0.11, 0.67], [0.66, 0.24]], (6, 2): [[0.91, 0.06], [0.82, 0.16]], (2, 5): [[0.73, 0.22], [0.21, 0.03]], (5, 2): [[0.38, 0.65], [0.53, 0.55]], (1, 0): [[0.65, 0.5], [0.2, 0.1]]}

M22_24_3.75_.9218_414105_169.226 = {(0, 1): [[0.24, 0.57], [0.28, 0.36]], (1, 3): [[0.3, 0.89], [0.82, 0.93]], (4, 1): [[0.94, 0.57], [0.36, 0.27]], (3, 1): [[0.55, 0.87], [0.1, 0.93]], (1, 4): [[0.01, 0.0], [0.07, 0.91]], (0, 2): [[0.21, 0.2], [0.75, 0.31]], (2, 0): [[0.88, 0.85], [0.36, 0.62]], (2, 6): [[0.15, 0.51], [0.25, 0.87]], (6, 2): [[0.7, 0.87], [0.65, 0.66]], (2, 5): [[0.96, 0.59], [0.3, 0.97]], (5, 2): [[0.18, 0.16], [0.15, 0.71]], (1, 0): [[0.88, 0.65], [0.92, 0.64]]}

M23_35_3.5_.9687_414105_238.550 = {(0, 1): [[0.05, 0.55], [0.07, 0.18]], (1, 3): [[0.9, 0.03], [0.68, 0.35]], (4, 1): [[0.11, 0.69], [0.04, 0.23]], (3, 1): [[0.5, 0.62], [0.99, 0.28]], (1, 4): [[0.35, 0.33], [0.63, 0.18]], (0, 2): [[0.44, 0.12], [0.05, 0.0]], (2, 0): [[0.49, 0.9], [0.97, 0.14]], (2, 6): [[0.46, 0.1], [0.35, 0.81]], (6, 2): [[0.3, 0.57], [0.63, 0.31]], (2, 5): [[0.13, 0.27], [0.4, 0.08]], (5, 2): [[0.8, 0.74], [0.51, 0.43]], (1, 0): [[0.49, 0.19], [0.11, 0.95]]}

M24_9_3.25_1.109_414105_2177.491 = {(0, 1): [[0.97, 0.08], [0.51, 0.19]], (1, 3): [[0.94, 0.32], [0.73, 0.53]], (4, 1): [[0.95, 0.77], [0.76, 0.7]], (3, 1): [[0.35, 0.46], [0.92, 0.1]], (1, 4): [[0.61, 0.02], [0.29, 0.28]], (0, 2): [[0.04, 0.91], [0.21, 0.25]], (2, 0): [[0.94, 0.98], [0.9, 0.21]], (2, 6): [[0.22, 0.98], [0.31, 0.06]], (6, 2): [[0.59, 0.65], [0.19, 0.41]], (2, 5): [[0.03, 0.52], [0.35, 0.76]], (5, 2): [[0.04, 0.36], [0.51, 0.05]], (1, 0): [[0.83, 0.23], [0.61, 0.88]]}


'''