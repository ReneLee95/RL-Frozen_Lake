import numpy as np 

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    delta = 0
    while 1:                                                                       
        delta = 0                                                                  
        for state in range(env.nS):
            Bellman = 0
            for Action, AProb in enumerate(policy[state]):
                for MDPprob, Nstate, reward in env.MDP[state][Action]:
                    Bellman += (AProb * MDPprob) * (reward + (gamma * V[Nstate]))

            delta = max(delta, np.abs(V[state]-Bellman))
            V[state] = Bellman
            
        if delta < theta:
            break

    return V

def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros([env.nS, env.nA]) / env.nA                                   
    totalarrow = 4
    for state in range(env.nS):
        maxQ = np.zeros(totalarrow)
        for arrowaction in range(totalarrow):
            for MDPprob, Nstate, reward in env.MDP[state][arrowaction]:
                maxQ[arrowaction] += MDPprob * (reward + gamma * V[Nstate])
        q = maxQ
        
        bestAction = np.argmax(q==np.max(q)).flatten()
        for i in bestAction:
            primeV = np.eye(env.nA)[i]
            
        policy[state] = primeV
    return policy


def policy_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA                                    
    totalarrow = 4
    while 1:                                                                      
        delta = 0                                                                  
        for state in range(env.nS):
            preV = V[state]
            
            maxQ = np.zeros(totalarrow)
            for arrowaction in range(totalarrow):
                for MDPprob, Nstate, reward in env.MDP[state][arrowaction]:
                    maxQ[arrowaction] += MDPprob * (reward + gamma * V[Nstate])
            V[state] = max(maxQ)
            delta = max(delta, abs(preV - V[state]))
            
        if delta < theta:
            break
    policy = policy_improvement(env, V, gamma)                                    
        
    return policy, V

def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA
    totalarrow = 4
    
    while 1:
        delta = 0
        for state in range(env.nS):
            preV = V[state]
            
            maxQ = np.zeros(totalarrow)
            for arrowaction in range(totalarrow):
                for MDPprob, Nstate, reward in env.MDP[state][arrowaction]:
                    maxQ[arrowaction] += MDPprob * (reward + gamma * V[Nstate])
            q = maxQ
            
            V[state] = max(q)
            delta = max(delta, abs(preV - V[state]))
            
        if delta < theta:
            break

    policy = policy_improvement(env, V, gamma)
    
    return policy, V