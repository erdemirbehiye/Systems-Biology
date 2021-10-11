# necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# to calculate the concentration of genes
def conc_act(t0, t_end, dt,betas, alphas,nodes, kappa, n_2gene, t_ext, t_act):
    # number of interval according to time variables
    N = int((t_end - t0) / dt + 1)
    # a node matrice, initially zero
    node_matrice = np.zeros((N, len(nodes)))
    # a zero array for initial concentration
    c_nodes = [0] * len(nodes)
    # initialization the array that indicates the change of nodes over time 
    dN_dt = [0] * len(nodes)
    # time array
    t_ax = np.arange(t0, t_end + dt, dt)
    # calculates the new concentration according to time
    for index, t in enumerate(t_ax):
        # change of transcription factor over time
        dN_dt[0] = betas[0] * ((t >= t_act[0]) & (t <= t_ext[0])).astype(float) - alphas[0] * c_nodes[0]
        #for more than one node
        if len(nodes) > 1:
            # calculates concentrations of other nodes
            for i in range(len(nodes) - 1):
                # initial variable for next genes
                new_n=[0] * (len(nodes) - 1)
                # formula of the instant concentration of gene according to activator
                dN_dt[i + 1] = betas[i + 1] * (((t >= t_act[i + 1]) & (t<=t_ext[i+1])).astype(float) 
                                * c_nodes[i] ** n_2gene) / (kappa ** n_2gene + ((t >= t_act[i + 1]) 
                                & (t <= t_ext[i + 1])).astype(float) * c_nodes[i] ** n_2gene) - alphas[i + 1] * c_nodes[i + 1]
                # total concentration                                                          
                new_n[i] = c_nodes[i + 1] + dN_dt[i + 1] * dt
                # assign the new value to the output matrice
                node_matrice[:,i + 1][index] = new_n[i]
                # to calculate cumulatively
                c_nodes[i + 1] = new_n[i]
        # new concentration valur for activator
        new_act = c_nodes[0] + dN_dt[0] * dt
        # assign the new value to the output matrice
        node_matrice[:,0][index] = new_act  
        # to calculate cumulatively
        c_nodes[0] = new_act
    #gives neew concentration values as outputs
    return node_matrice


# for 1 < alpha, beta_prime, beta_mRNA < 2
alpha = np.random.rand() + 1.0
beta_prime = np.random.rand() + 1.0
beta_mRNA = np.random.rand() + 1.0
# for alpha_mRNA >> alpha, 20 is selected randomly
alpha_mRNA = 20.0 * np.random.rand()
beta=beta_prime * beta_mRNA / alpha_mRNA
steady_state=beta_mRNA/alpha_mRNA


nodes = ['Y_mRNA','Y']               # e.g. Y->Z
betas = [beta_mRNA, beta_prime]     # Values of beta in node order
alphas = [alpha_mRNA, alpha ]       # Values of alpha in node order
kappa=0.5                           # kappa value 
n_2gene=10                          # n order

t0 = -1                             # time before activation
t_end = 5                           # end time
dt = 0.01                           # time interval
t_act=[0,0]                         # activation time
t_ext=[t_end,t_end]                 # for extra time interventions

# to calculate the concentrations
outs= conc_act(t0,t_end,dt,betas,alphas,nodes,kappa,n_2gene,t_ext,t_act)


# number of interval according to time variables
N = int((t_end - t0) / dt +1) # 
# figure size according to number of figures
fig = plt.figure(figsize=(20,len(outs[1])*6)) 

# loop for creating figures dinamically
for j in range(len(outs[1])):
    # create figures according to number of nodes
    ax = fig.add_subplot(len(outs[1]),1,j+1)
    # plot(t,nodes)
    ax.plot(np.linspace(t0, t_end, N),outs[:,j],'b',label=nodes[j])
    # for steady state
    if j==0:
      ax.axhline(y=beta_mRNA/alpha_mRNA, color='r', linestyle='--',label='Steady State for Y_mRNA')
    #legend position
    ax.legend(loc="lower right")
    # add x label
    ax.set_xlabel('t')
    # add y label
    ax.set_ylabel('Concentration of {}'.format(nodes[j]))
    # grid on
    ax.grid()


nodes = ['Y']          # e.g. Y->Z
betas = [beta]         # Values of beta in node order
alphas = [alpha]       # Values of alpha in node order

t0 = -1                # time before activation
t_end = 5              # end time
dt = 0.01              # time interval
t_act=[0]              # activation time
t_ext=[t_end]          # 

# to calculate the concentrations
outs2= conc_act(t0,t_end,dt,betas,alphas,nodes,kappa,n_2gene,t_ext,t_act)

# figure size according to number of figures
fig = plt.figure(figsize=(20,len(outs2[1])*6)) 

# loop for creating figures dinamically
for j in range(len(outs2[1])):
    # create figures according to number of nodes
    ax = fig.add_subplot(len(outs2[1]),1,j+1)
    # plot(t,nodes)
    ax.plot(np.linspace(t0, t_end, N),outs2[:,j],'b',label=nodes[j])
    ax.legend(loc="upper left")
    # add x label
    ax.set_xlabel('t')
    # add y label
    ax.set_ylabel('Concentration of {}'.format(nodes[j]))
    # grid on
    ax.grid()


# figure size
fig = plt.figure(figsize=(20,6)) 

# create figures according to number of nodes
ax = fig.add_subplot(len(outs2[1]),1,1)
# plot(t,nodes)
ax.plot(np.linspace(t0, t_end, N),outs[:,1],'--g', label='Y')
ax.plot(np.linspace(t0, t_end, N),outs2[:,0],'r',label='Maximal Y')
ax.legend(loc="upper left")
# add x label
ax.set_xlabel('t')
# add y label
ax.set_ylabel('Concentration of {}'.format(nodes[j]))
ax.set_yscale('log')
# grid on
ax.grid()


nodes = ['Y', 'Z']          # e.g. Y->Z
betas = [beta,beta]         # Values of beta in node order
alphas = [alpha,alpha]       # Values of alpha in node order

t0 = -1                # time before activation
t_end = 10              # end time
dt = 0.01              # time interval
t_act=[0,0]              # activation time
t_ext=[t_end,5]          # 
# to calculate the concentrations
outs3= conc_act(t0,t_end,dt,betas,alphas,nodes,kappa,n_2gene,t_ext,t_act)


# figure size according to number of figures
fig = plt.figure(figsize=(20,len(outs2[1])*6)) 
N = int((t_end - t0) / dt +1) # 
# loop for creating figures dinamically
for j in range(len(outs3[1])):
    # create figures according to number of nodes
    ax = fig.add_subplot(len(outs3[1]),1,j+1)
    # plot(t,nodes)
    ax.plot(np.linspace(t0, t_end, N),outs3[:,j],'b',label=nodes[j])
    ax.legend(loc="upper left")
    # add x label
    ax.set_xlabel('t')
    # add y label
    ax.set_ylabel('Concentration of {}'.format(nodes[j]))
    # grid on
    ax.grid()