# to calculate the concentration of genes
def conc_act(t0, t_end, dt,betas, alphas,nodes, kappa, n_2gene, t_ext, t_act,dependency,dep):
# number of interval according to time variables
N = int((t_end - t0) / dt + 1)
# a node matrice, initially zero
node_matrice = np.zeros((N, len(nodes)))
# a zero array for initial concentration
c_nodes = [0] * len(nodes)
# initialization the array that indicates the change of nodes over time dN_dt = [0] * len(nodes)
# time array
t_ax = np.arange(t0, t_end, dt)
print(len(t_ax))
# calculates the new concentration according to time
new_n=[0] * (len(nodes))
# calculates the new concentration according to time
for index, t in enumerate(t_ax):
# d
d=0 for #
to access dependency genes of the gene
i in range(len(nodes)):
genes that has no dependency
if (dependency[i]==0):
# change of transcription factor over time new_n[i] = betas[i] * ((t >= t_act[i]) & (t <=
t_ext[i])).astype(float)
# dN_dt[i] = betas[i] * ((t >= t_act[i]) & (t <=
t_ext[i])).astype(float) - alphas[i] * c_nodes[i]
        # for and operation
elif (dependency[i]==1):
dN_dt[i]=betas[i] * (c_nodes[dep[d][0]]**n_2gene)/(kappa**n_2gene
+ c_nodes[dep[d][0]]**n_2gene)*((c_nodes[dep[d][1]]**n_2gene)/(kappa**n_2gene + c_nodes[dep[d][1]]**n_2gene)) - alphas[i] * c_nodes[i]
d+=1
new_n[i] = c_nodes[i] + dN_dt[i] * dt
        # for or operation
elif (dependency[i]==2): dN_dt[i]=betas[i] *
max(((c_nodes[dep[d][0]]**n_2gene)/(kappa**n_2gene + c_nodes[dep[d][0]]**n_2gene)),(((c_nodes[dep[d][1]]**n_2gene)/(kappa**n_2ge ne + c_nodes[dep[d][1]]**n_2gene)))) - alphas[i] * c_nodes[i]
d+=1
new_n[i] = c_nodes[i] + dN_dt[i] * dt
 # new concentration value for activator #new_n[i] = c_nodes[i] + dN_dt[i] * dt
# assign the new value to the output matrice node_matrice[:,i][index] = new_n[i]
# to calculate cumulatively
c_nodes[i] = new_n[i]
#gives neew concentration values as outputs
return node_matrice
 # necessary libraries
import numpy as np
import matplotlib.pyplot as plt import itertools
import matplotlib as mpl
 kappa=0.5 # kappa value n_2gene=10 # n order
 # to define all combinations
l = [0.0, 1.0] l2=list(itertools.product(l, repeat=3)) print(l2)
 nodes = ['X1', 'X2', 'X3', 'Y', 'Z'] # e.g. Y-
 >Z
comb=7
combination
dependency=[0,0,0,1,2]
initial gene, 1 for and operation, 2 for or operation dep=[[0,1],[2,3]]
determine which genes a gene is linked to
betas = [l2[comb][0], l2[comb][1], l2[comb][2], 1.0, 1.0] of beta in node order
alphas = [1.0, 1.0, 1.0, 1.0, 1.0 ]
of alpha in node order
# type of
# 0 for
# to
# values
# Values
t0 = -1
dt = 0.01
t_end=10
t_ext=[10,10,10,10,10]
t_act=[0,0,0,0,0]
# time before activation
# time interval
# end time
# for extra time interventions
# activation time
# to calculate the concentrations
outs= conc_act(t0,t_end,dt,betas,alphas,nodes,kappa,n_2gene,t_ext,t_act,dependenc y,dep)

#for high resolution graphs
mpl.rcParams['figure.dpi'] = 250
# number of interval according to time variables N = int((t_end - t0) / dt +1) #
# figure size according to number of figures fig = plt.figure(figsize=(10,len(outs[1])*3))
# loop for creating figures dinamically
for j in range(len(outs[1])):
# create figures according to number of nodes
ax = fig.add_subplot(len(outs[1]),1,j+1)
# plot(t,nodes)
ax.plot(np.linspace(t0, t_end, N),outs[:,j],'b',label=nodes[j]) #legend position
ax.legend(loc="upper right")
# add x label
ax.set_xlabel('t')
# add y label
ax.set_ylabel('Concentration of {}'.format(nodes[j]))
# grid on
ax.grid()