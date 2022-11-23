# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:47:47 2022

@author: maxfu
"""

"""
This code generates the data that will later be used by the code in the learning process.

The code first defines a function with all the derivatives for Theta_1, Theta_2 and their derivatives.
The second function defines the initial conditions and solves the differential equations defined in the first function. 
"""

import numpy as np
import pickle
import gzip
from . import io_nn as io
from scipy.integrate import odeint

#########################################################################################################################
    
def derivatives(sol,t,L1,L2,m1,m2):
    
    g=9.81
    Theta_1,Theta_1_dot,Theta_2,Theta_2_dot = sol
    
    C1=np.cos(Theta_1-Theta_2)
    C2=np.sin(Theta_1-Theta_2)
    
    # The first derivative of Theta is named Z, and so the second derivative is named Z_dot
    
    Z_1=Theta_1_dot
    Z_2=Theta_2_dot
    Z_1_dot=(m2*g*np.sin(Theta_2)*C1-m2*C2*(L1*Z_1**2*C1+L2*Z_2**2)-(m1+m2)*g*np.sin(Theta_1))/(L1*(m1+m2*C2**2))
    Z_2_dot=((m1+m2)*(L1*Z_1**2*C2-g*np.sin(Theta_2)+g*np.sin(Theta_1)*C1)+m2*L2*Z_2**2*C2*C1)/(L2*(m1+m2*C2**2))
    
    return Z_1,Z_1_dot,Z_2,Z_2_dot
    
#########################################################################################################################

def generate_data(N, tmax, t_int, L1_int=[1,4], L2_int=[1,4], m1_int=[1,4], m2_int=[1,4], Th1_i_int=[0,2*np.pi], Th2_i_int=[0,2*np.pi], fileName=None):
       
    """
    N -> Number of runs of the code.
    t_max -> maximum value of the time.
    t_int -> number of time steps desired.
    L1 and L2 -> lengths of the rods of the double pendulum.
    m1 and m2 -> masses of the balls at the end of each rod.
    Th1_i and Th2_i -> initial values of the angles, only used for the first step.
    
    x1,x2,y1,y2 -> Arrays containing the cartesian coordinates for the double pendulum after every step.
    t -> the time variable, starts at 0 and goes to t_max, with t_int evenly spaced steps.
    Theta_1, Theta_2 -> Arrays containing the values of the angles after every step.
    """
    
    L1,L2,m1,m2,Th1_i,Th2_i=[],[],[],[],[],[]
    
    for i in range(N):
                
        L1.append(np.random.uniform(L1_int[0],L1_int[1]))
        L2.append(np.random.uniform(L2_int[0],L2_int[1]))
        m1.append(np.random.uniform(m1_int[0],m1_int[1]))
        m2.append(np.random.uniform(m2_int[0],m2_int[1]))
        Th1_i.append(np.random.uniform(Th1_i_int[0],Th1_i_int[1]))
        Th2_i.append(np.random.uniform(Th2_i_int[0],Th2_i_int[1]))
        
    # We now initialize the cartesian coordinate and the angle lists and define the time interval.
       
    x1,x2,y1,y2,th1,th2=[],[],[],[],[],[]
    
    t = np.linspace(0,tmax,t_int)
    
    # We will solve our differential equations N times with the given initial conditions.
    
    for i in range(N):
        
        # We create a set of lists for the cartesian coordinates that we will use only for a given set of initial 
        # conditions. We label them with the subindex t, meaning temporal. They are reset every time.
        
        x1_t,x2_t,y1_t,y2_t=[],[],[],[]
        
        sol_i = np.array([Th1_i[i],0,Th2_i[i],0])
        sol = odeint(derivatives, sol_i, t, args=(L1[i],L2[i],m1[i],m2[i]))
        
        Theta_1,Theta_2 = sol[:,0],sol[:,2]
        
        for j in range(len(Theta_1)):
            x1_t.append(L1[i]*np.sin(Theta_1[j]))
            x2_t.append(L1[i]*np.sin(Theta_1[j])+L2[i]*np.sin(Theta_2[j]))
            y1_t.append(-L1[i]*np.cos(Theta_1[j]))
            y2_t.append(-L1[i]*np.cos(Theta_1[j])-L2[i]*np.cos(Theta_2[j])) 
            
        x1.append(x1_t),x2.append(x2_t),y1.append(y1_t),y2.append(y2_t)
        th1.append(Theta_1),th2.append(Theta_2)
        
            
    x1,x2,y1,y2=np.array(x1),np.array(x2),np.array(y1),np.array(y2)
    th1,th2=np.array(th1),np.array(th2)
    
    data = np.dstack([x1,x2,y1,y2])
    states = np.dstack([th1,th2])
    result=(data,states)
    if fileName is not None:
        f = gzip.open(io.data_path + fileName + ".plk.gz", 'wb')
        pickle.dump(result, f, protocol=2)
        f.close()
    return (result)
        
    
    
    
    
