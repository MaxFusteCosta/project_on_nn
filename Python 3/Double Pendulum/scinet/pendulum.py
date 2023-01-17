# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:47:47 2022

@author: Max Fusté Costa
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

def generate_data(N, t_max, t_int, L1_int=[1,4], L2_int=[1,4], m1_int=[1,4], m2_int=[1,4], Theta_1_ini_int=[0,0], Theta_2_ini_int=[np.pi/8,np.pi/8], show=False, fileName=None):
       
    """
    N -> Number of runs of the code.
    t_max -> maximum value of the time.
    t_int -> number of time steps desired.
    L1 and L2 -> lengths of the rods of the double pendulum.
    m1 and m2 -> masses of the balls at the end of each rod.
    Th1_i and Th2_i -> initial values of the angles, only used for the first step, they are fixed. The first one is fixed to be 0 and the second
    one is fixed to be pi/8.
    
    x1,x2,y1,y2 -> Arrays containing the cartesian coordinates for the double pendulum after every step.
    t -> the time variable, starts at 0 and goes to t_max, with t_int evenly spaced steps.
    Theta_1, Theta_2 -> Arrays containing the values of the angles after every step.
    """
    
    L1,L2,m1,m2,Theta_1_ini,Theta_2_ini=[],[],[],[],[],[]
    
    # We want to make a total of N measures, so we need N sets of initial conditions. Each set of L, m and initial angles determines a measure.
    
    for i in range(N):
                
        L1.append(np.random.uniform(L1_int[0],L1_int[1]))
        L2.append(np.random.uniform(L2_int[0],L2_int[1]))
        m1.append(np.random.uniform(m1_int[0],m1_int[1]))
        m2.append(np.random.uniform(m2_int[0],m2_int[1]))
        Theta_1_ini.append(np.random.uniform(Theta_1_ini_int[0],Theta_1_ini_int[1]))
        Theta_2_ini.append(np.random.uniform(Theta_2_ini_int[0],Theta_2_ini_int[1]))
        
    # We initialize the angle lists and define the two time intervals. The first one is for the training itself, the second one
    # is for the prediction
       
    th1,th2=[],[]
    th1_pred,th2_pred=[],[]
    
    # We also initialize the cartesian coordinates.
    
#    x1,x2,y1,y2=[],[],[],[]
#    x1_pred,x2_pred,y1_pred,y2_pred=[],[],[],[]
    
    # These are the time steps for each measure. We want N measures, each with this amount of time steps in it.
    
    t = np.linspace(0,t_max,t_int)
    
    # This is the predicted time. Here, we have way more time steps because of the way this is generated, but we are only interested in t time steps.
    # We generate these random values because it gives the NN a harder time.
    
    t_pred_int = [0,2*t_max]
    t_pred = np.random.rand(N) * (t_pred_int[1] - t_pred_int[0]) + t_pred_int[0]
    t_pred.sort()
    
    # We will solve our differential equations N times with the given initial conditions.       
        
    for i in range(N):
        
        sol_i = np.array([Theta_1_ini[i],0,Theta_2_ini[i],0])
        sol = odeint(derivatives, sol_i, t, args=(L1[i],L2[i],m1[i],m2[i]))
        sol_i_pred = np.array([Theta_1_ini[i],0,Theta_2_ini[i],0])
        sol_pred = odeint(derivatives, sol_i_pred, t_pred, args=(L1[i],L2[i],m1[i],m2[i]))
        
        Theta_1,Theta_2 = sol[:,0],sol[:,2]
        Theta_1_pred,Theta_2_pred = sol_pred[:,0],sol_pred[:,2]
        # We append the whole Theta_1 and Theta_2 into the list because it is the full time evolution of a measure with a certain set of initial conditions.
        # In total, we will end up with N measures with t_int values for the angles in each of them.
        
        th1.append(Theta_1),th2.append(Theta_2)       
        th1_pred.append(Theta_1_pred[i]),th2_pred.append(Theta_2_pred[i])
        """
        x1_temp,x2_temp,y1_temp,y2_temp=[],[],[],[]
        x1_temp_pred,x2_temp_pred,y1_temp_pred,y2_temp_pred=[],[],[],[]
        
        for j in range(len(Theta_1)):
            
            x1_temp.append(L1[i]*np.sin(Theta_1[j]))
            x2_temp.append(L1[i]*np.sin(Theta_1[j])+L2[i]*np.sin(Theta_2[j]))
            y1_temp.append(-L1[i]*np.cos(Theta_1[j]))
            y2_temp.append(-L1[i]*np.cos(Theta_1[j])-L2[i]*np.cos(Theta_2[j]))
            
            x1_temp_pred.append(L1[i]*np.sin(Theta_1_pred[j]))
            x2_temp_pred.append(L1[i]*np.sin(Theta_1_pred[j])+L2[i]*np.sin(Theta_2_pred[j]))
            y1_temp_pred.append(-L1[i]*np.cos(Theta_1_pred[j]))
            y2_temp_pred.append(-L1[i]*np.cos(Theta_1_pred[j])-L2[i]*np.cos(Theta_2_pred[j]))
            
        x2_temp=np.array(x2_temp)
        x2_temp_pred=np.array(x2_temp_pred)
        x2.append(x2_temp)
        
        ratio = N/t_int
        if i<ratio:
            x2_pred.append(x2_temp_pred)"""
      
        if show is not False:
            if i%(N/10)==0:
                print('Calculations are at',100*i/N,'%')

# This is the new part.

    t_pred=np.reshape(t_pred,[N,1])
    th2 = np.array(th2)
    th2_pred = np.reshape(th2_pred, [N,1])     
    states = np.vstack([L1,L2,m1,m2]).T
#    states = np.vstack([th1,th2]).T
    result = ([th2,t_pred,th2_pred],states, [])
    """
    
# This works so anything bad happens just restore this.

    t_pred=np.reshape(t_pred,[N,1])
    th1,th2=np.array(th1),np.array(th2)
    th1_pred,th2_pred=np.reshape(th1_pred, [N,1]),np.reshape(th2_pred, [N,1])
    states = np.vstack([L1,L2,m1,m2]).T
#    result = ([th1, th2, t_pred, th1_pred, th2_pred], states, [])
    result = ([th2,t_pred,th2_pred],states, [])"""
    
    if fileName is not None:
        f = gzip.open(io.data_path + fileName + ".plk.gz", 'wb')
        pickle.dump(result, f, protocol=2)
        f.close()
    return ('Data generation complete')    
        
    
    
    
    
