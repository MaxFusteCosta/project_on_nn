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

def derivatives(y,t,L1,L2,m1,m2):
    
    g=9.81
    Theta_1,Theta_1_dot,Theta_2,Theta_2_dot = y
    
    C1=np.cos(Theta_1-Theta_2)
    C2=np.sin(Theta_1-Theta_2)
    
    # The first derivative of Theta is named Z, and so the second derivative is named Z_dot
    
    Z_1=Theta_1_dot
    Z_2=Theta_2_dot
    Z_1_dot=(m2*g*np.sin(Theta_2)*C1-m2*C2*(L1*Z_1**2*C1+L2*Z_2**2)-(m1+m2)*g*np.sin(Theta_1))/(L1*(m1+m2*C2**2))
    Z_2_dot=((m1+m2)*(L1*Z_1**2*C2-g*np.sin(Theta_2)+g*np.sin(Theta_1)*C1)+m2*L2*Z_2**2*C2*C1)/(L2*(m1+m2*C2**2))
    
    return Z_1,Z_1_dot,Z_2,Z_2_dot

def generate_data(N,t_sample=np.linspace(0, 5, 50),t_s_int=None, fileName=None):
    
    # First we define the intervals that the initial conditions can take. We will use these values to randomly select the initial conditions.
    
    L1_int,L2_int,m1_int,m2_int=[1,4],[1,4],[1,4],[1,4]
    Th1_i_int,Th2_i_int=[0,2*np.pi],[0,2*np.pi]
    
    L1_i=(L1_int[1]-L1_int[0])*np.random.rand(N)+L1_int[0]
    L2_i=(L2_int[1]-L2_int[0])*np.random.rand(N)+L2_int[0]
    m1_i=(m1_int[1]-m1_int[0])*np.random.rand(N)+m1_int[0]
    m2_i=(m2_int[1]-m2_int[0])*np.random.rand(N)+m2_int[0]
    Th1_i=(Th1_i_int[1]-Th1_i_int[0])*np.random.rand(N)+Th1_i_int[0]
    Th2_i=(Th2_i_int[1]-Th2_i_int[0])*np.random.rand(N)+Th2_i_int[0]
    
    x1,x2,y1,y2=[],[],[],[]
    
    if t_s_int is None:
        t_s_int = [t_sample[0], 2 * t_sample[-1]]
    t_s = np.reshape(np.random.rand(N) * (t_s_int[1] - t_s_int[0]) + t_s_int[0], [N, 1])
    
    for L1,L2,m1,m2,t in zip(L1_i,L2_i,m1_i,m2_i,t_s):
        
        # We now solve the differential equations that we defined in the "derivatives" function.
        
        sol_i = np.array([Th1_i,0,Th2_i,0])
        sol = odeint(derivatives, sol_i, t_sample, args=(L1,L2,m1,m2))
        
        # We extract only the angles and we use them to define the cartesian coordinates.
        
        Theta_1,Theta_2 = sol[:,0],sol[:,2]
                
        x1.append(L1*np.sin(Theta_1))
        x2.append(L1*np.sin(Theta_1)+L2*np.sin(Theta_2))
        y1.append(-L1*np.cos(Theta_1))
        y2.append(-L1*np.cos(Theta_1)-L2*np.cos(Theta_2))
        
    x1 = np.array(x1)
    x1 = np.reshape(x1,[N,1])
    x2 = np.array(x2)
    x2 = np.reshape(x2,[N,1])
    y1 = np.array(y1)
    y1 = np.reshape(y1,[N,1])
    y2 = np.array(y2)
    y2 = np.reshape(y2,[N,1])
    
    # We are only interested in two sets of variables: the angles and the cartesian coordinates.
    
    data = np.dstack([x1,x2,y1,y2])
    states = np.dstack([Theta_1,Theta_2])
    result=(data,states)
    if fileName is not None:
        f = gzip.open(io.data_path + fileName + ".plk.gz", 'wb')
        pickle.dump(result, f, protocol=2)
        f.close()
    return (result)
        
    
    
    
    