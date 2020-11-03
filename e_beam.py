#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.integrate import odeint
obj_path = '/home/guus/Documents/python_objects'
import numpy as np
import sys
sys.path.insert(1,obj_path)
from const import constants
cons = constants()
"""
Created on Mon Oct 12 15:15:03 2020

@author: Guus Reefman
This is an attempt to make a Monte Carlo model of SEE in python
Since this is something that is planned to be computationally expensive I plan
to convert this into more efficient C code but since I am a novice in C it is 
best to start small and work with a more understandable language such as 
python.

This is supposed to be relatively realistic and thus I start with the e-beam 
that will include space charge and such so working distance will be a parameter
of importance as well.

FYI: everything will be in SI units to avoid confusion.
"""

"""
Starting with the e-beam initiation
"""
#What you want to plot
plot_ini_pos = 0 #initial stuff
plot_gamma_error = 0 #to show if there is an error when you don't correct
plot_traj = 0

#Introducing constants
w_d     = 2e-2        #working distance
E_k0    = 300*cons.q_e #initial primary e-beam energy
sigma_x = 0.5e-3 #variance in x-direction
sigma_y = sigma_x
a = cons.m_e*cons.c

#Some calculated constants
gamma = np.divide(E_k0,cons.m_e*cons.c*cons.c) + 1 #classic gamma
beta_z = np.sqrt(1-1/(gamma*gamma))
k = np.divide(cons.q_e,4*np.pi*cons.eps_0)

#here we do some tinkering to understand stuff
p1 = [2,0,0]; p2 = [50e-9,0,0];
p3 = [-50e-9,0,0]; p4 = [-2,0,0];
points_0 = (p2,p3)

def Efield_norm(points):
    Efield_ph = np.zeros( (len(points),len(points[0])) )
    Efield = np.zeros( (len(points),len(points[0])) )
    for j in range(len(points)):
        for i in range(len(points)):
            r = np.subtract(points[j],points[i])
            if i == j:
                Efield_ph[i] = [0, 0, 0]
            else:
                Efield_ph[i] = np.divide(r,np.sum((r*r)**(3/2)))
            #Just in case there is a NaN in this matrix
            Efield_ph[np.isnan(Efield_ph)] = 0
            
        Efield[j,:] = [np.sum(Efield_ph[:,0]), np.sum(Efield_ph[:,1]), np.sum(Efield_ph[:,2])]
    return Efield

"""
First I will try and use a simple backward Euler scheme

UPDATE: first do x and y directions only plus 2 particles, 
then we will generalize
Lets check with an analytical solution as well
"""
tf = 1e-10; dt = 1e-16
t = np.arange(0,tf,dt)
N = len(t)
x0 = np.array(points_0)[0,0]


p_x = np.zeros((N,len(points_0))); p_y = np.zeros((N,len(points_0))); p_z = np.zeros((N,len(points_0))); 
x = np.zeros((N,len(points_0))); y = np.zeros((N,len(points_0))); z = np.zeros((N,len(points_0)));

x[0,0] = np.array(points_0)[0,0]; x[0,1] = np.array(points_0)[1,0];
y[0,0] = np.array(points_0)[0,1]; y[0,1] = np.array(points_0)[1,1];
z[0,0] = np.array(points_0)[0,2]; z[0,1] = np.array(points_0)[1,2];
p_x[0,0] = 0; p_x[0,1] = 0;
p_y[0,0] = 0; p_y[0,1] = 0;
p_z[0,0] = gamma*cons.c*beta_z*cons.m_e; p_z[0,1] = gamma*cons.c*beta_z*cons.m_e;


for i in range(1,N):
    #Calculate force
    points = ([x[i-1,0],y[i-1,0],z[i-1,0]], [x[i-1,1],y[i-1,1],z[i-1,1]])
    F = cons.q_e*k*Efield_norm(points)
    
    p_x[i,0] = p_x[i-1,0] + F[0,0]*dt
    p_y[i,0] = p_y[i-1,0] + F[0,1]*dt
    p_z[i,0] = p_z[i-1,0] + F[0,2]*dt
    x[i,0] = x[i-1,0] + np.divide(cons.c*p_x[i-1,0], np.sqrt(p_x[i-1,0]**2 + (a)**2))*dt
    y[i,0] = y[i-1,0] + np.divide(cons.c*p_y[i-1,0], np.sqrt(p_y[i-1,0]**2 + (a)**2))*dt
    z[i,0] = z[i-1,0] + np.divide(cons.c*p_z[i-1,0], np.sqrt(p_z[i-1,0]**2 + (a)**2))*dt

    p_x[i,1] = p_x[i-1,1] + F[1,0]*dt
    p_y[i,1] = p_y[i-1,1] + F[1,1]*dt
    p_z[i,1] = p_z[i-1,1] + F[1,2]*dt
    x[i,1] = x[i-1,1] + np.divide(cons.c*p_x[i-1,1], np.sqrt(p_x[i-1,1]**2 + (a)**2))*dt
    y[i,1] = y[i-1,1] + np.divide(cons.c*p_y[i-1,1], np.sqrt(p_y[i-1,1]**2 + (a)**2))*dt
    z[i,1] = z[i-1,1] + np.divide(cons.c*p_z[i-1,1], np.sqrt(p_z[i-1,1]**2 + (a)**2))*dt

A = x[:,0]/x0
t0 = np.sqrt(np.divide(8*np.pi*cons.eps_0*cons.m_e*(x0**3),cons.q_e*cons.q_e))
t_an = t0*(np.sqrt((A-1)*A) + np.log((A-1) + np.sqrt(A)))

#calculate error before gamma correction
error_0 = abs(t - t_an)
t_an = t_an/gamma*gamma
error_1 = abs(t - t_an)

fig = plt.figure(figsize=(15,15))
plt.plot(x[:,0], t, color='b',label='numerical')
plt.plot(x[:,0], t_an, color='r',label='analytical')
plt.ylim([0, max(t)])
plt.xlabel('x position (m)')
plt.ylabel('time (s)') 
plt.legend(loc='best')

if plot_traj == True:
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x[:,0], y[:,0], z[:,0])
    ax.scatter3D(x[:,1], y[:,1], z[:,1]) 
    ax.set_xlabel('x position (m)')
    ax.set_ylabel('y position (m)')
    ax.set_zlabel('z position (m)')

if plot_gamma_error == True:
    fig = plt.figure(figsize=(15,15))
    plt.plot(x[:,0], error_0, color='b',label=r'no $\gamma$ correction')
    plt.plot(x[:,0], error_1, color='r',label=r'$\gamma$ correction')
    plt.title('Error between numerical and analytical expressions, dt = %.1E' %dt)
    plt.xlabel('x position (m)')
    plt.ylabel(r'error (|$t$ - $t_{an}$|)') 
    plt.legend(loc='best')

if plot_ini_pos == True:
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p1[0],p1[1],p1[2],marker='o')
    ax.scatter(p2[0],p2[1],p2[2],marker='o')
    ax.scatter(p3[0],p3[1],p3[2],marker='o')
    ax.scatter(p4[0],p4[1],p4[2],marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')









