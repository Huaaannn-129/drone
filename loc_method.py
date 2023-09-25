# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 01:19:51 2022

@author: ren tsai
"""
import numpy as np
from scipy import optimize
import sys, collections, time
from scipy.optimize import lsq_linear, root, minimize
import random
# import matplotlib.pyplot as plt
import numpy.matlib 
from itertools import product
from itertools import combinations
from collections import Counter
import numpy as np
# import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import heapq
import random 
from sympy import *
import cmath
import math



def lsq_method(distances_to_anchors, anchor_positions, u):
    # distances_to_anchors, anchor_positions = np.array(distances_to_anchors), np.array(anchor_positions)
    # if not np.all(distances_to_anchors):
    #     raise ValueError('Bad uwb connection. distances_to_anchors must never be zero. ' + str(distances_to_anchors))
    anchor_offset = anchor_positions[0]
    anchor_positions = anchor_positions[1:] - anchor_offset
    K = np.sum(np.square(anchor_positions), axis=1)   #ax=1 列加
    squared_distances_to_anchors = np.square(distances_to_anchors)
    squared_distances_to_anchors = (squared_distances_to_anchors - squared_distances_to_anchors[0])[1:]
    b = (K - squared_distances_to_anchors) / 2.
    det = u.T @ b
    #res = lsq_linear(anchor_positions, b, lsmr_tol='auto', verbose=0)
    #res = np.dot(np.dot(np.linalg.inv(np.dot(anchor_positions.T, anchor_positions)),(anchor_positions.T)), b)
    res = np.linalg.lstsq(anchor_positions, b, rcond=None)[0]
    return res + anchor_offset, det, b

from sympy import symbols, expand
from sympy.parsing.sympy_parser import parse_expr
from collections import OrderedDict

def extract_coefficients(expression):
    z = symbols('z')
    expanded_expr = expand((expression))
    # print('expanded_expr', expanded_expr)
    coefficients = Poly(expanded_expr, z).all_coeffs()
    # print('coefficients', coefficients)

    return coefficients

def Cardano(a,b,c,d):
    
    complex_num = (-1+cmath.sqrt(3)*1j)/2
    complex_num_2 = complex_num**2
    z0=b/3/a
    a2,b2 = a*a,b*b
    p=-b2/3/a2 +c/a
    q=(b/27*(2*b2/a2-9*c/a)+d)/a
    D=-4*p*p*p-27*q*q
    delta = 18 * a * b * c * d - 4 * b**3 * d + b**2 * c**2 - 4 * a * c**3 - 27 * a**2 * d**2
    print('delta', delta)
    r=cmath.sqrt(-D/27+0j)
    u=((-q-r)/2)**0.33333333333333333333333
    v=((-q+r)/2)**0.33333333333333333333333

    
    z_candidate = [u+v-z0, u*complex_num + v *complex_num_2-z0, u*complex_num_2 + v*complex_num-z0]
    return z_candidate


def two_stage(distances_to_anchors, anchor_positions, u):
    tag_pos, det, b = lsq_method(distances_to_anchors, anchor_positions, u)
    
    z = symbols('z') #, real = True
    f = symbols('f', cls = Function)
    f = 0
    sum_delta, b_z, c_z, d_z = 0, 0, 0, 0
    for i in range(anchor_positions.shape[0]):
        delta = distances_to_anchors[i]**2 - ((tag_pos[0]- anchor_positions[i][0])**2 + (tag_pos[1]- anchor_positions[i][1])**2)
        f += 4 * ((z - anchor_positions[i][2]) ** 3 - delta*((z)-anchor_positions[i][2]))
    coeff = extract_coefficients(f)
    
    z_candidate = solve(f,z)
    # z_candidate = Cardano(coeff[0], coeff[1], coeff[2], coeff[3])
    # z_candidate = cardano_formula(coeff[0], coeff[1], coeff[2], coeff[3])
    # print('This is z candidate cardano ', z_candidate)

    z_candidate = np.array([complex(item) for item in z_candidate])
    # print('This is z candidate cardano', z_candidate)
    z_candidate = np.round(np.array([abs(z_candidate[0]), abs(z_candidate[1]), abs(z_candidate[2])]),5)

    result = list()
    check_ls = list()
    
        
    for i in range(z_candidate.shape[0]):
        check = abs(distances_to_anchors[0]**2 - (tag_pos[0] - anchor_positions[0][0])**2 - (tag_pos[1] - anchor_positions[0][1]) **2 - (z_candidate[i] - anchor_positions[0][2])**2)
        check_ls.append(check)
    index = check_ls.index(min(check_ls))
    # print('index', index)
    
    two_ans = np.array([tag_pos[0], tag_pos[1], z_candidate[index]])
    # print('two_ans', two_ans)
    result.append(two_ans)
    
    return np.array(result).astype(np.float32), det, b

def is_real_root(root, tolerance=1e-100):

    return abs(np.imag(root)) < tolerance


def two_stage_solve(distances_to_anchors, anchor_positions, u):
    tag_pos, det, b = lsq_method(distances_to_anchors, anchor_positions, u)
    
    z = symbols('z') #, real = True
    f = symbols('f', cls = Function)
    f = 0
    sum_delta, b_z, c_z, d_z = 0, 0, 0, 0

    # print('truth', truth)
    for i in range(anchor_positions.shape[0]):
        delta = distances_to_anchors[i]**2 - ((tag_pos[0]- anchor_positions[i][0])**2 + (tag_pos[1]- anchor_positions[i][1])**2)
        # print('delta', delta)
        f += 4 * ((z - anchor_positions[i][2]) ** 3 - delta*((z)-anchor_positions[i][2]))
    coeff = extract_coefficients(f)
    
    print('coeff', coeff)
    # z_candidate = solve(f,z)
    # z_candidate = solveset(f,z)
    # z_candidate = Cardano(coeff[0], coeff[1], coeff[2], coeff[3])
    z_candidate = np.roots(coeff)
    # print('z_candidate', z_candidate)
    a = coeff[0]
    b = coeff[1]
    c = coeff[2]
    d = coeff[3]
    # 计算判别式
    D0 = b**2 - 3*a*c
    D1 = 2*b**3 - 9*a*b*c + 27*a**2*d
    discriminant = D1**2 - 4*D0**3
    D = 18*a*b*c*d - 4*b**3*d + b**2*c**2 - 4*a*c**3 - 27*a**2*d**2
    discriminant = D
    # print('discriminant', discriminant)
    z_candidate = np.array([complex(item) for item in z_candidate])

    real_roots = collections.deque(maxlen = 3)
    tolerance = 1e-10
    
    if discriminant < 0:
        # 一實根
        print('one real root')
        truth = 1

        if is_real_root(z_candidate[0], tolerance):
            real_roots.append(z_candidate[0])
        
        if is_real_root(z_candidate[1], tolerance):
            real_roots.append(z_candidate[1])
        
        if is_real_root(z_candidate[2], tolerance):
            real_roots.append(z_candidate[2])
        
        if len(real_roots) == 1:
            real_roots = [real_roots[0], real_roots[0], real_roots[0]]
            z_candidate = real_roots
    else:
        # 三個不同實根
        print('three real root')
        truth = 0
        z_candidate = z_candidate
        
    # print('This is z candidate solve', z_candidate)
    # z_candidate = np.round(np.array([abs(z_candidate[0]), abs(z_candidate[1]), abs(z_candidate[2])]),5)
    # print('This is z candidate', z_candidate_max)  
    z_candidate = np.array(z_candidate)
    result = list()
    check_ls = list()
    
    for i in range(z_candidate.shape[0]):
        check1 = abs(distances_to_anchors[0]**2 - (tag_pos[0] - anchor_positions[0][0])**2 - (tag_pos[1] - anchor_positions[0][1]) **2 - (z_candidate[i] - anchor_positions[0][2])**2)
        check2 = abs(distances_to_anchors[1]**2 - (tag_pos[0] - anchor_positions[1][0])**2 - (tag_pos[1] - anchor_positions[1][1]) **2 - (z_candidate[i] - anchor_positions[1][2])**2)
        check3 = abs(distances_to_anchors[2]**2 - (tag_pos[0] - anchor_positions[2][0])**2 - (tag_pos[1] - anchor_positions[2][1]) **2 - (z_candidate[i] - anchor_positions[2][2])**2)
        check4 = abs(distances_to_anchors[3]**2 - (tag_pos[0] - anchor_positions[3][0])**2 - (tag_pos[1] - anchor_positions[3][1]) **2 - (z_candidate[i] - anchor_positions[3][2])**2)
        
        # check1 = abs((z_candidate[i] - anchor_positions[0][2]))
        # check2 = abs((z_candidate[i] - anchor_positions[1][2]))
        # check3 = abs((z_candidate[i] - anchor_positions[2][2]))
        # check4 = abs((z_candidate[i] - anchor_positions[3][2]))
        
        
        check = check1 + check2 + check3 + check4
        # print('check1', check1)
        # print('check2', check2)
        # print('check3', check3)
        # print('check4', check4)
        
        check_ls.append(check)
    # print('check_ls', check_ls)
    index = check_ls.index(min(check_ls))
    two_ans = np.array([tag_pos[0], tag_pos[1], abs(z_candidate[index])])
    # two_ans = np.array([tag_pos[0], tag_pos[1], z_candidate[index]])
    result.append(two_ans)
    
    return np.array(result).astype(np.float32), det, z_candidate, truth 

