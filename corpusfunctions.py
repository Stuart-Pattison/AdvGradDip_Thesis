#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:57:07 2023

@author: stuart
functions used in the particleverb_extraction_algorithm.py script to format the sentence string containers and the token array
"""
import numpy as np

def appendsentlist(sentences, listcoord, *npart):
    sentlist = []     
    sentences = sentences
    if npart:
        npart = npart[0]
        for j in range(npart):
            sentlist.append([])
    
        for k in range(npart):      
            for i in range(len(listcoord[k])):
                sentlist[k].append(sentences[listcoord[k][i]]) 
                 
    else:
        for i in range(len(listcoord)):
            sentlist.append(sentences[listcoord[i]]) 
           
    return sentlist

def multiverbappendsentlist(sentences, listcoord, nverb, *npart):
    if npart:
        npart = npart[0]
        sentlist =[]
        for i in range(nverb):
                sentlist.append([])
                for j in range(npart):
                        sentlist[i].append([])          
              
        for i in range(nverb):
            for j in range(npart):
                for k in range(len(listcoord[i][j])):
                    sentlist[i][j].append(sentences[listcoord[i][j][k]])
    else:
        sentlist =[]
        for i in range(nverb):
                sentlist.append([])        
              
        for i in range(nverb):
                for k in range(len(listcoord[i])):
                    sentlist[i].append(sentences[listcoord[i][k]])            
    return sentlist
  
def dataarray(verbpartpairs, nullpart, nullverb, noverb, nopart):
    noverb = noverb
    nopart = nopart
    
    verbpartarray = np.zeros([noverb+1,nopart+1], dtype = np.intp)
    for i in range(noverb):
        for j in range(nopart):
            verbpartarray[i][j] = verbpartpairs[i][j]
            verbpartarray[i][nopart] = nullpart[i]
            verbpartarray[noverb][j] = nullverb[j]
    return verbpartarray
    
     