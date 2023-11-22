#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:21:09 2023

@author: stuart

"""

import numpy as np # useful tools for data manipulation
import csv
from sys import exit
from nltk.stem.snowball import EnglishStemmer # used to stem a word into a %lexeme

# a function to remove tokens coded with a pronoun
def removepronouns(verbpart, noverb, nopart):
   noproverbpart = []
   for i in range(noverb):
       noproverbpart.append([])
       for j in range(nopart):
           noproverbpart[i].append([])
    
   for i in range(noverb):
       for j in range(nopart):
           for k in range(len(verbpart[i][j])):
               try:
                   if verbpart[i][j][k][1] == False:
                       noproverbpart[i][j].append(verbpart[i][j][k])
               except:
                   print(i,j,k)
   return noproverbpart

# a function to count transitive tokens for each particle verb type
def extractwordfreq(verbpart, noverb, nopart):
    worddis = []
    wordfreq = []
    for i in range(noverb):
       worddis.append([])
       wordfreq.append([])
       for j in range(nopart):
            worddis[i].append([])
            wordfreq[i].append([])
            
    for i in range(noverb):
        for j in range(nopart):
            for k in range(len(verbpart[i][j])):
                worddis[i][j].append(verbpart[i][j][k][0])
                
    for i in range(noverb):
        for j in range(nopart):
                if len(worddis[i][j]) != 0:
                    for k in range(max(worddis[i][j]) + 1):
                        wordfreq[i][j].append([])
                    for m in range(max(worddis[i][j]) + 1):
                        wordfreq[i][j][m] = 0
                    for l in range(len(worddis[i][j])):
                        for n in range(max(worddis[i][j]) + 1):
                            if worddis[i][j][l] == n:
                                wordfreq[i][j][n] = wordfreq[i][j][n] + 1
                else:
                    wordfreq[i][j] = [0]
    return wordfreq
# =============================================================================
#                           Things to input
# =============================================================================



# file paths for data storage
rootpath = "/home/stuart/Documents/Uni/Graduate Diploma Advanced/Thesis/Data/"
setfolder = "thesis_attempt/" 
spacefolder =  "verbspace/"
arrayfile = "verb_particle_array.csv"
verbkeyfile = "verb_key.csv"
partkeyfile = "particle_key.csv"
verbpartfile = "particle.csv"
totalverbfile = "total_verb_count.csv"
particleverbfolder = "particle_verbs/"

# removes pronoun cases from word distance data
removepro = True

# token cut off points
MItokenscutoff = 70
wordiscutoff = 9
graphtokencutoff = 30

# write the data to file
writecsv = True
pmiworddisfile = 'pmiworddis.csv'


# some variables to control what data is being looked at

# ploted verbs 
verbrange = []

# ploted particles 
particlerange = [] 

# mutual information range
MImin, MImax = False,False # left of at 1,2
# mean word distance range
Wordmin, Wordmax = False,False

# things to exclude from the calculations

# removed verbs from graphinh 
remverb =  ['lay']

# erroneous particle verbs to remove entirely from calculations and intranstive particle verbs to remove from dist calcs

erronverb = ['find up' , 'know out', 'do out','find up', 'want back','do up','run back','find back','see up'
,'make over' ,'smile back' ,'be in','be down','be over','be back','be along','peer down','be on','gaze out'
,'want back','pour down','see out','see down','be up','be around','get over','be off','be out','have up'
,'have down','have out','have in','be around','have back','have off','be about','have around','get back' 
,'run around','trickl down','set about','see back','get round','see around']

intranspartverb = [ 'dress up','hang on','hang out','hang down','open out','opt out','cri out' ,'dri up' 
,'dri out' ,'fall out','branch out','rush out' ,'lash out','hang about','stick up' ,'slid down','stream down'
,'rub off','play out','cash in' ,'fed up','glanc up','fell down','sit out' ,'run up','glanc back','hang around'
,'rise up','opt out','hang about','look down','look back','fed up','arrive back','jut out','fight back','home in'
,'driven up','cash in','peter out','touch down','line up','speak up','finish up' ,'stay out','stretch up','head off'
,'peer out','left over','show up','smile up','burst out','slip down','smile down','mess around','mess about','wander off'
,'ring out','drag on','crop up','sit around','move on','wander around','sign on','born out','fall off','grow up','splash out'
,'close in','live on','walk around','breath in','sit down','die out','go out','stand out' ,'go in','go back','look out'
,'thrown up','live in','came down','sign up','stop off','break out','spill over','come along','fit in','stay on','wake up'
,'light up','end up','move in','fall over','come back','come around','carry on','step in' ,'tune in','fall back','step back'
,'come on','go on','run off','sit in','fall in','go along','jump up','strike out','come about','step down','climb down'
,'gaze up','die down','set out','come down','give up','split up','stare back','move about','cheer up','drop out','join in'
,'set about','stand up','struck out','turn in', 'weigh in', 'walk along', 'go around', 'get down', 'got out', 'come over'
, 'get on', 'get up', 'get off', 'walk out', 'go off', 'come up', 'come out', 'go up', 'walk back', 'run out', 'come off'
, 'stay up']


# =============================================================================
#                           Under the hood
# =============================================================================

# updating rootpath
pmiworddispath = rootpath + pmiworddisfile
rootpath = rootpath + setfolder

# intialising stemmer
stemmer = EnglishStemmer()

if bool(verbrange) != False:
    for i in range(len(verbrange)):
        verbrange[i] = stemmer.stem(verbrange[i])
if bool(particlerange) != False:
    for i in range(len(particlerange)):
        particlerange[i] = stemmer.stem(particlerange[i])
        
for i in range(len(remverb)):
    remverb[i] = stemmer.stem(remverb[i])

# formatting list of erronious and intranstive particle verbs
erronverbcont = [[erronverb[i].split(' ')[0], erronverb[i].split(' ')[1]] for i in range(len(erronverb))]
erronverbcont.sort(key=lambda x: x[0])   

intranscont = []
for i in range(len(intranspartverb)):
    intranscont.append(intranspartverb[i].split(' '))
    intranscont[i][0] = stemmer.stem(intranscont[i][0])
    intranscont[i][1] = stemmer.stem(intranscont[i][1])
  
        
        
        
csv.register_dialect('datadialect', delimiter='|', quoting=csv.QUOTE_NONNUMERIC, quotechar='\'')


#count how many tokens removed from word dis cut off 
wordcutcount = 0
    
# =============================================================================
#     reading in the data from extraction algorithm
# =============================================================================
spacepath = rootpath + spacefolder

# reading in the verb key
verbkeypath = spacepath + verbkeyfile
with open(verbkeypath, 'r',) as file:
    reader = csv.DictReader(file, dialect ='datadialect')
    for row in reader:            
            verbkey = row
    
noverb = len(verbkey) - 1
verbs = []
for j in range(noverb):
    verbs.append(list(verbkey.keys())[j])


# reading in the particle key            
partkeypath = spacepath + partkeyfile
with open(partkeypath, 'r',) as file:
    reader = csv.DictReader(file, dialect ='datadialect')
    for row in reader:
            partkey = row
            
nopart = len(partkey) - 1
particles = []
for j in range(nopart):
    particles.append(list(partkey.keys())[j])


# reading in the array of verb particle counts
verbpartarray = np.zeros([noverb + 1, nopart + 1], dtype = np.intp) 
arraypath = spacepath + arrayfile
with open(arraypath, 'r') as file:
    reader = csv.reader(file, dialect = 'datadialect')
    rowindex = 0
    for row in reader:
        for j in range(nopart + 1):
            verbpartarray[rowindex][j] = row[j]
        rowindex = rowindex + 1

# reading the total amount of verbs found
totalverbpath = spacepath + totalverbfile  
with open(totalverbpath, 'r') as file:
    reader = csv.reader(file, dialect = 'datadialect')
    for row in reader:
        totalverbcount = row[1]
        
# reading in the data for word distances, pronouns and sentences        
verbpart = []
noproverbpart = []
for i in range(noverb):
    noproverbpart.append([])
    verbpart.append([])
    for j in range(nopart):
        noproverbpart[i].append([])
        verbpart[i].append([])

for i in range(noverb):
    verbpath = spacepath + particleverbfolder + verbs[i] + '_' + verbpartfile
    partindex = 0
    with open(verbpath, 'r') as file:
        reader = csv.reader(file, delimiter = '|', quotechar='\'')
        rowcount = 0
        for row in reader:
            rowlist = []
            rowcount = rowcount + 1
            try:
                while row[0] != particles[partindex]:
                    partindex = partindex + 1
                rowlist.append(int(row[1]))
                if row[2] == 'True':
                    rowlist.append(True)
                elif row[2] == 'False':
                    rowlist.append(False)
                rowlist.append(row[3])
                verbpart[i][partindex].append(rowlist)
            except:
                print('There\'s an error reading in the word distance data.')
                print('The error is in folder: ' + spacefolder)
                print('with word: ' + list(verbkey.keys())[i])
                print(' and row: ' + str(rowcount))
 

# combinng round and around particles
for i in range(noverb):
    verbpartarray[i][list(partkey.keys()).index('around')] = verbpartarray[i][list(partkey.keys()).index('around')] + verbpartarray[i][list(partkey.keys()).index('round')]
    verbpart[i][list(partkey.keys()).index('around')].extend(verbpart[i][list(partkey.keys()).index('round')])
    del verbpart[i][list(partkey.keys()).index('round')]    
verbpartarray = np.delete(verbpartarray, list(partkey.keys()).index('round'), 1)
del partkey['round']
nopart = nopart - 1
particles.remove('round')

# counting variables
errorcount = 0
transcount = 0 

# removing counts beyond cut off for word distance, total particle verb count and removing erronous particle verbs
for i in range(noverb):
    for j in range(nopart):
            nodel = 0
            for n in range(len(verbpart[i][j])):
                n = n - nodel
                if verbpart[i][j][n][0] >= wordiscutoff:
                    wordcutcount = wordcutcount + 1 
                    verbpartarray[i][j] = verbpartarray[i][j] - 1
                    del verbpart[i][j][n]
                    nodel = nodel + 1
                    
            for m in range(len(erronverbcont)):
                if  list(verbkey.keys())[i] == erronverbcont[m][0] and list(partkey.keys())[j] == erronverbcont[m][1]:
                    errorcount = errorcount + verbpartarray[i][j]
                    verbpartarray[i][j] = 0 
                    nodel = 0
                    for n in range(len(verbpart[i][j])):
                        n = n - nodel
                        del verbpart[i][j][n]
                        nodel = nodel + 1


# code to remove pronoun cases from word distance data
if removepro == True:
    verbpart = removepronouns(verbpart, noverb, nopart)


# make array for probabilty space
verbspace = np.zeros([noverb + 1, nopart + 1])
for i in range(noverb):
    for j in range(nopart):
        verbspace[i+1][j+1] = int(verbpartarray[i][j])


# counting all the verbs with and without a particle
for i in range(noverb):
    verbcount = 0
    for j in range(nopart + 1):
        verbcount = verbcount + int(verbpartarray[i][j])
    verbspace[i+1][0] = verbcount

# counting all particle verbs (trans and intrans)
totpartverb = 0 
for i in range(noverb):
    for j in range(nopart):
        totpartverb = totpartverb + int(verbpartarray[i][j])
    

# counting all the particles
totpart = 0
allpart = np.zeros([nopart + 1])
for i in range(nopart):
    partcount = 0  
    for j in range(noverb):
        partcount = partcount + int(verbpartarray[j][i])
        allpart[i + 1] = allpart[i + 1]  + verbpartarray[j][i] 
    verbspace[0][i+1] = partcount
    totpart = totpart + partcount
        
# counting token frequencies for each particle verb type
wordfreq =  extractwordfreq(verbpart, noverb, nopart)

# =============================================================================
# # calculutaing probability, suprisal and pmi
# =============================================================================

totalcount = totalverbcount
graphspace = verbspace


# to count how many transitve particle verbs are in the data
totaltranscount = 0 
    
# logic varible to check for particle verb rejection
removeverb = False 
# going over each particle verb combination possible      
MIallspace = []     
for i in range(noverb):
    for j in range(nopart):
       
        verb = list(verbkey.keys())[i]
        particle = list(partkey.keys())[j]
        
        if graphspace[i+1][j+1] == 0:
            continue
  
        # removing particle verbs from list of intranstive verbs
        for n in range(len(intranscont)):
            if verb == intranscont[n][0] and particle == intranscont[n][1]:
                transcount = transcount + graphspace[i+1][j+1]
                removeverb = True
                break
        if removeverb == True:
            removeverb = False
            continue
 
        # calculating the probability and suprisal from the verb counts
        countcont = np.zeros([3,3])
        if graphspace[i+1][0] <= MItokenscutoff:
            continue
        elif graphspace[i+1][0] > MItokenscutoff: 
            countcont[0][0] = graphspace[i+1][0]
            countcont[1][0] = countcont[0][0]/totalcount
            countcont[2][0] = -1*np.log2(countcont[1][0])

        # calculating the probability and suprisal from the particle counts
        if graphspace[0][j+1] <= MItokenscutoff:
            continue
        elif graphspace[0][j+1] > MItokenscutoff:
            countcont[0][1] = graphspace[0][j+1]
            countcont[1][1] = countcont[0][1]/totalcount
            countcont[2][1] = -1*np.log2(countcont[1][1])
        
        # calculating the probability and suprisal from the particle verb counts
        if graphspace[i+1][j+1] <= MItokenscutoff:
            continue  
        elif graphspace[i+1][j+1] > MItokenscutoff:
            countcont[0][2] = graphspace[i+1][j+1]
            countcont[1][2] = countcont[0][2]/totalcount
            countcont[2][2] = -1*np.log2(countcont[1][2])
         
        # checking token count cut offs and mean word distance for each pv type
        notokens = 0 
        wordfreqcont = []
        Exp = 0
        for m in range(len(wordfreq[i][j])):
            notokens = notokens + wordfreq[i][j][m]
            wordfreqcont.append([])
            if not m < wordiscutoff:
                continue
        totaltranscount = totaltranscount + notokens
        if notokens == 0:
            continue
        if notokens <= graphtokencutoff:
            continue
        for m in range(len(wordfreq[i][j])):
                wordfreqcont[m] = wordfreq[i][j][m]/notokens 
        for x in range(len(wordfreq[i][j])):
            Exp = Exp + x*wordfreqcont[x]
            
        # calculating the pmi two ways and comparing for a sanity check
        pmi = countcont[2][0] + countcont[2][1] - countcont[2][2]
        pmicheck = np.log2((totalcount*countcont[0][2])/(countcont[0][0]*countcont[0][1]))

        if np.round(pmi,2) != np.round(pmicheck,2):
            print('There\'s an error in the pmi calculation')
            exit()

        # storing all the data
        pmicont = [verb, particle, countcont, pmi, Exp, wordfreq[i][j][0], notokens]
        MIallspace.append(pmicont)

    
# some code the place restrictions on what data will be written
MImintruth = bool(MImin)
if type(MImin) is int:
    MImintruth = True
MImaxtruth = bool(MImax)
if type(MImax) is int:
    MImaxtruth = True
Wordmintruth = bool(Wordmin)
if type(Wordmin) is int:
    Wordmintruth = True
Wordmaxtruth = bool(Wordmax)
if type(Wordmax) is int:
    Wordmaxtruth = True

restrictedcont = []   
for j in range(len(MIallspace)):
    if MIallspace[j][0] in remverb:
        continue
    if MIallspace[j][6] <= graphtokencutoff:
        continue
    # applying criteria to data
    if bool(verbrange) == False or MIallspace[j][0] in verbrange:
        if bool(particlerange) == False or MIallspace[j][1] in particlerange:
            if MImintruth == False or MIallspace[j][3] >= MImin:
                if MImaxtruth == False or MIallspace[j][3] <= MImax:
                    if Wordmintruth == False or MIallspace[j][4] >= Wordmin:
                        if Wordmaxtruth == False or MIallspace[j][4] <= Wordmax:
                            restrictedcont.append([MIallspace[j][0], MIallspace[j][1], 
                                                 MIallspace[j][3]])

  
# puting the data into containers for writing                                    
Verb_val = []
Part_val = []
Vs_val = np.zeros(len(restrictedcont))

#  filling variable containers
for j in range(len(restrictedcont)):
    Vs_val[j] = restrictedcont[j][2]
    Verb_val.append(restrictedcont[j][0])
    Part_val.append(restrictedcont[j][1])


# writing the data to file
if writecsv == True:
    # formatting data to input into R
    Rdata = []
    for j in range(noverb):
        for l in range(len(Verb_val)):
            if Verb_val[l] == verbs[j]:
                for k in range(nopart):
                    if Part_val[l] == particles[k]:
                        for n in range(len(verbpart[j][k])):
                            if not verbpart[j][k][n][0] < wordiscutoff:
                                continue
                            row = [Verb_val[l], Part_val[l], Vs_val[l], verbpart[j][k][n][0], verbpart[j][k][n][1]]
                            Rdata.append(row)

    # this code (immediately below) was adapted from the tutorial at: https://www.programiz.com/python-programming/csv
    with open(pmiworddispath, 'w', newline='') as file:
        writer = csv.writer(file)        
        header = ['verb', 'particle', 'pmi', 'wd', 'pro']        
        writer.writerow(header)
        for j in range(len(Rdata)):    
            writer.writerow(Rdata[j]) 

                            
                
















