#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:47:17 2023

@author: stuart

Some code to read and write CSV files code was adapted from: https://www.programiz.com/python-programming/csv
BNC was download at: https://www.ota.ox.ac.uk/desc/2554
"""
from nltk.corpus.reader.bnc import BNCCorpusReader  # used to import BNC corpus
from nltk.stem.snowball import SnowballStemmer # used to stem a word into a %lexeme
import shutil # used to make line breaks in terminal
import numpy as np # useful tools for data manipulation
import time # used to measure how long the program is taking
import csv # to write data to files
from corpusfunctions import appendsentlist, multiverbappendsentlist, dataarray

start_time = time.time() # measuring how long it takes to run
terminalwidth = shutil.get_terminal_size()[0] # used to print out a breaking line between running the script and the results    

# =============================================================================
#                            Things to input
# =============================================================================

corpusroot = "/home/stuart/Documents/Uni/Graduate Diploma Advanced/Thesis/BNC/Texts" # the location of the corpus being used
# fileids = r'B/B2/B24.xml' # corpus file ids 
fileids = r'[A-K]/\w*/\w*\.xml' # to access all files in bnc corpus
# fileids = r'B/B2/B24.xml' # is a good one to practice code on

# the verb part of the particle verb being looked at
verbs = ['pick', 'put'] 
                
# will import the verb list from a .csv file rather than those listed above
importverbs = True
verb_listfile = "/Inputs/found_verbs_clean.csv"

# the set of particles
particles = ['about', 'along', 'around', 'back', 'by', 'down',
             'in', 'off', 'on', 'out', 'over', 'round', 'up']

# write the data to file
writedata = True

# file paths for data storage
rootpath = "/home/stuart/Documents/Uni/Graduate Diploma Advanced/Thesis/Data/thesis_attempt/"
folderpath = "verbspace/"
arrayfile = "verb_particle_array.csv"
verbkeyfile = "verb_key.csv"
partkeyfile = "particle_key.csv"
verbpartfile = "particle.csv" # each verb will be stored with the filename form 'verb_'+ verbpartfile
particleverbfolder = "particle_verbs/"
totalverbfile = "total_verb_count.csv"

# store ALL sentences with the verbs (warning it's alot)
storesentences = False

# write the compound particle sentences (includes other criteria with similar distribution)
writecomppart = True
comppartfile = "compound_prepositions.csv"
# write all the sentences tagged as passives.
writepassive = True
passivefile = "passive_sents.csv"
# write out yall sentences
yallprowrite = True
yallfile = "PRO_all.csv"

# write all the verbs that can take a particle found
writefoundverb = False 
foundverbs = "Inputs/found_verbs.csv"


# Lists of things used in checks

# compound particles to remove
compoundparticles = ['up to', 'out of', 'on to', 'back up', 'up against'
                     'back to', 'off to', 'round about', 'round to', 'up until', 'out to', 
                     'along with', 'in from', 'up with', 'up in', 'up on', 
                     'down on', 'down in', 'down through', 'down under', 'down to',
                     'back into', 'back to', 'back in', 'up into', 'on up', 
                     'out towards', 'out came', 'on about', 'back again', 'back through', 
                     'back towards', 'back out', 'up above', 'over to', 'in between', 'back up', 
                     'over by']

# Part + place adverb
partplace = ['up here', 'up front', 'up there', 'down here', 'down there', 'out there', 
             'out here', 'up ahead', 'back here', 'back there', 'back home']

# part + ad
partadv = ['out loud', 'up soon', 'up again', 'off again', 'over again']

# sentences with the following words after a joint particle verb won't be counted towards word distance
postpart = ['there', 'where', 'then', 'than', 'which', 'when', 'how', 'here']

# sentences with the following words between a verb and particle will consdiered a verb without a particle
midpart = ['where', 'then', 'which', 'when', 'how', 'so', 'if', 'whether', 'as', 'for', 'into']

waypart = ['way']

# y'all varaients
allcont = ['all', '\'ll', '\'\'ll', '\"ll', 'll']

# preceding particle
prepart = ['the', 'and']

# immediatly after verb
postverb = ['from', 'then', 'there']

# Prt + and + Prt
andpart = ['up and down', 'back and forth ', 'away and down', 'inside and out', 'over and over']


# erronous constructions to remove
timepart = ['earlier on', 'later on', 'now on', 'early on', 'year round', 'years back']

dispart = ['far back', 'turn back', 'further on', 'far out', 'round about', 'furlong out', 'further out', 'high up',  
           'further along', 'further in', 'miles down', 'miles up', 'feet up']

nompart = ['lit up', 'rained down', 'sit down', 'turn up', 'side up', 'head out', 'rushed out', 'set up', 
           'sucked up', 'leap out', 'spit out', 'fill out', 'runs out', 'built up', 'open up', 'thought out', 'set out',
           'looked up', 'pack up', 'washing up', 'break up', 'broke up', 'climb out', 'find out', 'face up', 'close up', 
           'spread out', 'stretch out', 'slip up', 'lash out','ring out', 'cut out', 'caught out', 'read out', 'cry out',
           'stepped back', 'take away', 'sucked in', 'stepped in', 'sorted out', 'sorted out', 'sort out', 'hanging out', 
           'hang out', 'worked up', 'work up', 'working up', 'stand up', 'knock back', 'laid down', 'look around', 'look round', 'stuck in' ]

idiompart = ['well off', 'better off', 'all round', 'straight on', 'size up', 'not out', 'upside down', 'inside out',
             'already up', 'world over', 'well up', 'win back', 'set back', 'back down', 'all over', 'all along']





# =============================================================================
#                           Under the hood
# =============================================================================

print(''.ljust(terminalwidth,'%'))
print('Begining Initialisation')

# importing the verb list
if importverbs == True:
    verbread = []
    verbpath = rootpath + verb_listfile
    with open(verbpath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            verbread.append(row)
    verbs = []
    for i in range(len(verbread)):
        verbs.append(verbread[i][0])


# initialisng stemming algorithm 
stemmer = SnowballStemmer("english")

# stemming verb and particle list
verbstem = []
noverb = len(verbs)
for x in range(noverb):
    verbstem.append(stemmer.stem(verbs[x]))
particlestem = []
nopart =len(particles)
for x in range(nopart):     
    particlestem.append(stemmer.stem(particles[x]))
    
# importing the BNC corpus
tagged_sentences = BNCCorpusReader(root=corpusroot, fileids=fileids, lazy=True).tagged_sents(c5=True)

# =============================================================================
# initialising and formatting containers for constructions used in checking criteria
# =============================================================================

# initalising compound preposition constructions
comppartcont = []
comppartstore = []
for i in range(len(compoundparticles)): 
    comppartcont.append(compoundparticles[i].split(' '))

# initalising part palce constructions
partplacecont = []
for i in range(len(partplace)): 
    partplacecont.append(partplace[i].split(' '))

# initialising part-adverb constructions 
partadvcont = []
for i in range(len(partadv)): 
    partadvcont.append(partadv[i].split(' '))

# initalising 'part and part' constructions
andpartcont = [[stemmer.stem(andpartcon.split(' ')[0]), stemmer.stem(andpartcon.split(' ')[2])] for andpartcon in andpart]
andpartcont.sort(key=lambda x: x[0])

# initialising erronious constructions

# erronious time adjuncts 
timepartcont = []
for i in range(len(timepart)):
    temp = timepart[i]
    timepartcont.append([stemmer.stem(temp.split()[1]), stemmer.stem(temp.split()[0])])
timepartcont.sort(key=lambda x: x[0])
ordertimepartcont = timepartcont

# erronious dis adjuncts 
dispartcont = []
for i in range(len(dispart)):
    temp = dispart[i]
    dispartcont.append([stemmer.stem(temp.split()[1]), stemmer.stem(temp.split()[0])])
dispartcont.sort(key=lambda x: x[0])
orderdispartcont = dispartcont

#erronious idioms
idiompartcont = []
for i in range(len(idiompart)):
    temp = idiompart[i]
    idiompartcont.append([stemmer.stem(temp.split()[1]), stemmer.stem(temp.split()[0])])
idiompartcont.sort(key=lambda x: x[0])
orderidiompartcont = idiompartcont

# erronious nominalisations
nompartcont = []
for i in range(len(nompart)):
    temp = nompart[i]
    nompartcont.append([stemmer.stem(temp.split()[1]), stemmer.stem(temp.split()[0])])
nompartcont.sort(key=lambda x: x[0])
ordernompartcont = nompartcont
            
# initialising words that are check preceding a particle
prepartstem = [stemmer.stem(word) for word in prepart]             

# =============================================================================
# Initialising containers for algoritm to fill
# =============================================================================

procverb = np.zeros(noverb,dtype=np.intp) # counts the tokens of each verb 
procpart = np.zeros(nopart,dtype=np.intp) # counts how many particles there are
nullpart = np.zeros(noverb,dtype=np.intp) # counts how many null/unassigned particles there are
nullverb = np.zeros(nopart,dtype=np.intp) # counts how many null/unassigned verbs there are
verbpartpairs = np.zeros([noverb,nopart]) # counts how many verb particles there are  
verbsentcoord = [] # creating a list to store the sentence coordinates for verb we are looking for
passivesentcoord = [] # creating a list to store the sentence coordinates for verb we are looking for
verbnopartcoord =[] # creating a list to store the sentence coordinates that don't have a particle
partsentcoord = [] # creating a list to store the sentence coordinates for the particles we are looking for
verbpartcoord = [] # creating a list to store the sentence coordinates for the particle verb we are looking for
verbpartdis = [] # creating a list to store the distance between the verb and particle
objpro = [] # creating a list to store whether the intermediatery word is a pronoun
yallprostore = [] # list to store Pro + all sentence coordinates

# formatting some containers to be divided by particle verb
for m in range(noverb):
    verbsentcoord.append([])
    verbnopartcoord.append([])
    verbpartpairs[m,:] = 0
    verbpartdis.append([])
    objpro.append([])
    verbpartcoord.append([])
for n in range(nopart): # making a seperate list for each particle in question
    partsentcoord.append([])
    verbpartpairs[:,n] = 0
for m in range(noverb):
    for n in range(nopart):
        verbpartdis[m].append([])  
        objpro[m].append([])
        verbpartcoord[m].append([])

# containters for finding verbs
missverbstemcont = []
missverbcont  = []

# =============================================================================
# Initialising counting variables
# =============================================================================

nosent = len(tagged_sentences) # total number of sentences being looked at
procsent = 0 # counts how many sentences have been processed
procword = 0 # counts how many words have been processed
totprocverb = 0 # counts total number of verb tokens there are
    
# some variables to track how far along the process is
per25 = 0
per50 = 0
per75 = 0

#variables to count how many instances of particluar featrues there are
# variables to count how many nonetypes there are
nonecount = 0 
wordonenonecount = 0
# count no of part verbs found
partverbcount = 0 
# count no of passives
passivecount = 0
# count no of pronouns
procount = 0
# count no of pro + all
yallcount = 0
# count number of compound prepositons
compartcount = 0 
# count the number of part place phrases e.g. up here
partplacecount = 0
# count number over part adverb phrases e.g. off again
partadvcount = 0
# count no of part and part constructions 
andpartcount = 0 
# count no of the part and  part constructions 
prepartcount = 0
# count the no of time adjuncts 
timepartcount = 0 
# count no of distance adjuncts
dispartcount = 0
# count no of idioms
idiompartcount = 0
# count no of nominalisations 
nompartcount = 0
# no of part verbs removed with PP or CP following joint part verb
postpartcount = 0
# no or part verbs removed due to no noun intervening 
nouncount = 0
# no of part verbs removed with PP or CP immediatly after verb
postverbcount = 0
# no of part verbs removed with subord CP between V and Prt
midcount = 0
# no of ways removed
waycount = 0
# counting how many things flasly flaged as passives
falsepassive = 0

# =============================================================================
# The algorithm proper
# =============================================================================

# timing algorithm
initial_time = time.time()
total_initial_time = (np.round(initial_time - start_time,2))

print('Initialising finished!')
print('')
print('Initialisation took '+str(total_initial_time)+' seconds to complete.')

print(''.ljust(terminalwidth,'%'))
print('Begining algorithm now!')

# intiliasing the running count variables
verbcoord = -1
partcoord = -1
wordcountstore = -1
for i in range(nosent): # counts over each sentence
   
    procsent = procsent +1 # counts how many sentences are processed
    
    # some code to figure out how far along in the proccess we are
    procent = np.round((procsent/nosent)*100,0)
    if per25 == 0 and procent == 25:
        per25 = 1   
        print ('We are about 25% along with '+str(procsent)+' out of '+str(nosent)+' sentences completed.')
    if per50 == 0 and procent == 50 :
        per50 = 1   
        print ('We are about 50% along with '+str(procsent)+' out of '+str(nosent)+' sentences completed.')
    if per75 == 0 and procent == 75:
        per75 = 1   
        print ('We are about 75% along with '+str(procsent)+' out of '+str(nosent)+' sentences completed.')
        
    # resetting running count and string variables
    wordcount = -1
    partsentplace = -1
    verbsentplace = -1
    missverbstem = ''
    missverb = ''
    
    # counts how many words in the sentence have been processed
    procwordsent = 0 

    # extracting the ith sentence
    loopsent = tagged_sentences[i] 
    # number of words in the sentence
    noloopsent = len(loopsent) 

    # logic variables to determine if token is accepted or rejected by the algorithm
    verbcount = False
    passive = False
    noun = False
    part = False
    verb = False 
    errorword = False
    midwordpart = False
    waywordpart = False
    pro = False
    lastword = False
    
    # counts over each word
    for j in range(noloopsent): 
        # checking if the word is the last in the sentence
        if j == noloopsent:
            lastword = True
        else:
            lastword = False
        
        # counts total words procesed
        procword = procword + 1 
        # counting distance between words
        wordcount = wordcount + 1
        # extracting the jth word
        taggedword = loopsent[j][1]
        
        # to avoid crashes caused by no tagging
        if taggedword == None: 
            nonecount = nonecount + 1
            continue
        # checking if the word is a verb or punctuation
        elif (taggedword.startswith('V') and '-' not in taggedword) or taggedword.startswith('PU'):
            # checking if there was a verb previously in sentence
            if verbcount == True:
                # checking if there was a particle in sentence
                if part == True:
                    verbpartpairs[verbcoord,partcoord] = verbpartpairs[verbcoord,partcoord] + 1 # counting how many particle verbs there are
                    if passive == False: # rejecting passives
                        if noun == True: # must be a noun after the verb and before the next verb or punctuation 
                            #checking if a joint construction is followed by complemetniser,.another prep, etc. to remove some intransitives
                            if wordcountstore == 0:
                                if (loopsent[partsentplace+1][0] in postpart or loopsent[partsentplace+1][1].startswith('CJ') or loopsent[partsentplace+1][1].startswith('PR')):
                                    postpartcount = postpartcount + 1
                                    continue
                                for k in range(len(partadvcont)):
                                    if loopsent[partsentplace][0] == partadvcont[k][0] and loopsent[partsentplace+1][0] == partadvcont[k][1]:
                                        partadvcount = partadvcount + 1
                                        continue
                             # checking if the word between the verb and particle is a pronoun for cases with only one word distance and also all + pro and 'each other' for 2 two word distance 
                            elif wordcountstore == 1 and loopsent[partsentplace-1][1].startswith('PN'):
                                procount = procount + 1    
                                pro = True
                            elif wordcountstore == 2 and loopsent[partsentplace-2][1].startswith('PN') and loopsent[partsentplace-1][0] in allcont: # checks for cases such 'it all', 'you all', etc.
                                        yallcount = yallcount + 1        
                                        yallprostore.append(i)
                                        pro = True
                                        Test1 = True
                            elif wordcountstore == 2 and loopsent[partsentplace-2][0] == 'each' and loopsent[partsentplace-1][0] == 'other':
                                    procount = procount + 1          
                                    pro = True
                            elif wordcountstore == 3 and loopsent[partsentplace-3][1].startswith('PN') and loopsent[partsentplace-1][0] in allcont: # checks for cases such 'it ' ll' and 'you ' ll' with the ' counted as a word
                                    yallcount = yallcount + 1        
                                    yallprostore.append(i)
                                    pro = True
                                    Test1 = True
                            else: 
                                pro = False
                            # storing the sentences, word distances, pronoun coding and token counts                                
                            objpro[verbcoord][partcoord].append(pro)
                            verbpartcoord[verbcoord][partcoord].append(i) 
                            verbpartdis[verbcoord][partcoord].append(wordcountstore) 
                            partverbcount = partverbcount + 1
                        if noun == False:
                            nouncount = nouncount + 1
                    if passive == True:
                        passivecount = passivecount + 1
                        passivesentcoord.append(i)
                elif part == False:
                        nullpart[verbcoord] = nullpart[verbcoord] + 1
                        verbnopartcoord[verbcoord].append(i)  
                # for if using the algorithm to find particle verb verbs
                if writefoundverb == True:
                    if missverbstem not in missverbstemcont: 
                        missverbstemcont.append(missverbstem)
                        missverbcont.append(missverb)
            # resetting logic variables and running counts                        
            verb = False
            noun = False
            part = False
            passive = False
            pro = False
            verbcount = False
            comppart = False
            midwordpart = False
            waywordpart = False
            nopostverb = False
            wordcount = -1
            verbsentplace = -1
            # checking if this word is a verb
            if taggedword.startswith('V'):
                totprocverb = totprocverb + 1
                wordcount = -1 
                verb = True
                # checking for passives by looking at BE + past particple
                if taggedword.startswith('VVN'):
                    if loopsent[j-1][1] != None and loopsent[j-1][0] != None:
                        if loopsent[j-1][1].startswith('VB'):
                            passive = True
                # stemming the verb                            
                stemword = stemmer.stem(loopsent[j][0])
                missverbstem = stemword
                missverb = loopsent[j][0]         
                # checking which verb it is and recording it
                for k in range(noverb):
                    if stemword == verbstem[k]:
                        verbsentplace = j
                        verbcount = True
                        procverb[k] = procverb[k] + 1
                        verbcoord = k
                        verbsentcoord[k].append(i) 
                # checkng selection critera that follows the verb                        
                if loopsent[j][0] != loopsent[-1][0]: 
                    if loopsent[j+1][1] != None:
                        if (verb == True and loopsent[j+1][0] in postverb) or loopsent[j+1][1].startswith('PR') or loopsent[j+1][1].startswith('CJ'):
                            nopostverb = True
                        elif stemmer.stem(loopsent[j+1][0]) in particlestem and not loopsent[j+1][1].startswith('AVP'):
                            nopostverb = True
                    elif stemmer.stem(loopsent[j+1][0]) in particlestem:
                        nopostverb = True
                    continue
                
        #checking if there is a noun in the sentence
        elif taggedword.startswith('PN') or taggedword.startswith('NN'):
            noun = True
        # checking of there is a subordinating conjuction between verb and particle
        elif loopsent[j][0] in midpart or loopsent[j][1].startswith('CJS'):
            midwordpart = True
        #checking if there is an instance of 'way' between verb and particle
        elif loopsent[j][0] in waypart:
            waywordpart = True
            
        # checking if the word is a particle  
        elif taggedword == 'AVP': 
            # to avoid the code breaking 
            if loopsent[verbsentplace+1][1] != None:
                if loopsent[verbsentplace+1][1].startswith('PN') and wordcount > 2:
                    continue
            # checking previous criteria to see if to reject this particle                
            if comppart == True:
                continue
            elif midwordpart == True:
                midcount = midcount + 1
                continue
            elif waywordpart == True:
                waycount = waycount + 1
                continue
            elif nopostverb == True:
                postverbcount = postverbcount + 1
                continue
            elif verbcoord == -1:
                continue
            # checking new criteria for rejection
            elif stemmer.stem(loopsent[j-1][0]) in prepartstem:
                prepartcount = prepartcount + 1
                continue
            elif loopsent[j-1][1] != None:
                if loopsent[j-1][1].startswith('PR') or loopsent[j-1][1].startswith('CJ') or loopsent[j-1][1].startswith('DT') or ('AJ0' in loopsent[j-1][1] and '-' in loopsent[j-1][1]):
                    continue    
            # additional crtieria checks if it is not the last word in the sentence                
            if not lastword:
                andconst = False           
                # checking for Prt and Prt constructions in two ways depending on how many words left in sentence
                if j + 2 < noloopsent:
                    for k in range(len(andpartcont)):
                        if stemmer.stem(loopsent[j][0]) == andpartcont[k][0] and stemmer.stem(loopsent[j+1][0]) == 'and' and stemmer.stem(loopsent[j+2][0]) == andpartcont[k][1]:
                            andconst = True
                            andpartcount = andpartcount + 1
                            break    
                        elif stemmer.stem(loopsent[j][0]) == andpartcont[k][1] and stemmer.stem(loopsent[j+1][0]) == 'and' and stemmer.stem(loopsent[j+2][0]) == andpartcont[k][0]:
                            andconst = True
                            andpartcount = andpartcount + 1
                            break    
                    if andconst == True:
                        andconst = False
                        continue
                if j > 3:
                    for k in range(len(andpartcont)):
                        if stemmer.stem(loopsent[j-2][0]) == andpartcont[k][0] and stemmer.stem(loopsent[j-1][0]) == 'and' and stemmer.stem(loopsent[j][0]) == andpartcont[k][1]:
                            andconst = True
                            andpartcount = andpartcount + 1
                            break    
                        elif stemmer.stem(loopsent[j-2][0]) == andpartcont[k][1] and stemmer.stem(loopsent[j-1][0]) == 'and' and stemmer.stem(loopsent[j][0]) == andpartcont[k][0]:
                            andconst = True
                            andpartcount = andpartcount + 1
                            break    
                    if andconst == True:
                        andconst = False
                        continue
                    
                # stems the particle
                stemword = stemmer.stem(loopsent[j][0]) 
                
                # checking more criteria of rejection
                for k in range(len(ordertimepartcont)):
                    if stemword == ordertimepartcont[k][0]:
                        if stemmer.stem(loopsent[j - 1][0]) == ordertimepartcont[k][1]:
                            if not loopsent[j - 1][1].startswith('V') or '-' in loopsent[j - 1][1]:
                                errorword = True
                                timepartcount = timepartcount + 1
                if errorword == True:
                    errorword = False
                    continue
                for k in range(len(orderdispartcont)):
                    if stemword == orderdispartcont[k][0]:
                        if stemmer.stem(loopsent[j - 1][0]) == orderdispartcont[k][1]:
                            if not loopsent[j - 1][1].startswith('V') or '-' in loopsent[j - 1][1]:
                                errorword = True
                                dispartcount = dispartcount + 1
                if errorword == True:
                    errorword = False
                    continue
                for k in range(len(orderidiompartcont)):
                    if stemword == orderidiompartcont[k][0]:
                        if stemmer.stem(loopsent[j - 1][0]) == orderidiompartcont[k][1]:
                            if not loopsent[j - 1][1].startswith('V') or '-' in loopsent[j - 1][1]:
                                errorword = True
                                idiompartcount = idiompartcount + 1
                if errorword == True:
                    errorword = False
                    continue
                for k in range(len(ordernompartcont)):
                    if stemword == ordernompartcont[k][0]:
                        if stemmer.stem(loopsent[j - 1][0]) == ordernompartcont[k][1]:
                            if not loopsent[j - 1][1].startswith('V') or '-' in loopsent[j - 1][1]:
                                errorword = True
                                nompartcount = nompartcount + 1            
                if errorword == True:
                    errorword = False
                    continue
                
                # checking if there is a previous verb in sentence
                elif verb == True:
                    comppart = False
                    if wordcountstore > 1:
                        if j + 1 < noloopsent:
                            stemword2 = stemmer.stem(loopsent[j+1][0])
                            stemword0 = stemmer.stem(loopsent[j-1][0])
                            # checking for compound prepositons
                            for p in range(len(comppartcont)):
                                if stemword == comppartcont[p][0] and stemword2 == comppartcont[p][1]:
                                    comppart = True
                                    part = False
                                    compartcount = compartcount + 1
                                    comppartstore.append([verbcoord, p, i])
                                elif stemword0 == comppartcont[p][0] and stemword == comppartcont[p][1]:
                                    comppart = True
                                    part = False
                                    compartcount = compartcount + 1
                                    comppartstore.append([verbcoord, p, i])
                            if comppart == True:
                                continue                    
                            for p in range(len(partplacecont)):
                                if stemword == partplacecont[p][0] and stemword2 == partplacecont[p][1]:
                                    comppart = True
                                    part = False
                                    partplacecount = partplacecount + 1
                                    comppartstore.append([verbcoord, p, i])
                            if comppart == True:
                                continue
                        else:
                            stemword0 = stemmer.stem(loopsent[j-1][0])
                            for p in range(len(comppartcont)):
                                if stemword0 == comppartcont[p][0] and stemword == comppartcont[p][1]:
                                    comppart = True
                                    part = False
                                    compartcount = compartcount + 1
                                    comppartstore.append([verbcoord, p, i])
                            if comppart == True:
                                continue                    
                    # to avoid the code breaking from untagged words        
                    if loopsent[j-1][1] != None:
                        if wordcount > 2 and loopsent[j-1][1].startswith('PN'):
                            continue
                    # checking which particle is being looked at and recording it
                    for p in range(nopart): # counting over each of the particles in question
                        if particlestem[p] == stemword: # matching the particle in the data with the partiles under investigation  
                            procpart[p] = procpart[p] + 1
                            partcoord = p
                            part = True  
                            wordcountstore = wordcount
                            partsentplace = j
                            break
                    if part == True and passive == True and wordcount != 0:
                       # print(wordcount, verbcoord, partcoord)
                       falsepassive = falsepassive + 1
                    elif part == True:
                        continue
            # checking same criteria as just above but adapted to work for when the particle is the last word in the sentence  
            elif lastword:
                if loopsent[j-1][1] != None:
                    # removing instances of pro + part beyond 2 word dis
                    if wordcount > 2 and loopsent[j-1][1].startswith('PN'): 
                        continue
                for k in range(len(ordertimepartcont)):
                    if stemword == ordertimepartcont[k][0]:
                        if stemmer.stem(loopsent[j - 1][0]) == ordertimepartcont[k][1]:
                            if not loopsent[j - 1][1].startswith('V') or '-' in loopsent[j - 1][1]:
                                errorword = True
                                timepartcount = timepartcount + 1
                if errorword == True:
                    errorword = False
                    continue
                for k in range(len(orderdispartcont)):
                    if stemword == orderdispartcont[k][0]:
                        if stemmer.stem(loopsent[j - 1][0]) == orderdispartcont[k][1]:
                            if not loopsent[j - 1][1].startswith('V') or '-' in loopsent[j - 1][1]:
                                errorword = True
                                dispartcount = dispartcount + 1
                if errorword == True:
                    errorword = False
                    continue
                for k in range(len(orderidiompartcont)):
                    if stemword == orderidiompartcont[k][0]:
                        if stemmer.stem(loopsent[j - 1][0]) == orderidiompartcont[k][1]:
                            if not loopsent[j - 1][1].startswith('V') or '-' in loopsent[j - 1][1]:
                                errorword = True
                                idiompartcount = idiompartcount + 1
                if errorword == True:
                    errorword = False
                    continue
                for k in range(len(ordernompartcont)):
                    if stemword == ordernompartcont[k][0]:
                        if stemmer.stem(loopsent[j - 1][0]) == ordernompartcont[k][1]:
                            if not loopsent[j - 1][1].startswith('V') or '-' in loopsent[j - 1][1]:
                                errorword = True
                                nompartcount = nompartcount + 1
                if errorword == True:
                    errorword = False
                    continue
                
                andconst = False
                if j > 3:
                    for k in range(len(andpartcont)):
                        if stemmer.stem(loopsent[j-2][0]) == andpartcont[k][0] and stemmer.stem(loopsent[j-1][0]) == 'and' and stemmer.stem(loopsent[j][0]) == andpartcont[k][1]:
                            andconst = True
                            andpartcount = andpartcount + 1
                            break    
                        elif stemmer.stem(loopsent[j-2][0]) == andpartcont[k][1] and stemmer.stem(loopsent[j-1][0]) == 'and' and stemmer.stem(loopsent[j][0]) == andpartcont[k][0]:
                            andconst = True
                            andpartcount = andpartcount + 1
                            break    
                    if andconst == True:
                        andconst = False
                        continue 
                
                if loopsent[j-1][1] != None:
                    if any(loopsent[j-1][1].startswith('PR'), loopsent[j-1][1].startswith('CJ'), loopsent[j-1][1].startswith('DT'), ('AJ0' in loopsent[j-1][1] and '-' in loopsent[j-1][1])):
                        continue
                if not any([comppart, midwordpart, nopostverb, verbcoord == -1, stemmer.stem(loopsent[j-1][0]) in prepartstem, errorword]):
                    for p in range(nopart):
                        if particlestem[p] == stemword: 
                            procpart[p] = procpart[p] + 1
                            partcoord = p
                            part = True  
                            wordcountstore = wordcount
                            partsentplace = j
                            break
          
        # to check the sentence for a particle verb once the algorithm reaches the final word (incase the above criteria didn't trigger the algorithm to check) note it is the same code as above that trigers when a verb or punctuation is found
        elif lastword:
            if verbcount == True:
                if part == True:
                    verbpartpairs[verbcoord,partcoord] = verbpartpairs[verbcoord,partcoord] + 1 # counting how many particle verbs there are
                    if passive == False:
                        if noun == True:
                            if wordcountstore == 0:
                                if (loopsent[partsentplace+1][0] in postpart or loopsent[partsentplace+1][1].startswith('CJ') or loopsent[partsentplace+1][1].startswith('PR')):
                                    postpartcount = postpartcount + 1
                                    continue
                            if wordcountstore == 1 and loopsent[partsentplace-1][1].startswith('PN'): # checking if the word between the verb and particle is a pronoun for cases with only one word
                                procount = procount + 1
                                pro = True
                            elif wordcountstore == 2 and loopsent[partsentplace-2][1].startswith('PN') and loopsent[partsentplace-1][0] in allcont: # checks for cases such 'it all' and 'you all'                               
                                    yallcount = yallcount + 1      
                                    yallprostore.append(i)   
                                    Test2 = True                                           
                                    pro = True
                            elif wordcountstore == 2 and loopsent[partsentplace-2][0].startswith('each') and loopsent[partsentplace-1][0].startswith('other'):
                                    procount = procount + 1                                                           
                                    pro = True
                            elif wordcountstore == 3:
                                if loopsent[partsentplace-3][1].startswith('PN') and loopsent[partsentplace-1][0] in allcont: # checks for cases such 'it all' and 'you all'                               
                                    yallcount = yallcount + 1      
                                    yallprostore.append(i)   
                                    Test2 = True                                           
                                    pro = True
                            else: 
                                pro = False
                            objpro[verbcoord][partcoord].append(pro)
                            verbpartcoord[verbcoord][partcoord].append(i)
                            verbpartdis[verbcoord][partcoord].append(wordcountstore) 
                            partverbcount = partverbcount + 1
                        if noun == False:
                            nouncount = nouncount + 1
                    if passive == True:
                        passivecount = passivecount + 1
                        passivesentcoord.append(i)
                elif part == False:
                        nullpart[verbcoord] = nullpart[verbcoord] + 1
                        verbnopartcoord[verbcoord].append(i)  
                if writefoundverb == True:
                    if missverbstem not in missverbstemcont: 
                        missverbstemcont.append(missverbstem)
                        missverbcont.append(missverb)
                   
print('Finished!')
print('')
end_time = time.time()
total_time = (np.round(end_time - initial_time,2))
wordtime = (np.round(procword/total_time,2))

print('That took '+str(total_time)+' seconds to execute.')
print('Making the program run on an average of '+str(wordtime)+' words per seconds.')
print('')

print('Just storing away the sentences and packaging the data...')

# =============================================================================
# combining irregular verb data
# =============================================================================
irregularverbs = [['be', 'are', 'is', 'was', 're', 'were', 'been', 'am' ], 
                    ['add','ad'],
                    ['begin', 'began'],
                    ['break' , 'broke', 'broken'],
                    ['bring', 'brought'],
                    ['beat', 'beaten'],
                    ['bear', 'bore', 'born'],
                    ['burn', 'burnt'],
                    ['buy', 'bought'],
                    ['build', 'built'],
                    ['catch','caught'],
                    ['come', 'came'],
                    ['become', 'became'],
                    ['do', 'did', 'done', 'does'],
                    ['dig', 'dug'],
                    ['draw', 'drew', 'drawn'],
                    ['driven', 'drived', 'drove', 'drive'],
                    ['drink','drank','drunk'],
                    ['dream', 'dreamt'],
                    ['eat', 'ate', 'eaten'],
                    ['feel', 'felt'],
                    ['fall', 'fell', 'fallen'],
                    ['find', 'found'],
                    ['fed', 'feed'],
                    ['fight', 'fought'],
                    ['get', 'got', 'gotten'],
                    ['grow', 'grew', 'grown'],
                    ['give', 'gave', 'given'],
                    ['go', 'went', 'gone', 'goes', 'goin'],
                    ['have', 'had', 'has'],
                    ['hang', 'hung'],
                    ['hear', 'heard'],
                    ['hold', 'held'],
                    ['keep', 'kept'],
                    ['know', 'knew', 'known'],
                    ['leave', 'left'],
                    ['lead', 'led'],
                    ['lay', 'lain', 'laid'],
                    ['lose', 'lost'],
                    ['light', 'lit'],
                    ['make', 'made'],
                    ['mean', 'meant'],
                    ['meet', 'met'],
                    ['pay', 'paid'],
                    ['run', 'ran'],
                    ['rise', 'rose', 'risen'],
                    ['ring', 'rang', 'rung'],
                    ['say', 'said'],
                    ['see', 'saw', 'seen'],
                    ['stick', 'stuck'],
                    ['show', 'shown'],
                    ['sell', 'sold'],
                    ['send', 'sent'],
                    ['sit', 'sat'],
                    ['spell', 'spelt'],
                    ['speak', 'spoke', 'spoken'],
                    ['spend', 'spent'],
                    ['stand', 'stood'],
                    ['swing', 'swung'],
                    ['shot', 'shoot'],
                    ['slide', 'slid'],
                    ['take', 'took', 'taken'],
                    ['tell', 'told'],
                    ['throw', 'threw', 'thrown'],
                    ['tear', 'tore', 'torn'],
                    ['think', 'thought'],
                    ['wear', 'wore', 'worn'],
                    ['win', 'won'],
                    ['write', 'wrote', 'written'],
                    ['wake', 'woke', 'woken']
                    ]

# some code to combine irregular verbs into single data entries

for i in range(len(irregularverbs)):
    for j in range(len(irregularverbs[i])):
        irregularverbs[i][j] = stemmer.stem(irregularverbs[i][j])
        
for l in range(len(irregularverbs)):
    irregularverbindex = []
    
    for i in range(len(verbstem)):
        if verbstem[i] in irregularverbs[l]:
            if verbstem[i] == irregularverbs[l][0]:
                irregularverbindex.insert(0, i)
            else: 
                irregularverbindex.append(i)
    
    for j in range(len(comppartstore)):
        if comppartstore[j][0] in irregularverbindex[1:]:
            comppartstore[j][0] =  irregularverbindex[0]
      
    for i in range(len(irregularverbindex) - 1):
        j = i + 1
        for n in range(nopart):
            verbpartpairs[irregularverbindex[0]][n] = verbpartpairs[irregularverbindex[0]][n] + verbpartpairs[irregularverbindex[j]][n]
            for k in range(len(objpro[irregularverbindex[j]][n])):
                    objpro[irregularverbindex[0]][n].append(objpro[irregularverbindex[j]][n][k])
                    verbpartdis[irregularverbindex[0]][n].append(verbpartdis[irregularverbindex[j]][n][k])
                    verbpartcoord[irregularverbindex[0]][n].append(verbpartcoord[irregularverbindex[j]][n][k])
                    
                    
    irregularverbindex.remove(irregularverbindex[0])
    irregularverbindex.sort(reverse=True)
    j = 0
    for i in range(len(irregularverbindex)):
        verbpartpairs = np.delete(verbpartpairs, irregularverbindex[i], 0)
        del objpro[irregularverbindex[i]]
        del verbpartdis[irregularverbindex[i]]
        del verbpartcoord[irregularverbindex[i]] 
        verbstem.remove(irregularverbs[l][i + 1])   
        for k in range(len(comppartstore)):
            if comppartstore[k][0] > irregularverbindex[i]:
                comppartstore[k][0] = comppartstore[k][0] - 1
        j = j + 1
        
# a variable to count how many verbs were found in the data        
noverb = len(verbstem)

# some code to extract the sentences from their corrosponding coordinates
verbsent = [] # creating a list to store the sentences that include the verb we are looking for
verbnopartlist = [] # creating a list to store the sentences that don't have a particle
partsentlist = [] # creating a list to store the sentences that include the particles we are looking for
verbpartlist = [] # creating a list to store the sentences that include the particle verb we are looking for
verbpartsent = [] # to store the final sentences
for n in range(nopart): # making a seperate list for each particle in question
    partsentlist.append([])
    verbpartlist.append([])
    verbpartsent.append([])
for i in range(noverb):
        verbpartsent.append([])        
        for j in range(nopart):
                verbpartsent[i].append([])      


# some code to convert sentences coordiantes to lists
sentences = BNCCorpusReader(root=corpusroot, fileids=fileids, lazy=True).sents() # importing the BNC corpus
verbpartlist = multiverbappendsentlist(sentences, verbpartcoord, noverb, nopart)

if storesentences:
    verbsent = multiverbappendsentlist(sentences, verbsentcoord, noverb)
    verbnopartlist = multiverbappendsentlist(sentences, verbnopartcoord, noverb)
    partsentlist = appendsentlist(sentences, partsentcoord, nopart)

# converting the particle verb sentnces into a single string each
for i in range(noverb):
    for j in range(nopart):
        for k in range(len(verbpartlist[i][j])):
            wholesent = ''
            for l in range(len(verbpartlist[i][j][k])):
                wholesent = wholesent + ' ' + verbpartlist[i][j][k][l]
            verbpartsent[i][j].append(wholesent)   
            

print('The sentences are all stored away!')

# putting the verb + particle combinations into an array
verbpartarray = dataarray(verbpartpairs, nullpart, nullverb, noverb, nopart)
                   
# making an index key for the verbs and particles
verbkey = {}
partkey = {}
for i in range(noverb):
    verbkey[verbstem[i]] = i
verbkey['null'] = noverb
for j in range(nopart):
    partkey[particlestem[j]] = j
partkey['null'] = nopart

# =============================================================================
# writing the data to file
# =============================================================================

rootpath = rootpath + folderpath


# writing the list of verbs found 
if writefoundverb == True:     
    # this code (immediately below) was adapted from the tutorial at: https://www.programiz.com/python-programming/csv
    csv.register_dialect('datadialect', delimiter='|', quoting=csv.QUOTE_NONNUMERIC, quotechar='\'') 
    foundverbpath = rootpath + foundverbs
    with open(foundverbpath, 'w', newline='') as file:
        writer = csv.writer(file, dialect = 'datadialect')  
        for i in range(len(missverbcont)):
            verbappend = [missverbcont[i]]
            writer.writerow(verbappend)


# writing data to CSV files
if writedata == True:
    csv.register_dialect('datadialect', delimiter='|', quoting=csv.QUOTE_NONNUMERIC, quotechar='\'')
    
    # storing yall pronoun senttences
    if yallprowrite == True:
        yallcont = []
        for i in range(len(yallprostore)):
            wholesent = ''
            for j in range(len(sentences[yallprostore[i]])):
                wholesent = wholesent + ' ' + sentences[yallprostore[i]][j]
            yallcont.append([wholesent])
            
        yallpath = rootpath + yallfile
        
        # this code (immediately below) was adapted from the tutorial at: https://www.programiz.com/python-programming/csv
        with open(yallpath, 'w', newline='') as file:
            writer = csv.writer(file, dialect = 'datadialect')
            for i in range(len(yallcont)):
                    writer.writerow(yallcont[i])  
    
    # writing out passive sentences
    if writepassive:
        # converting the passaive sentnces into a single string and storing them
        passivesent = []
        for i in range(len(passivesentcoord)):
            wholesent = ''
            for j in range(len(sentences[passivesentcoord[i]])):
                wholesent = wholesent + ' ' + sentences[passivesentcoord[i]][j]
            passivesent.append([wholesent])
    
        passivepath = rootpath + passivefile
        # this code (immediately below) was adapted from the tutorial at: https://www.programiz.com/python-programming/csv
        with open(passivepath, 'w', newline='') as file:
            writer = csv.writer(file, dialect = 'datadialect')
            for i in range(len(passivesent)):
                    writer.writerow(passivesent[i])  
    
    # writing out compound prepositon sentences
    if writecomppart:            
        # ordering the compound prepostion data
        comppartstore.sort(key=lambda x: x[0])   
    
        verbcomppart = 0
        verbcomppartlist = []
        verbcomppartrange = []
        ordercomppart = []
        for i in range(len(comppartstore)):
            if comppartstore[i][0] == verbcomppart:
                verbcomppartlist.append(i)
                continue
            if comppartstore[i][0] != verbcomppart:
                while comppartstore[i][0] != verbcomppart:
                    verbcomppart = verbcomppart + 1
                if comppartstore[i][0] == verbcomppart and i > 1:
                    verbcomppartrange.append([verbcomppartlist[0], verbcomppartlist[-1]])
                    verbcomppartlist = []
                    verbcomppartlist.append(i)
                else: 
                    continue
        for i in range(len(verbcomppartrange)):
            tempcont = comppartstore[verbcomppartrange[i][0]:(verbcomppartrange[i][1] + 1)] 
            tempcont.sort(key=lambda x: x[1])
            ordercomppart.extend(tempcont)
            
        # converting the compound prepsition sentnces into a single string and storing them
        comppartsent = []
        for i in range(len(ordercomppart)):
            wholesent = ''
            for j in range(len(sentences[ordercomppart[i][2]])):
                wholesent = wholesent + ' ' + sentences[ordercomppart[i][2]][j]
            comppartsent.append([verbstem[ordercomppart[i][0]], comppartcont[ordercomppart[i][1]][0], comppartcont[ordercomppart[i][1]][1], wholesent])
    
        comppartpath = rootpath + comppartfile
        # this code (immediately below) was adapted from the tutorial at: https://www.programiz.com/python-programming/csv
        with open(comppartpath, 'w', newline='') as file:
            writer = csv.writer(file, dialect = 'datadialect')
            for i in range(len(comppartsent)):
                    writer.writerow(comppartsent[i])  

    # storing the array of token counts
    arraypath = rootpath + arrayfile
    # this code (immediately below) was adapted from the tutorial at: https://www.programiz.com/python-programming/csv
    with open(arraypath, 'w', newline='') as file:
        writer = csv.writer(file, dialect = 'datadialect')
        for i in range(noverb + 1):
            row = []
            for j in range(nopart + 1):
                row.append(verbpartarray[i][j])
            writer.writerow(row)

    # writing the total number of verbs counted
    totalverb = ['Total Verb Count']
    totalverb.append(totprocverb)
    totalverbpath = rootpath + totalverbfile
    # this code (immediately below) was adapted from the tutorial at: https://www.programiz.com/python-programming/csv
    with open(totalverbpath, 'w', newline='') as file:
        writer = csv.writer(file, dialect = 'datadialect')        
        writer.writerow(totalverb)
    
    # writing the key for verbs
    verbkeypath = rootpath + verbkeyfile
    verbstem.append('null')
    # this code (immediately below) was adapted from the tutorial at: https://www.programiz.com/python-programming/csv
    with open(verbkeypath, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames = verbstem, dialect = 'datadialect')
        writer.writeheader()
        writer.writerow(verbkey) 

    # writing the key for particles        
    partkeypath = rootpath + partkeyfile
    particlestem.append('null')
    # this code (immediately below) was adapted from the tutorial at: https://www.programiz.com/python-programming/csv
    with open(partkeypath, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames = particlestem, dialect = 'datadialect')
        writer.writeheader()
        writer.writerow(partkey) 
        
    # writing each token's sentence, verb, particle and pronoun coding into a csv file
    for i in range(noverb):
        verbpartpath = rootpath + particleverbfolder + verbstem[i] + '_' + verbpartfile
        # this code (immediately below) was adapted from the tutorial at: https://www.programiz.com/python-programming/csv
        with open(verbpartpath, 'w', newline='') as file:
            writer = csv.writer(file, dialect = 'datadialect')
            for j in range(nopart):
                flagk = 0
                for k in range(len(verbpartdis[i][j])): 
                    row = []
                    row.append(particlestem[j])
                    row.append(verbpartdis[i][j][k])
                    row.append(objpro[i][j][k])                
                    row.append(verbpartsent[i][j][k])          
                    writer.writerow(row)  
 
sent_time = time.time()
total_sent_time = (np.round(sent_time - end_time,2))

print('It took '+str(total_sent_time)+' seconds to store the sentences and package the data.')

print(''.ljust(terminalwidth,'%'))
        
print('The number of processed words is '+str(procword))
print('The number of processed sentences is '+str(procsent))


print(''.ljust(terminalwidth,'%'))


