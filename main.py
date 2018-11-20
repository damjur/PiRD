#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from python_speech_features import mfcc
#https://python-speech-features.readthedocs.io/en/latest/#python_speech_features.base.mfcc
# import scipy.io.wavfile as wav
import wave, struct
import matplotlib.pyplot as plt
import os
from tqdm import tqdm_notebook
from collections import Counter
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

plt.style.use('grayscale')


# In[2]:


def read_wav_and_do_mfcc(filename):
    with wave.open(filename, 'r') as waveFile:
        result = []
        for _ in range(waveFile.getnframes()):
            waveData = waveFile.readframes(1)
            result += [struct.unpack("<h", waveData)]
            
        frames = waveFile.getnframes()
        rate = waveFile.getframerate()
        duration = frames / float(rate)
        
        return mfcc(np.array(result,dtype=np.float32), rate, winlen=0.0125, winstep=0.005, numcep=7, nfilt=7, nfft=512*2)
    
def getFeatures(filename, dirname):
#     print("getFeatures ", dirname)
#     cache_file = "features.npy"
#     if os.path.isfile(cache_file):
#         return np.load(cache_file)
    dirname = os.path.join(".",dirname)
    result = [[] for _ in range(10)]
    for fname in os.listdir(dirname):
        if fname == filename:
            continue
#         print(fname)
        if fname.endswith('.WAV'):
            i = int(fname.split("_")[-2])
            fname = os.path.join(dirname, fname)
            result[i].append(fname)
            
    result = np.array(list(map(lambda fnames:[read_wav_and_do_mfcc(fname) for fname in fnames],result)))
#     np.save(cache_file,result)
    return result


# In[3]:


def L(p):
    def fun(X1,X2):
        return ((np.abs(X1-X2)**p).sum())**(1/p)
    return fun

def build_matrix_of_distances(X1,X2,Lp):
    result = np.zeros((X1.shape[0],X2.shape[0]))
    for i,vx1 in enumerate(X1):
        for j,vx2 in enumerate(X2):
            result[i,j]=Lp(vx1,vx2)
    return result

def build_matrix_of_cumulated_distances(matrix_of_distances):
    result = np.zeros(matrix_of_distances.shape)
    for i,row in enumerate(matrix_of_distances):
        for j,cell in enumerate(row):
            result[i,j] += matrix_of_distances[i,j]
            if i==0 and j==0:
                continue
            elif j==0:
                result[i,j] += result[i-1,j]
            elif i==0:
                result[i,j] += result[i,j-1]
            else:
                result[i,j] += np.min([result[i-1,j],result[i,j-1],result[i-1,j-1]])
    return result

def build_shortest_path(matrix_of_cumulated_distances):
    a,b = matrix_of_cumulated_distances.shape
    
    a,b = a-1,b-1
    result = [[a,b]]
    while(a!=0 or b!=0):
        if a==0:
            b -= 1
        elif b==0:
            a -= 1
        else:
            if matrix_of_cumulated_distances[a-1,b-1]<=matrix_of_cumulated_distances[a-1,b] and matrix_of_cumulated_distances[a-1,b-1]<=matrix_of_cumulated_distances[a,b-1]:
                a-=1
                b-=1
            elif matrix_of_cumulated_distances[a-1,b]<=matrix_of_cumulated_distances[a-1,b-1] and matrix_of_cumulated_distances[a-1,b]<=matrix_of_cumulated_distances[a,b-1]:
                a-=1
            elif matrix_of_cumulated_distances[a,b-1]<=matrix_of_cumulated_distances[a-1,b-1] and matrix_of_cumulated_distances[a,b-1]<=matrix_of_cumulated_distances[a-1,b]:
                b-=1
        result += [[a,b]]
    return np.array(result).reshape((len(result),2))

def rank_path(path,shape):
    A,B = shape
    result = 0
    for x in path:
        result += abs(A*x[1]-B*x[0])
#     return result/(A**2+B**2)**0.5
    return 2*result/(A*(B**2 + (A-2)*B + 2 - 2*A))#? normalizacja od 0 do 1 (chyba)

def subdtw(A,B,verbose=False):
    mod = build_matrix_of_distances(A,B,L(1))
    mocd = build_matrix_of_cumulated_distances(mod)
    path = build_shortest_path(mocd)
    rank = rank_path(path,mocd.shape)
    
    if verbose:
        fig, ax = plt.subplots(figsize=(1*len(B),1*len(A)))
        im = ax.imshow(mocd)


        ax.set_xticks(np.arange(len(B)))
        ax.set_yticks(np.arange(len(A)))
        ax.set_xticklabels(B)
        ax.set_yticklabels(A)
        for i,row in enumerate(mocd):
            for j,cell in enumerate(row):
                ax.text(j, i, "{}({})".format(cell,mod[i,j]), ha="center", va="center", color="y")
    #             ax.text(j, i, cell, ha="center", va="center", color="w")
        epsilon = 0.1
        ax.plot(path[:,1]+epsilon,path[:,0],'r',label='Rzeczywiste dopasowanie')
        ax.plot([0,len(B)-1],[0,len(A)-1],'g',label='Idealne dopasowanie')
        bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax.legend( bbox_to_anchor=(1.4,1))
        plt.show()
    return rank

def better_dtw(A,B, verbose=False):
    rank,_ = fastdtw(A, B, dist=euclidean)
    return rank

def dtw(A,B,verbose=False):
    if len(A) <= len(B):
        return better_dtw(A,B,verbose)
    return better_dtw(B,A,verbose)


# In[4]:


def get_aggregator(agg):
    def aggregator(data):
        return np.argmin(agg(data,axis=1))
    return aggregator

def get_kNN_aggregator(k):
    def kNN_aggregator(data):
        return Counter(np.unravel_index(np.argsort(data, axis=None), data.shape)[0][:k]).most_common(1)[0][0]
    return kNN_aggregator

def classify(v,agg=get_aggregator(np.min)):
    return agg(np.array(list(map(lambda row:list(map(lambda cell:dtw(v,cell),row)),features))))

def draw_heatmap(data):
    plt.subplots(figsize=data.shape)
    plt.imshow(data)
#     plt.set_xticks(np.arange(data.shape[0]))
#     plt.set_yticks(np.arange(data.shape[0]))
    plt.colorbar()
    plt.show()


# In[5]:


dirname = "cyfry"
dirname = os.path.join(".",dirname)
filenames= os.listdir(dirname)
for test_filename in filenames:
    print(test_filename)
    features = getFeatures(test_filename, "cyfry")
    # print(features)
    aggregators = [get_kNN_aggregator(1),get_kNN_aggregator(3),get_kNN_aggregator(5)]#,get_aggregator(np.min),get_aggregator(np.mean)]
    for a in aggregators:
        results = np.zeros((10,10))
        if test_filename.endswith('.WAV'):
#             print("check ", fname)
            i = int(test_filename.split("_")[-2])
            fname = os.path.join(dirname, test_filename)
            fv = read_wav_and_do_mfcc(fname)
            dec= classify(fv,a)
            results[i,dec]+=1
        # print(results)
        recall = np.diag(results) / np.sum(results, axis = 1)
        precision = np.diag(results) / np.sum(results, axis = 0)
        print(np.nanmean(precision), np.nanmean(recall))

        # draw_heatmap(results)

