import sys
from datetime import datetime
import numpy as np
import pandas as pd
import math
import itertools

def epoch2datetime(epoch):
    return datetime.fromtimestamp(
        int(epoch)
    ).strftime('%Y-%m-%d %H:%M:%S')

def time_diff(log1, log2): # unit (sec)
    return (log2['datetime'] - log1['datetime']).total_seconds()


def distance(log1, log2): # unit (kilometer)
    if isinstance(log1, pd.Series):
        p1 = (log1['lon'], log1['lat'])
        p2 = (log2['lon'], log2['lat'])
    else:
        p1, p2 = log1, log2  # [long, lat]

    if p1 == p2: # special case: p1 = p2
        return 0.0
    # Great-circle distance
    phi_s = math.radians(p1[1])
    phi_f = math.radians(p2[1])
    delta_lampda = math.radians(p1[0]) - math.radians(p2[0])
    delta_phi = phi_s - phi_f
    delta_sigma = 2 * math.asin(math.sqrt( math.sin(delta_phi / 2) * math.sin(delta_phi / 2) + math.cos(phi_s) * math.cos(phi_f) * math.sin(delta_lampda / 2) * math.sin(delta_lampda / 2)))
    return delta_sigma * 6378.137 # unit (kilometer)


def speed(log1, log2): # unit (km/hr)
    if time_diff(log1, log2) == 0:
        return np.nan
    return distance(log1, log2)*3600/time_diff(log1, log2)


def load_data(dataPath):
    """
    return DataFrame(datetime, cell, lat, lon), List(idx: cid, value: (lat, lon))
    """
    data = []
    cellTable = []
    with open(dataPath, 'r') as f:
        for log in f.readlines():
            log = log.rstrip().split('|')
            epoch = log[4]
            lat = float(log[3])
            lon = float(log[2])
            if (lon, lat) not in cellTable:
                cellTable.append((lon, lat))
            data.append([pd.to_datetime(epoch2datetime(epoch)), cellTable.index((lon, lat)), lon, lat, epoch])    
    data = sorted(data, key=lambda x: x[0])
    dataDF = pd.DataFrame(data, columns=['datetime', 'cid', 'lon', 'lat', 'epoch'])
    return dataDF, cellTable


class StablePeriod:

    def __init__(self, df):
        self.sequences = df.reset_index()
        self.first = self.sequences.ix[0]
        self.last = self.sequences.ix[len(self.sequences)-1]
        self.cid = self.sequences.ix[0, 'cid']
        self.duration = time_diff(self.first, self.last)


class SuspiciousSequence:

    def __init__(self, df):
        self.sequences = df.reset_index()
        self.C = set(self.sequences['cid'])

    def check(self): # whether the sequence contains a cycle of events
        cids = []
        for i in self.sequences['cid']:
            #print cids, i
            if i in cids[:-1] and set(cids) >= 2:
                #print "contains a cycle!"
                return True
            cids.append(i)
        #print "didn't contains a cycle!"
        return False

    def remove(self, cellTable):
        if self.check():
            score = {} # cid -> score
            for c1 in self.C:
                fc = sum(self.sequences['cid'] == c1)
                dc = np.mean(map(lambda i: distance(cellTable[c1], cellTable[i]), self.C - set([c1])))
                score[c1] = fc/dc
            score = pd.Series(score)
            #print score.idxmax()
            return list(self.sequences[self.sequences['cid'] != score.idxmax()]['index'])
        else:
            return []


def detect_stable_period(dataDF, L): # L(min)
    """
    return SP List
    """
    SP = []
    sid = 0
    for i in range(len(dataDF)-1):
        if dataDF.ix[i, 'cid'] != dataDF.ix[i+1, 'cid']:
            #print i, sid
            if (i != sid) and (time_diff(dataDF.ix[sid], dataDF.ix[i]) > L*60):
                SP.append(StablePeriod(dataDF[sid:i+1]))
                #print SP[-1].sequences
            sid = i+1
    return SP


def calculate_d_v_t_t(dataDF):
    d = [np.nan]
    v = [np.nan]
    t = [0]
    increment = 0 
    for i in range(len(dataDF)-1):
        d.append(distance(dataDF.ix[i], dataDF.ix[i+1]))
        v.append(speed(dataDF.ix[i], dataDF.ix[i+1]))
        increment += time_diff(dataDF.ix[i], dataDF.ix[i+1])
        t.append(increment)
    return (d, v, t)


def detect_heuristic_1(SP, L1T): # L1T(min)
    h1 = []
    for i in range(len(SP)-1):
        if (SP[i].cid == SP[i+1].cid) and (time_diff(SP[i].last, SP[i+1].first) < L1T*60):
            h1.extend(range(SP[i].last['index']+1, SP[i+1].first['index']))
    return h1


def detect_heuristic_2(dataDF, SP, L2T, L2D): # L2T(min), L2D(km)
    h2 = []
    for i in range(len(SP)):
        Rj = dataDF.ix[SP[i].last['index']+1]
        if (time_diff(SP[i].last, Rj) < L2T*60) and (Rj['distance'] > L2D): #distance(SP[i].last, Rj)
            h2.append(SP[i].last['index']+1)
    return h2


def detect_heuristic_3(dataDF, V, L3): # V(km/h), L3(km)
    h3 = []
    for i in range(len(dataDF)-2):
        if (dataDF.ix[i+1, 'speed'] * dataDF.ix[i+2, 'speed'] > V * V) and (
            dataDF.ix[i+1, 'distance'] > L3) and (
            dataDF.ix[i+2, 'distance'] > L3) and (
            distance(dataDF.ix[i], dataDF.ix[i+2]) < L3/2):
            h3.append(i+1)
    return h3


def detect_heuristic_4(dataDF, a, b, c):
    i = 0
    j = 1
    SS = []
    while i < len(dataDF) and j < len(dataDF) and i < j:
        startlog = dataDF.ix[i]
        endlog = dataDF.ix[j]
        t = endlog['increment_time'] - startlog['increment_time']
        if (t <= a * 60):
            j += 1
            continue
        else: #(t > a * 60):
            if (j-i >= b) and (len(set(dataDF.ix[i:j-1, 'cid'])) >= c):
                if dataDF.ix[j-1]['increment_time'] - startlog['increment_time'] <= a * 60:
                    #print i, j
                    #print dataDF.ix[i:j-1, [0,1,6]]
                    SS.append(SuspiciousSequence(dataDF[i:j]))
                    #print SS[-1].sequences
                    i = j
                    j += 1
            i += 1
            j += 1
    return SS


def write_data(dataPath, data):
    data.to_csv(dataPath, columns=['cid', 'lon', 'lat', 'epoch'], header=True, index=False)


def main(dataPath):

    dataDF, cellTable = load_data(dataPath)
    SP = detect_stable_period(dataDF, 0)
    (d, v, t) = calculate_d_v_t_t(dataDF)
    dataDF['distance'] = d
    dataDF['speed'] = v
    dataDF['increment_time'] = t
    #for i in range(len(SP)):
    #    print SP[i].sequences
    h1 = detect_heuristic_1(SP, 1)
    print "h1", len(h1)
    h2 = detect_heuristic_2(dataDF, SP, 1, 3)
    print "h2", len(h2)
    SS = detect_heuristic_4(dataDF, 2, 3, 3)
    h4 = reduce(lambda lst1, lst2: lst1+lst2, [ss.remove(cellTable) for ss in SS]) if SS else []
    print "h4", len(h4)
    h3 = detect_heuristic_3(dataDF, 300, 5)
    print "h3", len(h3)

    print "total: ", len(h1+h2+h3+h4)
    name = dataPath.split('/')[-1]
    
    write_data('output/'+name, dataDF.drop(h1+h2+h3+h4))


if __name__ == '__main__':

    dataPath = sys.argv[1]
    main(dataPath)
