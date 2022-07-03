import itertools
import heapq
import string
import random
import bisect
from queue import PriorityQueue
import numpy as np

def get_hierSeg_WSarcs_arcRemove_Multiband(image,nhood=8,beta=None,doPrint=False):
    def euclDist(v1,v2):
    #     print("here3")
        return np.linalg.norm(np.array(v1) - np.array(v2)) 

    def generateRandomLabel(N = 8):
        return ''.join(random.choices(string.ascii_uppercase , k = N))

    def getNeighbours4(p,shp):
        N=[]
        for i in range(p[0]-1,p[0]+2):
            if i<0: continue
            if i>=shp[0]: continue
            for j in range(p[1]-1,p[1]+2):
                if j<0: continue
                if j>=shp[1]: continue
                if i==p[0] and j==p[1]: continue
                if i==p[0] or j==p[1]: N.append((i,j))
        return N
    def getNeighbours8(p,shp):
        N=[]
        for i in range(p[0]-1,p[0]+2):
            if i<0: continue
            if i>=shp[0]: continue
            for j in range(p[1]-1,p[1]+2):
                if j<0: continue
                if j>=shp[1]: continue
                if i==p[0] and j==p[1]: continue
                N.append((i,j))
        return N

    def getnbrBuckets(arc_bp):
        nbrBuckets={}
    #     print("arc_bp",arc_bp)
        for sp in arc_bp:
            nb1,nb2=arc_bp[sp][0],arc_bp[sp][1]
            if nbrBuckets.get(nb1) is None: nbrBuckets[nb1]=[]
            if nbrBuckets.get(nb2) is None: nbrBuckets[nb2]=[]
            nbrBuckets[nb1].append(sp)
            nbrBuckets[nb2].append(sp)
        return nbrBuckets
    def getNeighbours(x):
        nbrs=[]
        nbrs.extend([n for n in nbrBuckets[arc_bp[x][0]] if n!=x])
        nbrs.extend([n for n in nbrBuckets[arc_bp[x][1]] if n!=x])
        return nbrs

    def getNlist(x,level_zero=False):
        if level==0 or level_zero:
            if nhood==4:
                return getNeighbours4(x,image.shape)
            elif nhood==8:
                return getNeighbours8(x,image.shape)
    #         return getEucNeighbours(x,d=8)
        else:
            return getNeighbours(x)
        
    def getF_image(image):
        dist={}
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for n in getNlist((i,j),level_zero=True):
                    dist[((i,j),n)]=euclDist(image[i,j],image[n])  
        imageF_=np.zeros((image.shape[0],image.shape[1]))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                    imageF_[i,j]=min([dist[((i,j),n)] for n in getNlist((i,j),level_zero=True)]) 
        return imageF_
    def getCenters_imageF_(img):
        Pix=[(i,j) for i in range(img.shape[0]) for j in range(img.shape[1])]
        HQ={}
        for p in Pix:
            i=img[p]
            if HQ.get(i+1) is None:
                HQ[i+1]=[]
            HQ[i+1].append(p)

        shifted={p:False for p in Pix}
        visited={p:False for p in Pix}
        wsMarked={p:False for p in Pix}
        label={p:p for p in Pix}

        hList=list(HQ.keys())
        heapq.heapify(hList)

        centers=[]
        while len(HQ)!=0:
            minH=hList[0]
            p=HQ[minH].pop(0)
            if len(HQ[minH])==0:
                HQ.pop(minH)
                heapq.heappop(hList)
            if visited[p]:
                continue
            if minH>img[p]:
                centers.append(p)
                if HQ.get(img[p]) is None:
                    HQ[img[p]]=[p]
                    heapq.heappush(hList,img[p])
                else:
                    HQ[img[p]].append(p)
                shifted[p]=True
            else:
                visited[p]=True
                for n in getNlist(p,level_zero=True):
                    if label[n]==label[p]:
                        continue
                    if img[n]>=img[p]:
                        if not shifted[n]:
                            label[n]=label[p]
                            shifted[n]=True
                            if HQ.get(img[n]) is None:
                                HQ[img[n]]=[n]
                                heapq.heappush(hList,img[n])
                            else:
                                HQ[img[n]].append(n)
                        else:
                            wsMarked[n]=True
                            visited[n]=True
        return centers
    def getWS_multiBand(image,centers,doPrint=False):
        status={(i,j):'undiscovered' for i in range(image.shape[0]) for j in range(image.shape[1])}
        currentDist={(i,j):-np.Inf for i in range(image.shape[0]) for j in range(image.shape[1])}
        currentLabel={(i,j):np.Inf for i in range(image.shape[0]) for j in range(image.shape[1])}
        Watershed={(i,j):False for i in range(image.shape[0]) for j in range(image.shape[1])}
        visited={(i,j):False for i in range(image.shape[0]) for j in range(image.shape[1])}
        arcPoints={}


        Q=PriorityQueue()
        for c in range(len(centers)):
            Q.put((0,centers[c],c))
            status[centers[c]]='inQ'
            currentDist[centers[c]]=0
            currentLabel[centers[c]]=c
            visited[centers[c]]=True
    #     if doPrint: print("Q",Q.queue)
        while not Q.empty():
            dist_p,p,label_p=Q.get()
            if status[p]=='popped':
                continue

            status[p]='popped'
    #         if doPrint: print('status',status)
            for n in getNlist(p):
                if doPrint: print('   n',n)
                insertInQ=False
                if status[n]!='popped':
                    d=max(currentDist[p],euclDist(image[p],image[n]))
                    if status[n]=='undiscovered':
    #                     if doPrint: print('   undiscovered')
                        status[n]='inQ'
                        insertInQ=True
                    elif status[n]=='inQ':
                        if currentLabel[n]!=label_p:
                            Watershed[n]=True
                            status[n]='popped'
                            arcPoints[n]=d
    #                         arcPoints[n]=currentDist[n]
                        else:
                            if d<currentDist[n]:
                                insertInQ=True
                if insertInQ:
                    Q.put((d,n,label_p))
                    if doPrint: print("insertInQ p,n",p,n)
                    currentDist[n]=d
                    currentLabel[n]=label_p
                    visited[n]=True

        return currentLabel,Watershed,arcPoints

    def getWSarcs3(Pix,img,wsMarked,label,saddleType='mean'):
        WSarcs={}
        arc_bp={}
        saddleLabel={}
    #     ws_bp={}
        saddleValue={}

        bp_arcs={}
        for p in Pix:
            if wsMarked[p]:
                thisNbrBasins=list(set([label[q] for q in getNlist(p) if not wsMarked[q]]))
                thisarc_bps=[(thisNbrBasins[n1],thisNbrBasins[n2]) for n1 in range(len(thisNbrBasins)) for n2 in range(n1+1,len(thisNbrBasins))]
    #             print(p,thisarc_bps)
                for bp in thisarc_bps:
                    bp_alreadyFound=False
                    reverse=False
                    if bp_arcs.get(bp) is not None:
                        bp_alreadyFound=True
                    if bp_arcs.get((bp[1],bp[0])) is not None:
                        bp_alreadyFound=True
                        reverse=True
                    if not bp_alreadyFound:
                        bp_arcs[bp]=[p]

                    else:
                        if not reverse:
                            bp_arcs[bp].append(p)

                        else:
                            bp_arcs[(bp[1],bp[0])].append(p)
    #     print("bp_arcs",bp_arcs)

        w_arcs={}
        basinPair_arcs={}
        for bp in bp_arcs:
            thislabel=generateRandomLabel()
            basinPair_arcs[bp]=thislabel
            WSarcs[thislabel]=bp_arcs[bp]
            arc_bp[thislabel]=bp
            thisSaddle=(np.Inf,'dummy')
            for pix in bp_arcs[bp]:
                if img[pix]<thisSaddle[0]:
                    thisSaddle=(img[pix],pix)
            saddleLabel[thislabel]=thisSaddle[1]
            if saddleType=='min':
                saddleValue[thislabel]=thisSaddle[0]
            if saddleType in ['arc_mean','saddle_mean'] :
                saddleValue[thislabel]=np.mean([img[pix] for pix in WSarcs[thislabel]])
            for w in bp_arcs[bp]:
                if w_arcs.get(w) is None:
                    w_arcs[w]=[thislabel]
                else:
                    w_arcs[w].append(thislabel)

        basin_arcs={}
        basin_neighbourBasins={}
        for arc in arc_bp:
            basin1,basin2=arc_bp[arc]
            if basin_arcs.get(basin1) is None: basin_arcs[basin1]=[arc]
            else: basin_arcs[basin1].append(arc)
            if basin_arcs.get(basin2) is None: basin_arcs[basin2]=[arc]
            else: basin_arcs[basin2].append(arc)

            if basin_neighbourBasins.get(basin1) is None: basin_neighbourBasins[basin1]=[basin2]
            else: basin_neighbourBasins[basin1].append(basin2)
            if basin_neighbourBasins.get(basin2) is None: basin_neighbourBasins[basin2]=[basin1]
            else: basin_neighbourBasins[basin2].append(basin1)

        return WSarcs,arc_bp,saddleLabel,saddleValue,basinPair_arcs,w_arcs,basin_arcs,basin_neighbourBasins # ws_bp

    def getWSpoints(Pix,img,label,factor=None):
        HQ={}
        for p in Pix:
            i=img[p]
            if HQ.get(i+1) is None:
                HQ[i+1]=[]
            HQ[i+1].append(p)

        shifted={p:False for p in Pix}
        visited={p:False for p in Pix}
        wsMarked={p:False for p in Pix}
    #     label={p:p for p in Pix}

        if factor is not None:
            wsPtVal={p:np.Inf for p in Pix}
            factorWS={p:False for p in Pix}
            minimaHeight={}
            saddleHeight={}


        hList=list(HQ.keys())
        heapq.heapify(hList)

        while len(HQ)!=0:
            minH=hList[0]
            p=HQ[minH].pop(0)
            if len(HQ[minH])==0:
                HQ.pop(minH)
                heapq.heappop(hList)
            if visited[p]:
                continue
            if minH>img[p]:
    #             print(p)
                if HQ.get(img[p]) is None:
                    HQ[img[p]]=[p]
                    heapq.heappush(hList,img[p])
                else:
                    HQ[img[p]].append(p)
                shifted[p]=True

                if factor is not None:
                    wsPtVal[p]=img[p]
                    minimaHeight[p]=img[p]
            else:
                visited[p]=True
                for n in getNlist(p):
                    if label[n]==label[p]:
                        continue
                    if not shifted[n]:
                        label[n]=label[p]
                        shifted[n]=True
                        if HQ.get(img[n]) is None:
                            HQ[img[n]]=[n]
                            heapq.heappush(hList,img[n])
                        else:
                            HQ[img[n]].append(n)

                        if factor is not None:
                            minimaHeight[n]=minimaHeight[p]
                            wsPtVal[n]=img[n]
                    else:
                        wsMarked[n]=True
                        visited[n]=True
                        if factor is not None:
                            if saddleHeight.get(label[p]) is None: saddleHeight[label[p]]=img[n]
                            else: saddleHeight[label[p]]=min(saddleHeight[label[p]],img[n])
                            if saddleHeight.get(label[n]) is None: saddleHeight[label[n]]=img[n]
                            else: saddleHeight[label[n]]=min(saddleHeight[label[n]],img[n])

        if factor is not None:     
            if len(saddleHeight)>0:
                for p in Pix:
        #             if not wsMarked[p]:
        #                 print('p',p,'img[p]',img[p],'label[p]',label[p],'minimaHeight[label[p]]',minimaHeight[label[p]],'saddleHeight[label[p]]',saddleHeight[label[p]])
                    if not wsMarked[p] and factor*(saddleHeight[label[p]]-minimaHeight[label[p]])<=(img[p]-minimaHeight[label[p]]):
                        factorWS[p]=True
                        if upgradeFactorWS: wsPtVal[p]=max(img[p],saddleHeight[label[p]])

            #     print('factorWS',len([w for w in factorWS if factorWS[w]]))
            return label,wsMarked,factorWS,wsPtVal,saddleHeight
        else:
            return label,wsMarked
    
    def removeArc(arcInfo, doPrint=False):
        if sortingStep: A=arcInfo[1]
        else: A=arcInfo

        basin1,basin2=arc_bp[A][0],arc_bp[A][1]
        if doPrint: print('A',A,"basin1,basin2",basin1,basin2)
        allNbrBasins=list((set(basin_neighbourBasins[basin1]).union(set(basin_neighbourBasins[basin2]))).difference(set(arc_bp[A])))
        commonNbrBasins=list(set(basin_neighbourBasins[basin1]).intersection(set(basin_neighbourBasins[basin2])))
        nonCommonNbrBasins=list(set(allNbrBasins).difference(set(commonNbrBasins)))
        nonCommonNbrBasins_basin1=list(set(nonCommonNbrBasins).intersection(set(basin_neighbourBasins[basin1])))
        nonCommonNbrBasins_basin2=list(set(nonCommonNbrBasins).intersection(set(basin_neighbourBasins[basin2])))

        newBasinlabel=generateRandomLabel()
        basin_neighbourBasins[newBasinlabel]=[]
        basin_arcs[newBasinlabel]=[]

        if saddleType=='basin_meandiff':
            A_,A_sum=[],0 if image.ndim==2 else np.zeros(image.shape[2])
            for a in WSarcs[A]:
                if len(w_arcs[a])==1 and w_arcs[a][0]==A:
                    A_.append(a)
                    A_sum+=image[a]
            if doPrint: print('A_',A_)
            basin_Points[newBasinlabel]=basin_Points[basin1]+basin_Points[basin2]+A_
            basin_PointCount[newBasinlabel]=basin_PointCount[basin1]+basin_PointCount[basin2]+len(A_)
            basin_PointSum[newBasinlabel]=basin_PointSum[basin1]+basin_PointSum[basin2]+A_sum
            basin_PointAvg[newBasinlabel]=basin_PointSum[newBasinlabel]/basin_PointCount[newBasinlabel]
            basin_Points.pop(basin1)
            basin_Points.pop(basin2)

        if doPrint: 
            print("allNbrBasins",allNbrBasins)
            print("commonNbrBasins",commonNbrBasins)
            print("nonCommonNbrBasins",nonCommonNbrBasins)
            print("nonCommonNbrBasins_basin1",nonCommonNbrBasins_basin1)
            print("nonCommonNbrBasins_basin2",nonCommonNbrBasins_basin2)
            print("newBasinlabel",newBasinlabel)

        #update arc_bp 1 DONE
        #update basinPair_arcs 2 DONE
        #update saddleLabel 3 DONE
        #update ws_bp 4
        #update basin_neighbourBasins 5 DONE
        #update basin_arcs 6 DONE
        #update WSarcs 7 DONE
        #update w_arcs 8

        for b in nonCommonNbrBasins_basin1:
            thisBasinPair=(b,basin1)
            if basinPair_arcs.get((b,basin1)) is None: thisBasinPair=(basin1,b)

            arc_bp[basinPair_arcs[thisBasinPair]]=(b,newBasinlabel) #add 1
            basinPair_arcs[(b,newBasinlabel)]=basinPair_arcs[thisBasinPair] #add 2
            basin_neighbourBasins[b].append(newBasinlabel) #add 5
            basin_neighbourBasins[newBasinlabel].append(b) #add 5
            basin_arcs[newBasinlabel].append(basinPair_arcs[thisBasinPair]) #add 6

            if saddleType=='basin_meandiff':
    #             saddleValue[basinPair_arcs[thisBasinPair]]=euclDist(basin_PointAvg[newBasinlabel],basin_PointAvg[b])
                saddleValue[basinPair_arcs[thisBasinPair]]=max(euclDist(basin_PointAvg[newBasinlabel],basin_PointAvg[b]),saddleValue[basinPair_arcs[thisBasinPair]])

            basin_neighbourBasins[b].remove(basin1) #delete 5
            basinPair_arcs.pop(thisBasinPair) #delete 2



        for b in nonCommonNbrBasins_basin2:
            thisBasinPair=(b,basin2)
            if basinPair_arcs.get((b,basin2)) is None: thisBasinPair=(basin2,b)

            arc_bp[basinPair_arcs[thisBasinPair]]=(b,newBasinlabel) #add 1
            basinPair_arcs[(b,newBasinlabel)]=basinPair_arcs[thisBasinPair] #add 2
            basin_neighbourBasins[b].append(newBasinlabel) #add 5
            basin_neighbourBasins[newBasinlabel].append(b) #add 5
            basin_arcs[newBasinlabel].append(basinPair_arcs[thisBasinPair]) #add 6

            if saddleType=='basin_meandiff':
    #             saddleValue[basinPair_arcs[thisBasinPair]]=euclDist(basin_PointAvg[newBasinlabel],basin_PointAvg[b])
                saddleValue[basinPair_arcs[thisBasinPair]]=max(euclDist(basin_PointAvg[newBasinlabel],basin_PointAvg[b]),saddleValue[basinPair_arcs[thisBasinPair]])

            basinPair_arcs.pop(thisBasinPair) #delete 2
            basin_neighbourBasins[b].remove(basin2) #delete 5


        for b in commonNbrBasins:        
            oldArcs,oldBps=[],[]
            for bb in [basin1,basin2]:
                thisBasinPair=(b,bb)
                if basinPair_arcs.get((b,bb)) is None: thisBasinPair=(bb,b)
                oldBps.append(thisBasinPair)
                oldArcs.append(basinPair_arcs[thisBasinPair])
                for w in WSarcs[basinPair_arcs[thisBasinPair]]:#delete 8
                    if doPrint: print("w",w,"thisBasinPair",thisBasinPair,"basinPair_arcs[thisBasinPair]",basinPair_arcs[thisBasinPair])
                    w_arcs[w].remove(basinPair_arcs[thisBasinPair]) 


            newArcLabel=generateRandomLabel()
            if saddleType=='min':
                newArcSaddleVal=saddleValue[oldArcs[0]] if saddleValue[oldArcs[0]]>saddleValue[oldArcs[1]] else saddleValue[oldArcs[1]]
            elif saddleType in ['saddle_mean','arc_mean']:
                newArcSaddleVal=(saddleValue[oldArcs[0]]*len(WSarcs[oldArcs[0]])+saddleValue[oldArcs[1]]*len(WSarcs[oldArcs[1]]))/(len(WSarcs[oldArcs[0]])+len(WSarcs[oldArcs[1]]))
            elif saddleType=='basin_meandiff':
    #             newArcSaddleVal=euclDist(basin_PointAvg[newBasinlabel],basin_PointAvg[b])
                newArcSaddleVal=max(euclDist(basin_PointAvg[newBasinlabel],basin_PointAvg[b]),saddleValue[oldArcs[0]],saddleValue[oldArcs[1]])

            if doPrint: 
                print('oldArcs',oldArcs,'oldBps',oldBps,"newArcLabel",newArcLabel,'newArcSaddleVal',newArcSaddleVal)

            if doFactorWS: label[newArcLabel]=label[oldArcs[0]] if saddleHeight[label[oldArcs[0]]]<saddleHeight[label[oldArcs[1]]] else label[oldArcs[1]]

            newArcSaddle=saddleLabel[oldArcs[0]] if saddleValue[oldArcs[0]]>saddleValue[oldArcs[1]] else saddleLabel[oldArcs[1]]
            saddleValue[newArcLabel]=newArcSaddleVal
            arc_bp[newArcLabel]=(b,newBasinlabel) #add 1
            basinPair_arcs[(b,newBasinlabel)]=newArcLabel #add 2
            saddleLabel[newArcLabel]=newArcSaddle #add 3
            basin_neighbourBasins[b].append(newBasinlabel) #add 5
            basin_neighbourBasins[newBasinlabel].append(b) #add 5
            basin_arcs[newBasinlabel].append(newArcLabel) #add 6
            basin_arcs[b].append(newArcLabel) #add 6
            WSarcs[newArcLabel]=list(set(WSarcs[oldArcs[0]]).union(WSarcs[oldArcs[1]])) #add 7
            for w in WSarcs[newArcLabel]:
                w_arcs[w].append(newArcLabel) #add 8

            if arcStatus.get(oldArcs[0]) is not None and arcStatus.get(oldArcs[1]) is not None:
    #                 toRemove.append(newArcLabel)
                if sortingStep: bisect.insort(toRemove, (newArcSaddleVal,newArcLabel))
                else: toRemove.append(newArcLabel)
                arcStatus[newArcLabel]='ok'

            for a in oldArcs:
                arc_bp.pop(a) #delete 1
                saddleValue.pop(a)
                basin_arcs[b].remove(a)
                WSarcs.pop(a) #delete 7
                arcStatus[a]='deleted'
            for bp in oldBps:
                basinPair_arcs.pop(bp) #delete 2
            saddleLabel.pop(oldArcs[0]) #delete 3
            saddleLabel.pop(oldArcs[1]) #delete 3
            basin_neighbourBasins[b].remove(basin1) #delete 5
            basin_neighbourBasins[b].remove(basin2) #delete 5


        thisBasinPair=(basin1,basin2)
        if basinPair_arcs.get((basin1,basin2)) is None: thisBasinPair=(basin2,basin1)

        for w in WSarcs[A]:
            if len(w_arcs[w])==1: #delete 8
                w_arcs.pop(w)
            else:
                w_arcs[w].remove(A)

        for w in WSarcs[A]:
    #         nbrWpoints_otherBp=[(p,[arc_bp[a] for a in list(set(w_arcs[p]).difference(set(basin_arcs[newBasinlabel])))]) for p in getNlist(lookupTable[w],level_zero=True) if w_arcs.get(p) is not None]
            nbrWpoints_otherBp=[(p,[arc_bp[a] for a in list(set(w_arcs[p]).difference(set(basin_arcs[newBasinlabel])))]) for p in getNeighbours4(w,image.shape) if w_arcs.get(p) is not None]
    #         print('w',w,'nbrWpoints_otherBp',nbrWpoints_otherBp)
            for p_a in nbrWpoints_otherBp:
                thisP=p_a[0]
                thisOtherBp=p_a[1]
                for bp in thisOtherBp:
                    for aBasin in [bp[0],bp[1]]:
    #                     print('arc extension')
                        if basinPair_arcs.get((aBasin,newBasinlabel)) is not None:
    #                         print("here1",A)
                            thisArc=basinPair_arcs[(aBasin,newBasinlabel)]
                            WSarcs[thisArc].append(thisP)
                            w_arcs[thisP].append(thisArc)
                        else:
    #                         print("here2",A)
                            newArcLabel=generateRandomLabel()
    #                         saddleValue[newArcLabel]=Img[thisP] #???
                            saddleValue[newArcLabel]=saddleValue[A] #???
                            if doFactorWS: label[newArcLabel]=label[A]

                            arc_bp[newArcLabel]=(aBasin,newBasinlabel) #add 1
                            basinPair_arcs[(aBasin,newBasinlabel)]=newArcLabel #add 2
                            saddleLabel[newArcLabel]=thisP #add 3
                            basin_neighbourBasins[aBasin].append(newBasinlabel) #add 5
                            basin_neighbourBasins[newBasinlabel].append(aBasin) #add 5
                            basin_arcs[newBasinlabel].append(newArcLabel) #add 6
                            basin_arcs[aBasin].append(newArcLabel) #add 6
                            WSarcs[newArcLabel]=[thisP]
                            w_arcs[thisP].append(newArcLabel) #add 8
    #                         print('w',w,'thisP',thisP,'aBasin',aBasin)

        arc_bp.pop(A) #delete 1
        basinPair_arcs.pop(thisBasinPair) #delete 2
        saddleLabel.pop(A) #delete 3
        basin_neighbourBasins.pop(basin1) #delete 5
        basin_neighbourBasins.pop(basin2) #delete 5
        basin_arcs.pop(basin1) #delete 6
        basin_arcs.pop(basin2) #delete 6
        WSarcs.pop(A) #delete 7
        arcStatus[A]='deleted'
        
    saddleType='basin_meandiff' #'arc_mean','min','saddle_mean', 'basin_meandiff'


    level=0

    P=[(i,j) for i in range(image.shape[0]) for j in range (image.shape[1])]
    imageF_=getF_image(image)
    initCenters=getCenters_imageF_(imageF_)

    level=0

    # label,currentDist=getWS_multiBand2(image,initCenters)
    label,wsMarked,arcPoints=getWS_multiBand(image,initCenters)


    WSarcs,arc_bp,saddleLabel,saddleValue,basinPair_arcs,w_arcs,basin_arcs,basin_neighbourBasins=getWSarcs3(P,arcPoints,wsMarked,label,saddleType=saddleType)


    basin_Points={}
    basin_PointCount={}
    basin_PointSum={}
    basin_PointAvg={}
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if wsMarked[(i,j)]: continue
            if basin_Points.get(label[(i,j)]) is None: 
                basin_Points[label[(i,j)]]=[]
                basin_PointCount[label[(i,j)]]=0
                basin_PointSum[label[(i,j)]]=0 if image.ndim==2 else np.zeros(image.shape[2])
            basin_Points[label[(i,j)]].append((i,j))
            basin_PointCount[label[(i,j)]]+=1
            basin_PointSum[label[(i,j)]]+=image[i,j]
    basinCount=len(basin_Points)
    for b in range(len(basin_Points)):
        basin_PointAvg[b]=basin_PointSum[b]/basin_PointCount[b]
    if saddleType=='basin_meandiff':
        for arc in arc_bp:
            saddleValue[arc]=euclDist(basin_PointAvg[arc_bp[arc][0]],basin_PointAvg[arc_bp[arc][1]])
            
    while True:
        arcImage=np.zeros((image.shape[0],image.shape[1]),dtype=np.int)
        for a in WSarcs:
            for p in WSarcs[a]:
                arcImage[p]=1
        level+=1

        upgradeFactorWS=True
        sortingStep=True
        doFactorWS=True
        if beta is None:
            factor=1
        elif type(beta)==float or type(beta)==int:
            factor=beta
        elif type(beta)==list or type(beta)==np.ndarray:
            if len(list(beta))==0:
                factor=1
            else:
                factor=beta[level-1] if len(beta)>=level else beta[-1]
        if doPrint: print("level",level,'beta',factor)

        nbrBuckets = getnbrBuckets(arc_bp)
        Pix=list(WSarcs.keys())
        if doFactorWS: label,wsMarked,factorWS,saddleValue,saddleHeight=getWSpoints(Pix,saddleValue,{p:p for p in WSarcs},factor=factor)
        else: label,wsMarked=getWSpoints(Pix,saddleValue,{p:p for p in saddleValue})
            
        if len([1 for w in wsMarked if wsMarked[w]])==0:
            break
                
        if len([1 for w in wsMarked if wsMarked[w]])>0:

            if sortingStep: 
                if doFactorWS: toRemove= [(saddleValue[bp],bp) for bp in wsMarked if not wsMarked[bp]  and not factorWS[bp]]
                else: toRemove= [(saddleValue[bp],bp) for bp in wsMarked if not wsMarked[bp]]
                toRemove.sort()
                arcStatus={bp[1]:'ok' for bp in toRemove}
            else:
                if doFactorWS: toRemove= [bp for bp in wsMarked if not wsMarked[bp]  and not factorWS[bp]]
                else: toRemove= [bp for bp in wsMarked if not wsMarked[bp]]
                arcStatus={bp:'ok' for bp in toRemove}

            while len(toRemove)>0:
                A=toRemove.pop(0)
                if sortingStep: 
                    if arcStatus[A[1]]!='ok': continue
        #             if doFactorWS: 
        #                 if saddleValue[A[1]]>=saddleHeight[label[A[1]]]: continue
                else:
                    if arcStatus[A]!='ok': continue
        #             if doFactorWS: 
        #                 if saddleValue[A]>=saddleHeight[label[A]]: continue

            #     print(A)
                removeArc(A) #,doPrint=True
    return arcImage