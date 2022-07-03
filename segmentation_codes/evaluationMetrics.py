import numpy as np
import cv2
from sklearn.metrics.cluster import adjusted_mutual_info_score

def getSliceLabel(image):
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
    shp=image.shape
    bSegImage=np.zeros(shp,dtype=np.int32)
    toSegMask=np.full(shp,True)
    segNo=0
    for i in range(shp[0]):
        for j in range(shp[1]):
            if not toSegMask[i,j]: continue
            segNo+=1
            thisLabel=image[i,j]
            Q=[(i,j)]
            toSegMask[i,j]=False
            bSegImage[i,j]=segNo
            while len(Q)>0:
                q=Q.pop(0)
                for n in getNeighbours8(q,shp):
                    if toSegMask[n] and image[n]==thisLabel:
                        Q.append(n)
                        toSegMask[n]=False
                        bSegImage[n]=segNo
    return bSegImage

def nC2(n):
    return (n*(n-1))/2

def ariCalc(res,gt):
    res=res.flatten()
    gt=gt.flatten()
    Labels_res=list(set(res))
    posres={x:i for i,x in enumerate(Labels_res)}
    Labels_gt=list(set(gt))
    posgt={x:i for i,x in enumerate(Labels_gt)}
    mat=np.zeros((len(Labels_res),len(Labels_gt)))
    for l in range(len(res)):
        mat[posres[res[l]],posgt[gt[l]]]+=1
    FT=sum([nC2(mat[i,j]) for i in range(mat.shape[0]) for j in range(mat.shape[1])])
    ST=sum([nC2(sum(mat[i,:]))*nC2(sum(mat[:,j]) ) for i in range(mat.shape[0]) for j in range(mat.shape[1])])/nC2(len(res))
    TT=(sum([nC2(sum(mat[i,:])) for i in range(mat.shape[0])])+sum([nC2(sum(mat[:,j])) for j in range(mat.shape[1])]))/2
    return (FT-ST)/(TT-ST)
    
def getARI(res,gt,binaryLabel=False):
    if binaryLabel:
        gt=np.where(gt==1,1,0)
        return ariCalc(res,gt)
    else:
        ari1=ariCalc(res,gt)
        gt_sl=getSliceLabel(gt)
        if np.max(gt_sl)-np.min(gt_sl)==np.max(gt)-np.min(gt): return ari1
        else: return max(ari1,ariCalc(res,gt_sl))

def getSeg(bounImage,frame=None,segBoundary=False):
    if frame is not None: frame=cv2.morphologyEx(frame.astype(np.uint8), cv2.MORPH_ERODE, np.ones((3,3))).astype(np.bool)
    def getWSseg(image):
        shp=image.shape
        wsSegImage=np.copy(image)
        toSegMask=np.where(image==0,True,False)
        if frame is not None: toSegMask=toSegMask&frame
        for i in range(shp[0]):
            for j in range(shp[1]):
                if toSegMask[i,j]:
                    if j+1<shp[1] and not toSegMask[i,j+1]:
                        wsSegImage[i,j]=image[i,j+1]
                    elif i+1<shp[0] and not toSegMask[i+1,j]:
                        wsSegImage[i,j]=image[i+1,j]
                    elif j-1>=0 and not toSegMask[i,j-1]:
                        wsSegImage[i,j]=image[i,j-1]
                    elif i-1>=0 and not toSegMask[i-1,j]:
                        wsSegImage[i,j]=image[i-1,j]
        return wsSegImage
    def getBasinSeg(image):
        shp=image.shape
        bSegImage=np.zeros(shp,dtype=np.int32)
        toSegMask=np.where(image==1,False,True)
        if frame is not None: toSegMask=toSegMask&frame
        segNo=0
        for i in range(shp[0]):
            for j in range(shp[1]):
                if not toSegMask[i,j]: continue
                segNo+=1
                Q=[(i,j)]
                toSegMask[i,j]=False
                bSegImage[i,j]=segNo
                while len(Q)>0:
                    q=Q.pop(0)
                    for n in get4Neighbours(q,shp):
                        if toSegMask[n]:
                            Q.append(n)
                            toSegMask[n]=False
                            bSegImage[n]=segNo
        return bSegImage
    def get4Neighbours(p,shp):
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
    
    if segBoundary:
        return getBasinSeg(bounImage)
    else:
        return getWSseg(getBasinSeg(bounImage))


def getAMI(res,gt,binaryLabel=False):
    if binaryLabel:
        gt=np.where(gt==1,1,0)   
        return adjusted_mutual_info_score(res.flatten(),gt.flatten())
    else:
        ami1=adjusted_mutual_info_score(res.flatten(),gt.flatten())
        gt_sl=getSliceLabel(gt)
        if np.max(gt_sl)-np.min(gt_sl)==np.max(gt)-np.min(gt): return ami1
        else: return max(ami1,adjusted_mutual_info_score(res.flatten(),gt_sl.flatten()))

def getEvalScores(gtObj,method_seg, evalPriority='AMI',obj=2,binaryLabel=False,doPrint=False,doWrite=False):
    maxQualMeasure=-np.Inf
    bestMatchedGtIndex=np.Inf
    for g in range(len(gtObj)):
        if evalPriority=='ARI':
            thisQualMeasure=getARI(method_seg,gtObj[g],binaryLabel=binaryLabel)
        elif evalPriority=='CA':
            thisQualMeasure=getClusteringAccuracy(method_seg,gtObj[g],binaryLabel=binaryLabel,obj=obj)
        else:
            thisQualMeasure=getAMI(method_seg,gtObj[g],binaryLabel=binaryLabel)
            
        if maxQualMeasure<thisQualMeasure:
            maxQualMeasure=thisQualMeasure
            bestMatchedGtIndex=g
            
    if evalPriority=='ARI':
        ari=maxQualMeasure
        ca=getClusteringAccuracy(method_seg,gtObj[bestMatchedGtIndex],binaryLabel=binaryLabel,obj=obj)
        ami=getAMI(method_seg,gtObj[bestMatchedGtIndex],binaryLabel=binaryLabel)
    elif evalPriority=='CA':
        ca=maxQualMeasure
        ari=getARI(method_seg,gtObj[g],binaryLabel=binaryLabel)
        ami=getAMI(method_seg,gtObj[bestMatchedGtIndex],binaryLabel=binaryLabel)
    else:
        ami=maxQualMeasure
        ca=getClusteringAccuracy(method_seg,gtObj[bestMatchedGtIndex],binaryLabel=binaryLabel,obj=obj)
        ari=getARI(method_seg,gtObj[bestMatchedGtIndex],binaryLabel=binaryLabel)
    
    return bestMatchedGtIndex, ca, ami, ari

def getSliceLabel(image):
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
    shp=image.shape
    bSegImage=np.zeros(shp,dtype=np.int32)
    toSegMask=np.full(shp,True)
    segNo=0
    for i in range(shp[0]):
        for j in range(shp[1]):
            if not toSegMask[i,j]: continue
            segNo+=1
            thisLabel=image[i,j]
            Q=[(i,j)]
            toSegMask[i,j]=False
            bSegImage[i,j]=segNo
            while len(Q)>0:
                q=Q.pop(0)
                for n in getNeighbours8(q,shp):
                    if toSegMask[n] and image[n]==thisLabel:
                        Q.append(n)
                        toSegMask[n]=False
                        bSegImage[n]=segNo
    return bSegImage
        
def ClusteringAccuracy(res,gt,obj=2):
    resSlices=[np.where(res==i,True,False) for i in range(np.min(res),np.max(res)+1)]
    gtSlices=[np.where(gt==i,True,False) for i in range(np.min(gt),np.max(gt)+1)]
    gtSliceSizes=[np.sum(gtSlice) for gtSlice in gtSlices]
    gtSliceSizesArgSort=np.argsort(gtSliceSizes)
    gtSlices_target=[gtSlices[gtSliceSizesArgSort[i]] for i in range(len(gtSlices)-1,len(gtSlices)-1-obj,-1)]
    
    omitResSlicesIdx=[]
    for G in gtSlices_target:
        maxIntersection,maxIntersectionIdx=-np.Inf,-np.Inf
        for r in range(len(resSlices)):
            if r not in omitResSlicesIdx:
                thisIntersection=np.sum(resSlices[r]&G)/np.sum(resSlices[r]|G)
                if thisIntersection>maxIntersection:
                    maxIntersection=thisIntersection
                    maxIntersectionIdx=r
        omitResSlicesIdx.append(maxIntersectionIdx)

    return np.sum([np.sum(gtSlices_target[i]&resSlices[omitResSlicesIdx[i]]) if i<len(resSlices) else 0 for i in range(len(gtSlices_target))])/np.sum([np.sum(gtSlices_target[i]|resSlices[omitResSlicesIdx[i]]) if i<len(resSlices) else np.sum(gtSlices_target[i]) for i in range(len(gtSlices_target))])

def getClusteringAccuracy(res,gt,binaryLabel=False,obj=2):
    if binaryLabel:
        gt=np.where(gt==1,1,0)   
        return ClusteringAccuracy(res,gt)
    else:
        ca1=ClusteringAccuracy(res,gt,obj=obj)
        gt_sl=getSliceLabel(gt)
        if np.max(gt_sl)-np.min(gt_sl)==np.max(gt)-np.min(gt): return ca1
        else: return max(ca1,ClusteringAccuracy(res,gt_sl,obj=obj))