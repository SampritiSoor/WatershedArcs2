from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np

def RAGmerge(img):
    def _weight_mean_color(graph, src, dst, n):
        diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
        diff = np.linalg.norm(diff)
        return {'weight': diff}

    def merge_mean_color(graph, src, dst):
        graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
        graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
        graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                          graph.nodes[dst]['pixel count'])

    labels = segmentation.slic(img, compactness=500, n_segments=200)
    g = graph.rag_mean_color(img, labels)

    labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)
    return labels2

def Ncut(img):
    compactness=500
    while True:
        labels1 = segmentation.slic(img, compactness=compactness, n_segments=200)
        g = graph.rag_mean_color(img, labels1, mode='similarity')
        labels2 = graph.cut_normalized(labels1, g)
        if np.max(labels2)==0:
            compactness+=500
            if compactness>5000:
                break
        else:
            break
    return labels2