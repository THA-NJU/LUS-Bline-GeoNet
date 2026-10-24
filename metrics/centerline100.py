import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

def centerline_length(mask):
    sk=skeletonize(mask)
    ys,xs=np.nonzero(sk);n=len(ys)
    if n<2:return 0.
    ids=np.full(sk.shape,-1,np.int32);ids[ys,xs]=np.arange(n)
    src=[];dst=[];weights=[]
    for dy,dx in [(0,1),(1,-1),(1,0),(1,1)]:
        yy=ys+dy;xx=xs+dx
        valid=(yy>=0)&(yy<sk.shape[0])&(xx>=0)&(xx<sk.shape[1])
        a=np.flatnonzero(valid);bb=ids[yy[valid],xx[valid]];good=bb>=0;a=a[good];bb=bb[good]
        src.extend(a);dst.extend(bb);weights.extend([float(np.hypot(dy,dx))]*len(a))
    graph=csr_matrix((weights+weights,(src+dst,dst+src)),shape=(n,n))
    diameter=0.
    for start in range(0,n,128):
        distances=dijkstra(graph,directed=False,indices=np.arange(start,min(n,start+128)))
        finite=distances[np.isfinite(distances)]
        if finite.size:diameter=max(diameter,float(finite.max()))
    return diameter

def filter_short(mask):
    labels,n=ndimage.label(mask,np.ones((3,3),np.uint8));objects=ndimage.find_objects(labels)
    output=mask.copy();removed=0;removed_pixels=0
    for k,sl in enumerate(objects,1):
        component=labels[sl]==k;length=centerline_length(component)
        if length<100.:
            output[sl][component]=False;removed+=1;removed_pixels+=int(component.sum())
    return output,dict(components_before=int(n),components_removed=removed,pixels_removed=removed_pixels)

def self_test():
    for pixels,keep in [(100,False),(101,True),(102,True)]:
        m=np.zeros((120,10),bool);m[5:5+pixels,5]=True
        assert abs(centerline_length(m)-(pixels-1))<1e-9
        out,_=filter_short(m);assert out.any()==keep
    m=np.eye(80,dtype=bool);assert abs(centerline_length(m)-79*np.sqrt(2))<1e-8
    out,_=filter_short(m);assert np.array_equal(out,m)
    print('CENTERLINE_SELF_TEST_PASSED',flush=True)
