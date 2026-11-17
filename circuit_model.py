import numpy as np

def problem(circuit):
    dem=circuit.detector_error_model()
    cols={}; ps=[]
    for ins in dem.flattened():
        if ins.type!='error': continue
        ds=set(); ls=set()
        for t in ins.targets_copy():
            if t.is_relative_detector_id(): ds.symmetric_difference_update([t.val])
            elif t.is_logical_observable_id(): ls.symmetric_difference_update([t.val])
        key=(tuple(sorted(ds)),tuple(sorted(ls))); p=ins.args_copy()[0]
        if key in cols:
            j=cols[key]; ps[j]=ps[j]*(1-p)+p*(1-ps[j])
        else: cols[key]=len(ps); ps.append(p)
    H=np.zeros((dem.num_detectors,len(ps)),np.uint8)
    L=np.zeros((dem.num_observables,len(ps)),np.uint8)
    for (ds,ls),j in cols.items():
        H[list(ds),j]=1; L[list(ls),j]=1
    rr,cc=np.nonzero(H); order=np.argsort(rr,kind='stable'); rr=rr[order]; cc=cc[order]
    ptr=np.r_[0,np.cumsum(np.bincount(rr,minlength=len(H)))]
    return H,L,np.log((1-np.array(ps))/np.array(ps)),cc,ptr
