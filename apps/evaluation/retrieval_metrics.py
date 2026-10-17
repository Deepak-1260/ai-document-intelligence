from typing import List
def precision_at_k(retrieval_labels:List[int], k:int)->float:
    n=len(retrieval_labels)
    
    if  k<=0:
        raise ValueError("Provide valid arguments for precision at K")
    if n<k:
            k=n
    c=0
    for i in retrieval_labels[:k]:
        c+=i
    
    return float(c/k) if k>0 else 0.0

def recall_at_k(retrieval_labels:List[int], k:int, ttl_relevant: int)->float:
    n=len(retrieval_labels)
    if k<=0:
        raise ValueError("Provide valid arguments for  K")
    if ttl_relevant<0:
            raise ValueError("Provide valid arguments for  ttl_relevant")
    if n<k:
        k=n
    c=0
    for i in retrieval_labels[:k]:
        c+=i
    return float(c/ttl_relevant) if ttl_relevant!=0 else 0.0

def reciprocal_rank(retrieval_labels:List[int])->float:
     i=0
     for idx,x in enumerate(retrieval_labels):
          if x==1:
               i=idx+1
               break
     return float(1/i) if i!=0 else 0.0

def evaluate_retrieval(retrieval_labels:List[int], k:int, ttl_relevant:int)->dict:
     ans={}
     ans["precision_at_k"]=precision_at_k(retrieval_labels,k)
     ans["recall_at_k"]=recall_at_k(retrieval_labels,k,ttl_relevant)
     ans["reciprocal_rank"]=reciprocal_rank(retrieval_labels)
     return ans

    
