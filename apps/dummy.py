from evaluation.retrieval_metrics import precision_at_k,recall_at_k,reciprocal_rank,evaluate_retrieval

l_labels=[
[1, 0, 1],	
[0, 1, 1]	,
[0, 0, 0]	,
[]	
]
l_ttl=[4,3,2,5,3]
l_k=[3,2,2,1,3]

for labels,k,t in zip(l_labels,l_k,l_ttl):
    res=evaluate_retrieval(labels,k,t)
    print(res)
    # print(res['precision_at_k'], res['recall_at_k'], res['reciprocal_rank'])

# for labels in l_labels:
#     print(reciprocal_rank(labels))