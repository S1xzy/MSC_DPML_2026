import torch, json, numpy as np, sys
out = sys.argv[1]
P = torch.load('data/pheme/processed/seed2_time20_T200.pt', weights_only=False, map_location='cpu')
S = torch.load('data/pheme/generated_fix/seed2/stage1.pt', weights_only=False, map_location='cpu')
O = torch.load('data/pheme/generated_final/seed2/stage1.pt', weights_only=False, map_location='cpu')
nu = P['num_users']; labels = P['labels'].numpy(); train = set(P['split']['train'])
fake = np.zeros(nu); tot = np.zeros(nu)
for i in train:
    for (p, u, t) in P['eng'][i]:
        tot[u] += 1; fake[u] += (labels[i] == 0)
known = S['known'].numpy()
def zs(E, k):
    E = E.numpy(); m = E[k].mean(0); s = E[k].std(0) + 1e-6
    return (E - m) / s
np.save(f'{out}/emb_fix.npy', zs(S['user_emb'], known).astype(np.float32))
np.save(f'{out}/emb_old.npy', O['user_emb'].numpy().astype(np.float32))
np.save(f'{out}/known.npy', known); np.save(f'{out}/fake.npy', fake); np.save(f'{out}/tot.npy', tot)
print('old unique rows', torch.unique(O['user_emb'][torch.from_numpy(known)], dim=0).shape[0])
print('fix unique rows', torch.unique(S['user_emb'][torch.from_numpy(known)], dim=0).shape[0])
idx = [i for i, t in enumerate(P['texts']) if 'Love is Stronger than Hate' in t]
print('idx', idx, [P['texts'][i][:90] for i in idx], [int(labels[i]) for i in idx], [('test' if i in P['split']['test'] else 'x') for i in idx])
i = idx[0]
json.dump({'i': i, 'text': P['texts'][i], 'label': int(labels[i]), 'eng': [list(map(int, e)) for e in P['eng'][i]],
           'early': [list(map(int, e)) for e in P['early'][i]],
           'path': {'early': [list(map(int, e)) for e in S['paths'][i]['early']],
                    'steps': [[list(map(int, e)) for e in st] for st in S['paths'][i]['steps']]},
           'known': [int(u) for u in np.where(known)[0]][:0]}, open(f'{out}/path.json', 'w'))
print('steps', len(S['paths'][i]['steps']), sum(len(s) for s in S['paths'][i]['steps']), 'eng', len(P['eng'][i]))
print(S['paths'][i].keys())
