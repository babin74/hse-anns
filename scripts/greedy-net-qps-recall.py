import os

shard_cnt = 1

S, R, Q = [], [], []

while shard_cnt <= 10 ** 4:
    print(shard_cnt)
    out = os.popen(f'./tann {shard_cnt}').read()
    qps, recall = map(float, out.split())
    S.append(shard_cnt)
    R.append(recall)
    Q.append(qps)
    shard_cnt = max(shard_cnt+1, int(shard_cnt*1.5))

print('S =', S)
print('R =', R)
print('Q =', Q)