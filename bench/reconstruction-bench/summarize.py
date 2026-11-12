#!/usr/bin/env python3
import csv,pathlib,statistics,sys
root=pathlib.Path(sys.argv[1]); groups={}; allrows=[]
for path in sorted(root.glob('*_pass*.csv')):
    for r in csv.DictReader(path.open()):
        r['file']=path.name;allrows.append(r)
        key=tuple(r[k] for k in ['format','variant','mode','m','k','nmat'])
        groups.setdefault(key,[]).append(r)
if not allrows: raise SystemExit('No timing samples')
with (root/'all_samples.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
with (root/'summary.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['format','variant','mode','m','k','nmat','samples','mean_us','stdev_us','mean_ns_per_weight','registers','local_bytes'])
    for key,rr in sorted(groups.items()):
        times=[float(r['us_per_launch']) for r in rr]
        w.writerow([*key,len(rr),statistics.mean(times),statistics.stdev(times) if len(rr)>1 else 0,
          statistics.mean(float(r['ns_per_weight']) for r in rr),rr[0]['registers'],rr[0]['local_bytes']])
print(f'Wrote {len(groups)} summaries to {root}/summary.csv')
