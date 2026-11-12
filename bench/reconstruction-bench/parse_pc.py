#!/usr/bin/env python3
"""Keep per-launch PC counts and static disassembly; do not guess source roles."""
import csv,pathlib,re,sys
root=pathlib.Path(sys.argv[1])
rows=[]
for path in sorted(root.glob('*_pc.log')):
    kernel=None; current=[]; launch_count=0
    def finish():
        if kernel is not None and sum(x['count'] for x in current)!=kernel[1]:
            raise ValueError(f'{path}: histogram does not sum to launch {kernel[0]} total')
        rows.extend(current)
    for line in path.read_text().splitlines():
        m=re.match(r'kernel (\d+) - (.*?) - #thread-blocks (\d+),\s+kernel instructions (\d+),',line)
        if m:
            finish(); kernel=(int(m[1]),int(m[4]),m[2]); current=[]; launch_count+=1
        elif kernel and (m:=re.match(r'  (.*?)\|pc=(\d+)\|([A-Z0-9_.]+) = (\d+)$',line)):
            current.append(dict(log=path.name,launch=kernel[0],launched_function=kernel[2],
                function=m[1],pc=int(m[2]),opcode=m[3],count=int(m[4])))
    finish()
    if not launch_count: raise ValueError(f'{path}: no captured launches')
if not rows: raise SystemExit('No per-PC counts found')
with (root/'instruction_addresses.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(f'Wrote {len(rows)} per-launch instruction-address counts')
