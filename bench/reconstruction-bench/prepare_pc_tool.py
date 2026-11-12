#!/usr/bin/env python3
"""Create a PC-keyed copy of NVBit's opcode_hist without changing the original."""
import pathlib, shutil, sys
root=pathlib.Path(sys.argv[1]).resolve()
src=root/'tools/opcode_hist'
dst=root/'tools/pc_hist'
if not (src/'opcode_hist.cu').is_file():
    raise SystemExit(f'Not an NVBit release directory: {root}')
text=(src/'opcode_hist.cu').read_text()
old='std::string opcode = i->getOpcode();'
if text.count(old)!=1:
    raise SystemExit('NVBit source differs from expected 1.8.1; refusing to patch')
new='''std::string opcode = kernel_name + "|pc=" + std::to_string(i->getOffset()) + "|" + i->getOpcode();'''
shutil.copytree(src,dst,dirs_exist_ok=True,ignore=shutil.ignore_patterns('build'))
(dst/'opcode_hist.cu').write_text(text.replace(old,new))
print(f'Prepared {dst}; labels contain function, byte offset, and opcode.')
