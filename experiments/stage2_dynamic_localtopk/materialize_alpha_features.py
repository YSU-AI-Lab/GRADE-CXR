#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def l2n(x,eps=1e-8): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),eps)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--components-root',required=True); ap.add_argument('--output-root',required=True); ap.add_argument('--alpha',type=float,required=True); ap.add_argument('--gate-scale',type=float,default=1.0); ap.add_argument('--splits',default='train,val,test')
 args=ap.parse_args(); comp=Path(args.components_root); out=Path(args.output_root)
 for split in args.splits.split(','):
  z=np.load(comp/split/'z_base.npy'); u=np.load(comp/split/'u.npy'); g=np.load(comp/split/'gate.npy')
  feat=l2n(z + float(args.alpha)*float(args.gate_scale)*g*u).astype('float32')
  od=out/split; od.mkdir(parents=True,exist_ok=True); np.save(od/'features_stage2_v2_refined.npy',feat)
 (out/'materialized.json').write_text(json.dumps({'components_root':str(comp),'alpha':args.alpha,'gate_scale':args.gate_scale},indent=2),encoding='utf-8')
if __name__=='__main__': main()
