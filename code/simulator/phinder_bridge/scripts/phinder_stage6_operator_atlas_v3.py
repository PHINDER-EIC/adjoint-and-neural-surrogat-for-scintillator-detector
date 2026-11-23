#!/usr/bin/env python3
"""
PHINDER Stage-6 v3 high-fidelity atlas generator for arbitrary rounded guide skeletons.

A geometry is supplied through JSON and consists of:
  cell_cm: [dx,dy,dz]
  width_cm
  radius_cm          (one global fillet radius for all turns, initially)
  start_xy_cm
  turns_xy_cm        (ordered internal turning points; <= 10 recommended)
  end_xy_cm

Guide 0 is constructed on the +z face. Guide 1 is its exact x-mirror on the -z
face.  Each polyline corner is replaced by a circular fillet of radius R and
faceted into tangent FrustumVolume pieces.  The script stores full per-photon
truth with emission time fixed to zero.

The source design uses common normalized source positions and common initial
photon directions across all geometries, which substantially reduces sampling
noise in geometry comparisons.  For derivative-pair campaigns, use identical
seeds and, ideally, run the detailed engine with deterministic per-photon RNG
for Fresnel/absorption decisions as well.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import qmc

for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v,"1")

from Photon_Simulator import (
    AIR, CLAD, CORE, EJ204,
    BoxVolume, FrustumVolume, Scene, ReadoutPort,
    ApproximateEngine, PhotonBatch, PhotonStatus,
)
from Photon_Simulator.core.photonsEngines.approximateEngine import PhotonBatchSimulator, ParallelConfig

import phinder_stage5_bend_lab as bendlab

FORMAT="PHINDER optical operator atlas"
FORMAT_VERSION=3
GUIDE_THICKNESS=0.10
CLADDING_THICKNESS=0.001


@dataclass(frozen=True)
class GeometrySpec:
    geometry_id:str
    cell_cm:np.ndarray
    width_cm:float
    radius_cm:float
    start_xy_cm:np.ndarray
    turns_xy_cm:np.ndarray
    end_xy_cm:np.ndarray
    raw:dict

    @property
    def vertices(self):
        return np.vstack([self.start_xy_cm[None,:],self.turns_xy_cm.reshape(-1,2),self.end_xy_cm[None,:]])

    @property
    def r_over_w(self):return self.radius_cm/self.width_cm


def parse_geometry_json(path:Path,max_turns:int)->list[GeometrySpec]:
    obj=json.loads(path.read_text())
    items=obj.get("geometries",[obj]);out=[];ids=set()
    for i,d in enumerate(items):
        gid=str(d.get("id",f"g{i:04d}"))
        if gid in ids:raise ValueError(f"duplicate geometry id {gid}")
        ids.add(gid)
        cell=np.asarray(d["cell_cm"],float)
        if cell.shape!=(3,) or np.any(cell<=0):raise ValueError(f"{gid}: bad cell_cm")
        turns=np.asarray(d.get("turns_xy_cm",[]),float).reshape(-1,2)
        if len(turns)>max_turns:raise ValueError(f"{gid}: {len(turns)} turns > max {max_turns}")
        spec=GeometrySpec(gid,cell,float(d["width_cm"]),float(d["radius_cm"]),np.asarray(d["start_xy_cm"],float),turns,np.asarray(d["end_xy_cm"],float),d)
        if spec.width_cm<=0 or spec.radius_cm<=0:raise ValueError(f"{gid}: width/radius must be >0")
        out.append(spec)
    if not out:raise ValueError("no geometries")
    return out


def _left_normal(u):return np.array([-u[1],u[0]],float)

def rounded_polyline(vertices:np.ndarray,radius:float,facets_per_90:int)->bendlab.GuidePath:
    """Replace each interior polyline vertex with a circular fillet.

    The requested radius is exact.  Invalid geometries (U-turns, overlapping
    fillets, zero segments) are rejected rather than silently modified.
    """
    v=np.asarray(vertices,float)
    if len(v)<2:raise ValueError("guide needs start/end")
    seg=v[1:]-v[:-1];lens=np.linalg.norm(seg,axis=1)
    if np.any(lens<1e-6):raise ValueError("zero/near-zero guide segment")
    dirs=seg/lens[:,None]
    ncorner=len(v)-2
    d=np.zeros(ncorner,float);theta=np.zeros(ncorner,float);sign=np.zeros(ncorner,float)
    for j in range(ncorner):
        u0,u1=dirs[j],dirs[j+1]
        dot=float(np.clip(np.dot(u0,u1),-1,1));ang=math.acos(dot)
        cr=float(u0[0]*u1[1]-u0[1]*u1[0])
        if ang<math.radians(1.0):
            d[j]=0;theta[j]=0;sign[j]=0;continue
        if ang>math.radians(175.0) or abs(cr)<1e-8:raise ValueError(f"unsupported near-U-turn at internal vertex {j}")
        theta[j]=ang;sign[j]=1.0 if cr>0 else -1.0;d[j]=radius*math.tan(0.5*ang)
    # segment i has outgoing fillet from corner i-1 and incoming fillet to corner i
    for i,L in enumerate(lens):
        take0=d[i-1] if i-1>=0 else 0.0
        take1=d[i] if i<ncorner else 0.0
        if take0+take1>=L-1e-5:
            raise ValueError(f"radius={radius:g} too large: fillets overlap on segment {i} (need {take0+take1:g}, have {L:g})")

    pts=[v[0].copy()];tans=[dirs[0].copy()]
    for j in range(ncorner):
        if theta[j]<math.radians(1.0):
            # Collinear-ish control point: keep no geometrical kink.
            continue
        corner=v[j+1];u0,u1=dirs[j],dirs[j+1];dd=d[j];sgn=sign[j]
        entry=corner-u0*dd;exitp=corner+u1*dd
        center=entry+sgn*radius*_left_normal(u0)
        r0=entry-center
        a0=math.atan2(r0[1],r0[0])
        nfac=max(2,int(math.ceil(facets_per_90*theta[j]/(0.5*math.pi))))
        # Append entry (straight segment ends here), then arc points excluding duplicate entry.
        if np.linalg.norm(np.asarray(pts[-1])-entry)>1e-10:
            pts.append(entry);tans.append(u0.copy())
        for k in range(1,nfac+1):
            a=a0+sgn*theta[j]*k/nfac
            rad=np.array([math.cos(a),math.sin(a)])*radius
            q=center+rad
            tan=sgn*_left_normal(rad/radius)
            pts.append(q);tans.append(tan/np.linalg.norm(tan))
        # numerical endpoint should match exitp; force it exactly
        pts[-1]=exitp;tans[-1]=u1.copy()
    if np.linalg.norm(np.asarray(pts[-1])-v[-1])>1e-10:
        pts.append(v[-1].copy());tans.append(dirs[-1].copy())
    return bendlab.GuidePath(points=np.asarray(pts,float),tangents=np.asarray(tans,float),kind="rounded_polyline",angle_deg=float("nan"),radius_cm=radius,facets=facets_per_90)


def mirror_x_path(path:bendlab.GuidePath,cell_x:float):
    p=path.points.copy();p[:,0]=cell_x-p[:,0]
    t=path.tangents.copy();t[:,0]*=-1
    return bendlab.GuidePath(points=p,tangents=t,kind=path.kind+"_mirror",angle_deg=path.angle_deg,radius_cm=path.radius_cm,facets=path.facets)


def cross_section_z(center,tangent,width,thickness,zc):
    a=bendlab.cross_section(center,tangent,width,thickness).copy();a[:,2]+=float(zc);return a


def validate_path(spec:GeometrySpec,path:bendlab.GuidePath):
    X,Y,Z=spec.cell_cm;pad=.5*spec.width_cm
    xmin=float(path.points[:,0].min()-pad);xmax=float(path.points[:,0].max()+pad);ymin=float(path.points[:,1].min()-pad)
    if xmin<-1e-8 or xmax>X+1e-8 or ymin<-1e-8:
        raise ValueError(f"{spec.geometry_id}: guide footprint outside allowed face in x/-y: x=[{xmin:.3g},{xmax:.3g}], ymin={ymin:.3g}")
    # Internal turns and start should live on/near face; terminal may extend beyond +y for readout.
    for q in np.vstack([spec.start_xy_cm[None,:],spec.turns_xy_cm.reshape(-1,2)]):
        if q[0]<0 or q[0]>X or q[1]<0 or q[1]>Y:
            raise ValueError(f"{spec.geometry_id}: start/turn {q.tolist()} outside cell face")


def build_scene(spec:GeometrySpec,facets_per_90:int):
    X,Y,Z=map(float,spec.cell_cm)
    if 2*CLADDING_THICKNESS>=min(spec.width_cm,GUIDE_THICKNESS):raise ValueError("cladding too thick")
    path0=rounded_polyline(spec.vertices,spec.radius_cm,facets_per_90);validate_path(spec,path0);path1=mirror_x_path(path0,X)
    core_w=spec.width_cm-2*CLADDING_THICKNESS;core_h=GUIDE_THICKNESS-2*CLADDING_THICKNESS
    volumes=[BoxVolume("scintillator",EJ204,[0,0,0],[X,Y,Z],priority=0)];ports=[]
    for ch,path,zc in ((0,path0,Z+.5*GUIDE_THICKNESS),(1,path1,-.5*GUIDE_THICKNESS)):
        nseg=len(path.points)-1
        for i in range(nseg):
            p0,p1=path.points[i],path.points[i+1];t0,t1=path.tangents[i],path.tangents[i+1]
            outer0=cross_section_z(p0,t0,spec.width_cm,GUIDE_THICKNESS,zc);outer1=cross_section_z(p1,t1,spec.width_cm,GUIDE_THICKNESS,zc)
            core0=cross_section_z(p0,t0,core_w,core_h,zc);core1=cross_section_z(p1,t1,core_w,core_h,zc)
            volumes.append(FrustumVolume(f"guide{ch}_clad_{i:03d}",CLAD,bottom_vertices=outer0,top_vertices=outer1,priority=10))
            name=f"guide{ch}_terminal" if i==nseg-1 else f"guide{ch}_core_{i:03d}"
            volumes.append(FrustumVolume(name,CORE,bottom_vertices=core0,top_vertices=core1,priority=20))
        tf=path.tangents[-1]/np.linalg.norm(path.tangents[-1])
        ports.append(ReadoutPort(channel=ch,volume_name=f"guide{ch}_terminal",point=(float(path.points[-1,0]),float(path.points[-1,1]),float(zc)),outward_normal=(float(tf[0]),float(tf[1]),0.),label=f"stage6-v3 ch{ch}"))
    return Scene(volumes=volumes,outside_material=AIR,scintillator_material=EJ204,readout_ports=ports),path0,path1


def normalized_sources(n:int,seed:int,margin_xy:float,margin_z:float):
    if not(0<=margin_xy<.5 and 0<=margin_z<.5):raise ValueError("source margins are fractions and must be in [0,0.5)")
    m=int(math.ceil(math.log2(max(1,n))));u=qmc.Sobol(d=3,scramble=True,seed=seed).random_base2(m)[:n]
    lo=np.array([margin_xy,margin_xy,margin_z]);hi=1-lo
    return lo+(hi-lo)*u


def direction_bank(nsrc:int,nph:int,seed:int):
    rng=np.random.default_rng(seed);N=nsrc*nph
    mu=rng.uniform(-1,1,N);phi=rng.uniform(0,2*math.pi,N);st=np.sqrt(np.maximum(0,1-mu*mu))
    dirs=np.column_stack([st*np.cos(phi),st*np.sin(phi),mu])
    sid=np.repeat(np.arange(nsrc,dtype=np.int64),nph);pid=np.tile(np.arange(nph,dtype=np.int64),nsrc)
    return dirs,sid,pid


def make_batch(spec:GeometrySpec,u:np.ndarray,dirs:np.ndarray,sid:np.ndarray):
    pts=u*np.asarray(spec.cell_cm,float)[None,:];pts=np.repeat(pts, np.sum(sid==0), axis=0)
    return PhotonBatch(interaction_points=pts,directions=dirs,times=np.zeros(len(dirs)),statuses=np.full(len(dirs),int(PhotonStatus.RUNNING_INSIDE),dtype=np.int8)),u*np.asarray(spec.cell_cm,float)[None,:]


def channels_from_final(scene,end):
    status=np.asarray(end.statuses);ch=np.full(len(status),-1,np.int8)
    for i in np.flatnonzero(status==int(PhotonStatus.CAUGHT)):
        c=scene.readout_channel_at(end.interaction_points[i],end.directions[i])
        if c is None:raise RuntimeError(f"CAUGHT photon {i} has no channel")
        ch[i]=int(c)
    return ch


def ds(group,name,data):
    a=np.asarray(data);kw={}
    if a.size:kw=dict(compression="gzip",compression_opts=4,shuffle=True,chunks=True)
    group.create_dataset(name,data=a,**kw)


def geometry_hash(spec:GeometrySpec):
    s=json.dumps(spec.raw,sort_keys=True,separators=(",",":"));return hashlib.sha256(s.encode()).hexdigest()


def write_geometry(h5,spec,path0,path1,sources,start,end,sid,pid,ch,facets):
    g=h5["geometries"].create_group(spec.geometry_id)
    X,Y,Z=map(float,spec.cell_cm)
    for k,v in dict(geometry_id=spec.geometry_id,cell_x_cm=X,cell_y_cm=Y,cell_z_cm=Z,width_cm=spec.width_cm,radius_cm=spec.radius_cm,r_over_w=spec.r_over_w,n_turns=len(spec.turns_xy_cm),mirrored_guides=True,guide_thickness_cm=GUIDE_THICKNESS,cladding_thickness_cm=CLADDING_THICKNESS,facets_per_90=facets,topology="mirrored_rounded_polyline",design_sha256=geometry_hash(spec),design_json=json.dumps(spec.raw,sort_keys=True)).items():g.attrs[k]=v
    ds(g,"start_xy_cm",spec.start_xy_cm);ds(g,"turns_xy_cm",spec.turns_xy_cm);ds(g,"end_xy_cm",spec.end_xy_cm);ds(g,"guide0_points_cm",path0.points);ds(g,"guide0_tangents",path0.tangents);ds(g,"guide1_points_cm",path1.points);ds(g,"guide1_tangents",path1.tangents)
    sg=g.create_group("sources");ds(sg,"source_id",np.arange(len(sources),dtype=np.int64));ds(sg,"x_cm",sources[:,0]);ds(sg,"y_cm",sources[:,1]);ds(sg,"z_cm",sources[:,2]);ds(sg,"u",sources[:,0]/X);ds(sg,"v",sources[:,1]/Y);ds(sg,"w",sources[:,2]/Z)
    p=g.create_group("photons");ds(p,"source_id",sid);ds(p,"photon_id",pid);ds(p,"start_direction",start.directions);ds(p,"end_status",end.statuses);ds(p,"channel",ch);ds(p,"transport_time_ns",end.times-start.times);ds(p,"end_point_cm",end.interaction_points);ds(p,"end_direction",end.directions)


def main():
    ap=argparse.ArgumentParser(description="PHINDER Stage-6 v3 arbitrary-geometry optical atlas")
    ap.add_argument("--geometry-json",required=True,type=Path);ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--max-turns",type=int,default=10);ap.add_argument("--sources",type=int,default=32);ap.add_argument("--photons-per-source",type=int,default=2000)
    ap.add_argument("--source-seed",type=int,default=629001);ap.add_argument("--direction-seed",type=int,default=629002);ap.add_argument("--margin-xy-frac",type=float,default=.05);ap.add_argument("--margin-z-frac",type=float,default=.05)
    ap.add_argument("--facets-per-90",type=int,default=16);ap.add_argument("--jobs",type=int,default=min(10,os.cpu_count() or 1));ap.add_argument("--resume",action="store_true")
    args=ap.parse_args();specs=parse_geometry_json(args.geometry_json,args.max_turns)
    # Preflight every geometry before any expensive photons.
    built={}
    for s in specs:
        scene,p0,p1=build_scene(s,args.facets_per_90);built[s.geometry_id]=(scene,p0,p1)
        print(f"preflight PASS {s.geometry_id}: cell={s.cell_cm.tolist()} w={s.width_cm:g} R={s.radius_cm:g} turns={len(s.turns_xy_cm)} segments={len(p0.points)-1}")
    u=normalized_sources(args.sources,args.source_seed,args.margin_xy_frac,args.margin_z_frac);dirs,sid,pid=direction_bank(args.sources,args.photons_per_source,args.direction_seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    mode="a" if args.resume and args.output.exists() else "w"
    with h5py.File(args.output,mode) as h5:
        if mode=="w":
            h5.attrs["format"]=FORMAT;h5.attrs["format_version"]=FORMAT_VERSION;h5.attrs["source_type"]="point_isotropic_zero_emission_time";h5.attrs["timing_binned"]=False;h5.attrs["n_sources"]=args.sources;h5.attrs["photons_per_source"]=args.photons_per_source;h5.attrs["source_seed"]=args.source_seed;h5.attrs["direction_seed"]=args.direction_seed;h5.attrs["max_turns"]=args.max_turns;h5.attrs["note"]="single-point optical transport only; scintillation production/delay excluded";h5.create_group("geometries")
        else:
            if str(h5.attrs.get("format",""))!=FORMAT or int(h5.attrs.get("format_version",-1))!=FORMAT_VERSION:raise ValueError("resume target is not a v3 operator atlas")
            for k,v in (("n_sources",args.sources),("photons_per_source",args.photons_per_source),("source_seed",args.source_seed),("direction_seed",args.direction_seed)):
                if int(h5.attrs[k])!=int(v):raise ValueError(f"resume metadata mismatch: {k}")
        for j,s in enumerate(specs,1):
            if s.geometry_id in h5["geometries"]:
                old=h5["geometries"][s.geometry_id].attrs.get("design_sha256","")
                if old!=geometry_hash(s):raise ValueError(f"resume geometry {s.geometry_id} exists with different design")
                print(f"[{j}/{len(specs)}] skip completed {s.geometry_id}");continue
            scene,p0,p1=built[s.geometry_id];start,sources=make_batch(s,u,dirs,sid)
            print(f"[{j}/{len(specs)}] simulate {s.geometry_id}: {len(dirs)} photons")
            sim=PhotonBatchSimulator(ApproximateEngine(scene=scene),parallel_config=ParallelConfig(n_workers=args.jobs));end=sim.run(start,parallel=(args.jobs>1),show_progress=True);ch=channels_from_final(scene,end)
            write_geometry(h5,s,p0,p1,sources,start,end,sid,pid,ch,args.facets_per_90);h5.flush()
            c0=int(np.sum(ch==0));c1=int(np.sum(ch==1));print(f"    caught={c0+c1} ({100*(c0+c1)/len(ch):.3f}%) ch0/ch1={c0}/{c1}")
    print(f"wrote {args.output}")

if __name__=="__main__":main()
