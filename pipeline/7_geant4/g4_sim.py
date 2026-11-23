"""Geant4 (11.4, geant4_pybind) optical forward simulation of a Stage-6B test geometry.

Scene from g4_export.py: scintillator box, per guide a core mesh (segments + terminal) and a 10-um clad shell, in air.
Physics: G4OpticalPhysics with only OpAbsorption and OpBoundary active (polished dielectric_dielectric
boundaries with no optical surfaces defined: Fresnel/Snell with polarization), constant RINDEX and ABSLENGTH
equal to the repo materials, photon energy 3.04 eV (408 nm).
Detection: a photon in a core that reaches the readout-port plane of that guide is caught (channel 0/1) and killed.
Modes:
  unpol : photons entering air are killed (as the repo engines do) and the polarization is re-randomized
          before every step -> reflection probability (Rs+Rp)/2 per interface, the repo engines' model
  pol   : as unpol but true polarization tracking (Geant4 default)
  full  : pol, and photons in air keep propagating (can re-enter the guide or scintillator)
Emission: isotropic, t = 0, uniform in the voxel box [lo, hi) of the 20x20x8 grid; each caught photon is stored
with its emission voxel, channel and time.

usage: python g4_sim.py <gid> <mode> <lo x,y,z> <hi x,y,z> <n_photons> <seed> <out.npz>
"""
import math
import os
import sys
import time
import random

import numpy as np

os.environ.setdefault("GEANT4_DATA_DIR", os.path.join(os.environ.get("PHADJ_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "work")), "g4", "nodata"))
os.makedirs(os.environ["GEANT4_DATA_DIR"], exist_ok=True)
from geant4_pybind import *  # noqa: E402,F403

gid, mode = sys.argv[1], sys.argv[2]
lo = np.array([int(x) for x in sys.argv[3].split(",")])
hi = np.array([int(x) for x in sys.argv[4].split(",")])
NPH, SEED, OUTF = int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
PER_EVENT = 500
G = np.load(os.path.join(os.environ.get("PHADJ_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "work")), "g4", "geom", f"{gid}.npz"))
import json  # noqa: E402
MATS = json.loads(str(G["materials_json"]))
CELL = G["cell_cm"] * 10.0                          # mm
SP = CELL / np.array([20, 20, 8])
E_PH = 3.04 * eV
KILL_AIR = mode in ("unpol", "pol")
UNPOL = mode == "unpol"

PORT = [(G[f"port{k}_point"] * 10.0, G[f"port{k}_normal"]) for k in (0, 1)]
WORLD, SCINT, CLAD0, CORE0, CLAD1, CORE1 = range(6)


def optical_material(name, density, n, mu_per_cm):
    m = G4Material(name, density, 2)
    m.AddElement(G4NistManager.Instance().FindOrBuildElement("C"), 9)
    m.AddElement(G4NistManager.Instance().FindOrBuildElement("H"), 10)
    t = G4MaterialPropertiesTable()
    e = [1.5 * eV, 6.2 * eV]
    t.AddProperty("RINDEX", G4doubleVector(e), G4doubleVector([n, n]))
    L = (1.0 / mu_per_cm) * cm if mu_per_cm > 0 else 1e9 * m_unit
    t.AddProperty("ABSLENGTH", G4doubleVector(e), G4doubleVector([L, L]))
    m.SetMaterialPropertiesTable(t)
    return m, t


m_unit = m


class Det(G4VUserDetectorConstruction):
    def Construct(self):
        self.keep = []
        nist = G4NistManager.Instance()
        air = G4Material("vac_air", 1.2e-3 * g / cm3, 1)
        air.AddElement(nist.FindOrBuildElement("N"), 1)
        ta = G4MaterialPropertiesTable()
        ta.AddProperty("RINDEX", G4doubleVector([1.5 * eV, 6.2 * eV]), G4doubleVector([MATS["air"][0]] * 2))
        air.SetMaterialPropertiesTable(ta)
        scm, t1 = optical_material("EJ204", 1.023 * g / cm3, *MATS["EJ-204"])
        core, t2 = optical_material("core", 1.05 * g / cm3, *MATS["fiber_core"])
        clad, t3 = optical_material("clad", 1.19 * g / cm3, *MATS["fiber_clad"])
        self.keep += [air, ta, scm, t1, core, t2, clad, t3]
        half = CELL / 2
        ws = G4Box("world", 3 * half[0] + 50, 3 * half[1] + 50, 3 * half[2] + 50)
        wl = G4LogicalVolume(ws, air, "world")
        wp = G4PVPlacement(None, G4ThreeVector(), wl, "world", None, False, WORLD)
        ss = G4Box("scint", *half)
        sl = G4LogicalVolume(ss, scm, "scint")
        sp = G4PVPlacement(None, G4ThreeVector(*half), sl, "scint", wl, False, SCINT)
        self.keep += [ws, wl, wp, ss, sl, sp]
        for k in (0, 1):
            for nm, mat, cn in ((f"shell{k}", clad, CLAD0 + 2 * k), (f"core{k}", core, CORE0 + 2 * k)):
                ts = G4TessellatedSolid(nm)
                for tri in G[nm] * 10.0:
                    f = G4TriangularFacet(G4ThreeVector(*tri[0]), G4ThreeVector(*tri[1]), G4ThreeVector(*tri[2]), G4FacetVertexType.ABSOLUTE)
                    ts.AddFacet(f)
                ts.SetSolidClosed(True)
                lv = G4LogicalVolume(ts, mat, nm)
                pv = G4PVPlacement(None, G4ThreeVector(), lv, nm, wl, False, cn)
                self.keep += [ts, lv, pv]
        return wp


class Physics(G4VModularPhysicsList):
    def __init__(self):
        super().__init__()
        self.SetVerboseLevel(0)
        self.optical = G4OpticalPhysics()
        self.RegisterPhysics(self.optical)


rng = np.random.default_rng(SEED)
random.seed(SEED)
gauss = random.gauss
caught = []          # (voxel flat index, channel, time ns)
emitted_vox = np.zeros(20 * 20 * 8, np.int64)


def iso(n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def perp(d):
    a = rng.normal(size=3)
    a -= a.dot(d) * d
    return a / np.linalg.norm(a)


class Gen(G4VUserPrimaryGeneratorAction):
    def __init__(self):
        super().__init__()
        self.gun = G4ParticleGun(1)
        self.gun.SetParticleDefinition(G4OpticalPhoton.Definition())
        self.gun.SetParticleEnergy(E_PH)
        self.gun.SetParticleTime(0.0)

    def GeneratePrimaries(self, event):
        n = PER_EVENT
        pos = (lo + rng.random((n, 3)) * (hi - lo)) * SP
        vox = np.minimum((pos / SP).astype(int), [19, 19, 7])
        np.add.at(emitted_vox, (vox[:, 0] * 20 + vox[:, 1]) * 8 + vox[:, 2], 1)
        d = iso(n)
        for i in range(n):
            self.gun.SetParticlePosition(G4ThreeVector(*pos[i]))
            self.gun.SetParticleMomentumDirection(G4ThreeVector(*d[i]))
            p = perp(d[i])
            self.gun.SetParticlePolarization(G4ThreeVector(*p))
            self.gun.GeneratePrimaryVertex(event)


class Stepping(G4UserSteppingAction):
    def UserSteppingAction(self, step):
        pre = step.GetPreStepPoint()
        cn = pre.GetTouchable().GetCopyNumber()
        track = step.GetTrack()
        if cn == WORLD:
            # a reflected photon makes a zero-length step whose pre-step volume is the air: keep it
            if KILL_AIR and step.GetStepLength() > 1e-7:
                track.SetTrackStatus(G4TrackStatus.fStopAndKill)
            return
        post = step.GetPostStepPoint()
        if post.GetStepStatus() != G4StepStatus.fGeomBoundary:
            return
        if cn == CORE0 or cn == CORE1:
            k = (cn - CORE0) // 2
            x = post.GetPosition()
            P, N = PORT[k]
            if abs((x.x - P[0]) * N[0] + (x.y - P[1]) * N[1] + (x.z - P[2]) * N[2]) < 1e-5:
                v = track.GetVertexPosition()
                ix = min(int(v.x / SP[0]), 19); iy = min(int(v.y / SP[1]), 19); iz = min(int(v.z / SP[2]), 7)
                caught.append(((ix * 20 + iy) * 8 + iz, k, post.GetGlobalTime()))
                track.SetTrackStatus(G4TrackStatus.fStopAndKill)
                return
        if UNPOL:
            d = track.GetMomentumDirection()
            ax, ay, az = gauss(0, 1), gauss(0, 1), gauss(0, 1)
            s = ax * d.x + ay * d.y + az * d.z
            ax -= s * d.x; ay -= s * d.y; az -= s * d.z
            r = math.sqrt(ax * ax + ay * ay + az * az)
            track.SetPolarization(G4ThreeVector(ax / r, ay / r, az / r))


class Actions(G4VUserActionInitialization):
    def Build(self):
        self.SetUserAction(Gen())
        self.SetUserAction(Stepping())


ui = G4UImanager.GetUIpointer()
rm = G4RunManagerFactory.CreateRunManager(G4RunManagerType.Serial)
G4Random.setTheSeed(SEED % 2147483647)
rm.SetUserInitialization(Det())
PHYS = Physics()
rm.SetUserInitialization(PHYS)
for proc in ("Cerenkov", "Scintillation", "OpWLS", "OpWLS2", "OpRayleigh", "OpMieHG"):
    ui.ApplyCommand(f"/process/optical/processActivation {proc} false")
ui.ApplyCommand("/process/optical/verbose 0")
rm.SetUserInitialization(Actions())
for c in ("/control/verbose 0", "/run/verbose 0", "/event/verbose 0", "/tracking/verbose 0"):
    ui.ApplyCommand(c)
rm.Initialize()
if os.environ.get('G4DUMP'):
    ui.ApplyCommand('/particle/select opticalphoton'); ui.ApplyCommand('/particle/process/dump')
t0 = time.perf_counter()
cpu0 = time.process_time()
rm.BeamOn(int(math.ceil(NPH / PER_EVENT)))
wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
c = np.array(caught, dtype=np.float64).reshape(-1, 3)
np.savez_compressed(OUTF, vox=c[:, 0].astype(np.int32), ch=c[:, 1].astype(np.int8), t=c[:, 2],
                    emitted_vox=emitted_vox, wall_s=wall, cpu_s=cpu, mode=mode, lo=lo, hi=hi, seed=SEED)
n = emitted_vox.sum()
print(f"{gid} {mode} n={n} caught0 {np.sum(c[:,1]==0)/n:.5f} caught1 {np.sum(c[:,1]==1)/n:.5f} "
      f"t0 {c[c[:,1]==0,2].mean() if len(c) else float('nan'):.3f} ns  {wall:.1f}s ({n/wall:.0f} ph/s)", flush=True)
