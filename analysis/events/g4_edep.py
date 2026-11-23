"""Geant4 energy deposition of muons crossing a Stage-6B cell (no optical photons).

Scene as g4_sim.py (scintillator box, guide cores and 10-um clad shells in air), materials with real densities.
Physics: G4EmStandardPhysics (no decay: irrelevant for a 1-cm crossing). Primary: mu-, 4 GeV, zenith angle from the flux through a horizontal
plane (pdf ~ cos^3(theta) sin(theta)), crossing the top face at a uniform random point. Every step in the scintillator
with energy loss is stored (pre/post position in mm, pre/post global time in ns, edep in MeV, step length in mm, pdg).

usage: python g4_edep.py <gid> <n_muons> <seed> <out.npz>
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.environ.get("PHADJ_EVENTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "derived", "events"))
from geant4_pybind import *  # noqa: E402,F403

gid, NMU, SEED, OUTF = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
G = np.load(os.path.join(HERE, "..", "g4", "geom", f"{gid}.npz"))
CELL = G["cell_cm"] * 10.0
E_MU = 4.0 * GeV
WORLD, SCINT, CLAD0, CORE0, CLAD1, CORE1 = range(6)


def mat(name, density, comp):
    nist = G4NistManager.Instance()
    m = G4Material(name, density, len(comp))
    for el, n in comp:
        m.AddElement(nist.FindOrBuildElement(el), n)
    return m


class Det(G4VUserDetectorConstruction):
    def Construct(self):
        self.keep = []
        nist = G4NistManager.Instance()
        air = nist.FindOrBuildMaterial("G4_AIR")
        scm = mat("EJ204", 1.023 * g / cm3, [("C", 9), ("H", 10)])       # PVT base
        core = mat("core", 1.05 * g / cm3, [("C", 8), ("H", 8)])          # polystyrene
        clad = mat("clad", 1.19 * g / cm3, [("C", 5), ("H", 8), ("O", 2)])  # PMMA
        self.keep += [scm, core, clad]
        half = CELL / 2
        ws = G4Box("world", 3 * half[0] + 50, 3 * half[1] + 50, 3 * half[2] + 50)
        wl = G4LogicalVolume(ws, air, "world")
        wp = G4PVPlacement(None, G4ThreeVector(), wl, "world", None, False, WORLD)
        ss = G4Box("scint", *half)
        sl = G4LogicalVolume(ss, scm, "scint")
        sp = G4PVPlacement(None, G4ThreeVector(*half), sl, "scint", wl, False, SCINT)
        self.keep += [ws, wl, wp, ss, sl, sp]
        for k in (0, 1):
            for nm, mt, cn in ((f"shell{k}", clad, CLAD0 + 2 * k), (f"core{k}", core, CORE0 + 2 * k)):
                ts = G4TessellatedSolid(nm)
                for tri in G[nm] * 10.0:
                    ts.AddFacet(G4TriangularFacet(G4ThreeVector(*tri[0]), G4ThreeVector(*tri[1]), G4ThreeVector(*tri[2]),
                                                  G4FacetVertexType.ABSOLUTE))
                ts.SetSolidClosed(True)
                lv = G4LogicalVolume(ts, mt, nm)
                pv = G4PVPlacement(None, G4ThreeVector(), lv, nm, wl, False, cn)
                self.keep += [ts, lv, pv]
        return wp


class Physics(G4VModularPhysicsList):
    def __init__(self):
        super().__init__()
        self.SetVerboseLevel(0)
        self.em = G4EmStandardPhysics()
        self.RegisterPhysics(self.em)

    def ConstructParticle(self):
        # G4EmStandardPhysics' own particle set crashes this Windows build of geant4_pybind; define the particles here
        for n in PARTICLES:
            globals()[n].Definition()

    def SetCuts(self):
        self.SetDefaultCutValue(0.7 * mm)                 # Geant4's usual default production cut
        self.SetCutsWithDefault()


PARTICLES = ["G4Gamma", "G4Electron", "G4Positron", "G4MuonMinus", "G4MuonPlus", "G4Geantino", "G4ChargedGeantino", "G4NeutrinoE",
             "G4AntiNeutrinoE", "G4NeutrinoMu", "G4AntiNeutrinoMu", "G4PionPlus", "G4PionMinus", "G4PionZero", "G4KaonPlus",
             "G4KaonMinus", "G4Proton", "G4AntiProton", "G4Neutron", "G4Deuteron", "G4Triton", "G4He3", "G4Alpha", "G4GenericIon"]


rng = np.random.default_rng(SEED)
rows = []          # event, pdg, x0,y0,z0, x1,y1,z1, t0, t1, edep, len
cur = [0]


class Gen(G4VUserPrimaryGeneratorAction):
    def __init__(self):
        super().__init__()
        self.gun = G4ParticleGun(1)
        self.gun.SetParticleDefinition(G4MuonMinus.Definition())
        self.gun.SetParticleEnergy(E_MU)

    def GeneratePrimaries(self, event):
        cur[0] = event.GetEventID()
        c = rng.random() ** 0.25                       # cos(theta), pdf ~ cos^3 sin
        s = math.sqrt(1 - c * c); ph = 2 * math.pi * rng.random()
        d = np.array([s * math.cos(ph), s * math.sin(ph), -c])
        hit = np.array([rng.random() * CELL[0], rng.random() * CELL[1], CELL[2]])
        start = hit - d * 1.0                          # 1 mm above the top face, in air
        self.gun.SetParticlePosition(G4ThreeVector(*start))
        self.gun.SetParticleMomentumDirection(G4ThreeVector(*d))
        self.gun.SetParticleTime(0.0)
        self.gun.GeneratePrimaryVertex(event)


class Stepping(G4UserSteppingAction):
    def UserSteppingAction(self, step):
        e = step.GetTotalEnergyDeposit()
        if e <= 0:
            return
        pre = step.GetPreStepPoint()
        if pre.GetTouchable().GetCopyNumber() != SCINT:
            return
        post = step.GetPostStepPoint()
        a, b = pre.GetPosition(), post.GetPosition()
        rows.append((cur[0], step.GetTrack().GetDefinition().GetPDGEncoding(), a.x, a.y, a.z, b.x, b.y, b.z,
                     pre.GetGlobalTime(), post.GetGlobalTime(), e / MeV, step.GetStepLength() / mm))


class Actions(G4VUserActionInitialization):
    def Build(self):
        self.SetUserAction(Gen())
        self.SetUserAction(Stepping())


ui = G4UImanager.GetUIpointer()
rm = G4RunManagerFactory.CreateRunManager(G4RunManagerType.Serial)
G4Random.setTheSeed(SEED % 2147483647)
DET = Det(); rm.SetUserInitialization(DET)
PHYS = Physics(); rm.SetUserInitialization(PHYS)
ACT = Actions(); rm.SetUserInitialization(ACT)
for c in ("/control/verbose 0", "/run/verbose 0", "/event/verbose 0", "/tracking/verbose 0", "/process/em/verbose 0"):
    ui.ApplyCommand(c)
rm.Initialize()
t0 = time.perf_counter(); cpu0 = time.process_time()
rm.BeamOn(NMU)
wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
r = np.array(rows, dtype=np.float64).reshape(-1, 12)
np.savez_compressed(OUTF, event=r[:, 0].astype(np.int32), pdg=r[:, 1].astype(np.int32), pre=r[:, 2:5], post=r[:, 5:8],
                    t_pre=r[:, 8], t_post=r[:, 9], edep=r[:, 10], length=r[:, 11], n_muons=NMU, wall_s=wall, cpu_s=cpu,
                    cell_mm=CELL, seed=SEED, gid=gid)
E = np.bincount(r[:, 0].astype(int), weights=r[:, 10], minlength=NMU)
print(f"{gid} muons={NMU} steps={len(r)} mean edep {E.mean():.3f} MeV (median {np.median(E):.3f}), "
      f"{wall:.1f} s ({1e3 * wall / NMU:.2f} ms/muon)", flush=True)
sys.stdout.flush()
os._exit(0)                                        # skip Geant4 teardown (crashes in this build)
