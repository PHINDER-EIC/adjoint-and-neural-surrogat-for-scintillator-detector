"""Geant4 optical transport of scintillation photons from muon energy deposits (same scene and physics as g4_sim.py).

Input npz (from make_photons.py): pos [n,3] mm, t [n] ns (deposit time + emission delay), event [n].
Photons are isotropic with random linear polarisation. Modes:
  unpol : as g4_sim.py unpol (the configuration that matches the optical model): air kills, polarisation redrawn per step
  full  : true polarisation, photons in air keep propagating; detection at the ports (Python stepping action)
  bare  : as full but without any user stepping action (no detection), to time Geant4's own tracking without the
          per-step Python callback
Output: caught photons (photon index, channel, arrival time), wall and CPU time.

usage: python g4_ev_opt.py <gid> <mode> <photons.npz> <first> <count> <seed> <out.npz>
"""
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.environ.get("PHADJ_EVENTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "derived", "events"))
os.environ.setdefault("GEANT4_DATA_DIR", os.path.join(HERE, "..", "g4", "nodata"))
from geant4_pybind import *  # noqa: E402,F403
import json  # noqa: E402

gid, mode, PHF = sys.argv[1], sys.argv[2], sys.argv[3]
FIRST, COUNT, SEED, OUTF = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
PH = np.load(PHF)
POS = PH["pos"][FIRST:FIRST + COUNT]; TT = PH["t"][FIRST:FIRST + COUNT]
N = len(TT)
PER_EVENT = 500
G = np.load(os.path.join(HERE, "..", "g4", "geom", f"{gid}.npz"))
MATS = json.loads(str(G["materials_json"]))
CELL = G["cell_cm"] * 10.0
E_PH = 3.04 * eV
KILL_AIR = mode == "unpol"
UNPOL = mode == "unpol"
PORT = [(G[f"port{k}_point"] * 10.0, G[f"port{k}_normal"]) for k in (0, 1)]
WORLD, SCINT, CLAD0, CORE0, CLAD1, CORE1 = range(6)
m_unit = m


def optical_material(name, density, n, mu_per_cm):
    mt = G4Material(name, density, 2)
    mt.AddElement(G4NistManager.Instance().FindOrBuildElement("C"), 9)
    mt.AddElement(G4NistManager.Instance().FindOrBuildElement("H"), 10)
    t = G4MaterialPropertiesTable()
    e = [1.5 * eV, 6.2 * eV]
    t.AddProperty("RINDEX", G4doubleVector(e), G4doubleVector([n, n]))
    L = (1.0 / mu_per_cm) * cm if mu_per_cm > 0 else 1e9 * m_unit
    t.AddProperty("ABSLENGTH", G4doubleVector(e), G4doubleVector([L, L]))
    mt.SetMaterialPropertiesTable(t)
    return mt, t


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
        self.optical = G4OpticalPhysics()
        self.RegisterPhysics(self.optical)


rng = np.random.default_rng(SEED)
random.seed(SEED)
gauss = random.gauss
caught = []          # (photon index, channel, time)
nxt = [0]


class Gen(G4VUserPrimaryGeneratorAction):
    def __init__(self):
        super().__init__()
        self.gun = G4ParticleGun(1)
        self.gun.SetParticleDefinition(G4OpticalPhoton.Definition())
        self.gun.SetParticleEnergy(E_PH)

    def GeneratePrimaries(self, event):
        a = nxt[0]; b = min(a + PER_EVENT, N); nxt[0] = b
        d = rng.normal(size=(b - a, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        for i in range(a, b):
            di = d[i - a]
            self.gun.SetParticlePosition(G4ThreeVector(*POS[i]))
            self.gun.SetParticleTime(float(TT[i]))
            self.gun.SetParticleMomentumDirection(G4ThreeVector(*di))
            q = rng.normal(size=3); q -= q.dot(di) * di; q /= np.linalg.norm(q)
            self.gun.SetParticlePolarization(G4ThreeVector(*q))
            self.gun.GeneratePrimaryVertex(event)


class Stepping(G4UserSteppingAction):
    def UserSteppingAction(self, step):
        pre = step.GetPreStepPoint()
        cn = pre.GetTouchable().GetCopyNumber()
        track = step.GetTrack()
        if cn == WORLD:
            if KILL_AIR and step.GetStepLength() > 1e-7:
                track.SetTrackStatus(G4TrackStatus.fStopAndKill)
            return
        post = step.GetPostStepPoint()
        if post.GetStepStatus() != G4StepStatus.fGeomBoundary:
            return
        if cn == CORE0 or cn == CORE1:
            k = (cn - CORE0) // 2
            x = post.GetPosition()
            P, Nn = PORT[k]
            if abs((x.x - P[0]) * Nn[0] + (x.y - P[1]) * Nn[1] + (x.z - P[2]) * Nn[2]) < 1e-5:
                caught.append((FIRST + track.GetTrackID() - 1 + track_base[0], k, post.GetGlobalTime()))
                track.SetTrackStatus(G4TrackStatus.fStopAndKill)
                return
        if UNPOL:
            d = track.GetMomentumDirection()
            ax, ay, az = gauss(0, 1), gauss(0, 1), gauss(0, 1)
            s = ax * d.x + ay * d.y + az * d.z
            ax -= s * d.x; ay -= s * d.y; az -= s * d.z
            r = math.sqrt(ax * ax + ay * ay + az * az)
            track.SetPolarization(G4ThreeVector(ax / r, ay / r, az / r))


track_base = [0]


class EventAct(G4UserEventAction):
    def BeginOfEventAction(self, event):
        track_base[0] = event.GetEventID() * PER_EVENT     # primaries of event e are photons e*PER_EVENT + (trackID-1)


class Actions(G4VUserActionInitialization):
    def Build(self):
        self.SetUserAction(Gen())
        self.SetUserAction(EventAct())
        if mode != "bare":
            self.SetUserAction(Stepping())


ui = G4UImanager.GetUIpointer()
rm = G4RunManagerFactory.CreateRunManager(G4RunManagerType.Serial)
G4Random.setTheSeed(SEED % 2147483647)
rm.SetUserInitialization(Det())
rm.SetUserInitialization(Physics())
for proc in ("Cerenkov", "Scintillation", "OpWLS", "OpWLS2", "OpRayleigh", "OpMieHG"):
    ui.ApplyCommand(f"/process/optical/processActivation {proc} false")
ui.ApplyCommand("/process/optical/verbose 0")
rm.SetUserInitialization(Actions())
for c in ("/control/verbose 0", "/run/verbose 0", "/event/verbose 0", "/tracking/verbose 0"):
    ui.ApplyCommand(c)
rm.Initialize()
t0 = time.perf_counter(); cpu0 = time.process_time()
rm.BeamOn(int(math.ceil(N / PER_EVENT)))
wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
c = np.array(caught, dtype=np.float64).reshape(-1, 3)
np.savez_compressed(OUTF, idx=c[:, 0].astype(np.int64), ch=c[:, 1].astype(np.int8), t=c[:, 2], n=N, first=FIRST,
                    wall_s=wall, cpu_s=cpu, mode=mode, seed=SEED)
print(f"{gid} {mode} photons={N} caught0 {np.sum(c[:, 1] == 0) / N:.5f} caught1 {np.sum(c[:, 1] == 1) / N:.5f} "
      f"{wall:.1f}s ({N / wall:.0f} ph/s)", flush=True)
sys.stdout.flush()
os._exit(0)
