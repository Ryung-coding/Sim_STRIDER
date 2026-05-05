from acados_template.acados_ocp_solver import AcadosOcpSolver

import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from .mmap_manager import MMapWriter

# Make codegen quieter.
os.environ.setdefault("MAKEFLAGS", "-s")

BASE_DIR = Path(__file__).resolve().parent

class StriderNMPC:
  def __init__(self):
    # Build USE-FULL solver.
    from .use_full.model import build_ocp as use_full_build_ocp
    self.use_full_ocp = use_full_build_ocp()
    use_full_json_path = BASE_DIR / "use_full" / f"{self.use_full_ocp.model.name}.json"
    self.use_full_solver = AcadosOcpSolver(self.use_full_ocp, json_file=str(use_full_json_path))

    self.nx = int(self.use_full_ocp.model.x.size()[0])
    self.nu = int(self.use_full_ocp.model.u.size()[0])
    self.np = int(self.use_full_ocp.model.p.size()[0])

    from . import params as p
    self.N = int(p.N)

    mmap_path = os.environ.get("MRG_MMAP", "/tmp/MRG_debug.mmap")
    self._mmap_writer = MMapWriter(mmap_path, self.N, self.nx, self.nu, self.np)

    # Horizon buffers for mmap/debug output.
    self._xs = np.zeros((self.N + 1, self.nx), dtype=np.float64)
    self._us = np.zeros((self.N, self.nu), dtype=np.float64)
    self._ps = np.zeros((self.N + 1, self.np), dtype=np.float64)

    # Returned stage-wise outputs.
    self._x_stage_steps = np.zeros((self.nx, self.N), dtype=np.float64, order="F")
    self._u_stage_steps = np.zeros((self.nu, self.N), dtype=np.float64, order="F")

  def _set_initial_guess_all_stages(self, solver: AcadosOcpSolver, x0: np.ndarray, u0: np.ndarray, p0: np.ndarray) -> None:
    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    for k in range(self.N + 1):
      solver.set(k, "x", x0)
      solver.set(k, "p", p0)

    for k in range(self.N):
      solver.set(k, "u", u0)

  def _solve(self, x_0, u_0, p, steps_req: int):
    x_0 = np.asarray(x_0, dtype=np.float64).ravel()
    u_0 = np.asarray(u_0, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()

    self._set_initial_guess_all_stages(self.use_full_solver, x_0, u_0, p)

    t0 = time.perf_counter()
    status = self.use_full_solver.solve()
    solve_ms = (time.perf_counter() - t0) * 1000.0

    xs, us, ps, x_stage, u_stage = self._extract_all_xup(steps_req)

    self._mmap_writer.write(x_all=xs, u_all=us, p_all=ps, solve_ms=float(solve_ms), status=int(status))

    return (
      x_stage[:, 0:steps_req],
      u_stage[:, 0:steps_req],
      float(solve_ms),
      int(status),
    )

  def compute_MPC(self, mpci: Dict[str, Any]) -> Dict[str, Any]:
    x_0 = np.asarray(mpci.get("x_0", np.zeros(self.nx)), dtype=np.float64).ravel()
    u_0 = np.asarray(mpci.get("u_0", np.zeros(self.nu)), dtype=np.float64).ravel()
    p = np.asarray(mpci.get("p", np.zeros(self.np)), dtype=np.float64).ravel()

    steps_req = int(mpci.get("steps_req", 1))
    if steps_req < 0 or steps_req > self.N:
      raise ValueError(f"steps_req out of range: got {steps_req}, valid=[0, {self.N}]")

    if x_0.size != self.nx:
      raise ValueError(f"x_0 size mismatch: got {x_0.size}, expected {self.nx}")
    if p.size != self.np:
      raise ValueError(f"p size mismatch: got {p.size}, expected {self.np}")
    if u_0.size != self.nu:
      u_0 = np.zeros(self.nu, dtype=np.float64)

    x_stage, u_stage, solve_ms, status = self._solve(x_0, u_0, p, steps_req)

    return {
      "x_stage": x_stage,
      "u_stage": u_stage,
      "solve_ms": float(solve_ms),
      "state": int(status),
    }

  def _extract_all_xup(self, steps_req: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N = self.N
    sol = self.use_full_solver
    xs = self._xs
    us = self._us
    ps = self._ps

    self._x_stage_steps.fill(0.0)
    self._u_stage_steps.fill(0.0)
    xs.fill(0.0)
    us.fill(0.0)
    ps.fill(0.0)

    for k in range(N + 1):
      xk = sol.get(k, "x").reshape(-1)
      pk = sol.get(k, "p").reshape(-1)

      xs[k, :] = xk
      ps[k, :] = pk

      if 1 <= k <= steps_req:
        self._x_stage_steps[:, k - 1] = xk

    for k in range(N):
      uk = sol.get(k, "u").reshape(-1)

      us[k, :] = uk

      if k < steps_req:
        self._u_stage_steps[:, k] = uk

    return xs, us, ps, self._x_stage_steps, self._u_stage_steps