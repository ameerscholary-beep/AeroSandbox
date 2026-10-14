# design_agent — Uncertainty-Aware Multi-Agent AI for Automated Aircraft Conceptual Design

Natural-language mission statement in → optimized, *verified*, robustness-quantified
aircraft design out. Built on AeroSandbox + NeuralFoil with a CasADi/IPOPT
optimizer, a numeric truth layer, and a Monte-Carlo robustness stage.

```bash
python -m design_agent.cli "Design a 4-seat piston trainer. Payload 340 kg plus
2 crew. Range 800 km, cruise 60 m/s at 3000 m. Takeoff under 600 m, stall below
30 m/s, 20 min reserve, climb at least 5 m/s." \
    --robustness 16 --figures figs/ --report report.md --out manifest.json
```

Exit code 0 = every constraint passed. One command produces the optimized
design, a constraint-by-constraint verification table, Monte-Carlo robustness
statistics, three figures, a Markdown report, and a JSON manifest.

## Pipeline

```
mission text ──► nl_parser ──► MissionSpec (pydantic)
                   │ LLM client optional; deterministic rule-based fallback
                   ▼
              optimizer ──► Opti NLP: symbolic DesignVector + trim alphas
                   │        + TOGW closure; objective = gross mass
                   │  polar_lut: NeuralFoil→CasADi interpolant LUTs patch the
                   │  broken NF symbolic path during problem assembly only
                   │  failure philosophy: never raise — return_last iterate +
                   │  full numeric diagnostics; repair loop reseeds flat
                   │  directions (dihedral, tail arm) and moves tail-volume floors
                   ▼
              constraints ──► 19-record catalog, verified numerically
                   │          (true NeuralFoil, AeroBuildup, dynamic modes)
                   ▼
              uncertainty ──► Monte-Carlo: geometry/CG/mass/density/temperature
                   │          dispersion; every sample re-trimmed (Newton on
                   │          alpha) and re-verified; per-constraint reliability
                   ▼
              plots + report ──► PNG figures + Markdown report + JSON manifest
```

## Module map

| module | role |
|---|---|
| `mission_spec.py` | pydantic mission requirements model |
| `nl_parser.py` | LLM mission parsing with rule-based fallback + parse trace |
| `parameterization.py` | `DesignVector`: bounded flat design vector, CST airfoil mode, `perturb()` |
| `analysis.py` | `analyze()`: mass closure, AeroBuildup, modes; `Dispersion` first-order uncertainty model |
| `polar_lut.py` | NeuralFoil→CasADi LUT compatibility layer (cached per airfoil) |
| `constraints.py` | `Margins` + 19-constraint catalog, symbolic registration and numeric evaluation |
| `optimizer.py` | Opti assembly, `behavior_on_failure="return_last"`, repair-by-reseeding |
| `agent_loop.py` | `DesignAgent`: parse → optimize → multi-start restarts → diagnose (audit trail) |
| `uncertainty.py` | `run_monte_carlo()` → `RobustnessReport` (p_success, per-constraint reliability) |
| `plots.py` | three-view geometry, constraint dashboard, robustness figure (headless) |
| `report.py` | Markdown design report |
| `cli.py` | command-line entry point |

## Key design decisions (the "why" for the thesis)

* **NeuralFoil's CasADi path is untrustworthy** (verified: CL 0.48 vs 1.14 for a
  NACA 2412 at α=2°, far beyond interpolation error). The optimizer therefore
  traces against linear-interpolant LUTs built from the *numeric* NF path, and
  every final number is re-verified with true numeric NeuralFoil. Symbolic speed,
  numeric truth.
* **Success is defined by the numeric constraint records, not by IPOPT's
  convergence flag.** A non-converged last iterate that passes all 19 checks is
  a valid design; a converged iterate that fails numeric verification is not.
* **Dynamic modes are numeric-only**, so the optimizer cannot see them. The
  repair loop handles them by reseeding flat problem directions — dihedral has
  ~zero symbolic gradient, so IPOPT leaves the seed untouched, making reseeding
  an effective in-loop lever for spiral stability.
* **Robustness is a first-class output.** A mass-minimal design sizes power
  exactly to the climb requirement, so `climb` and `static_margin_max` are
  typically the binding constraints under dispersion (measured: p_success ≈ 0.3
  for a nominal-optimal trainer). This motivates robustness-aware margins as
  future work.

## Python API

```python
from design_agent.agent_loop import AgentConfig, DesignAgent
from design_agent.uncertainty import run_monte_carlo

agent = DesignAgent(AgentConfig(frozen_airfoil="naca2412", restarts=2))
report = agent.design_from_text("Design a 4-seat piston trainer. ...")
print(report.result.mass_total_kg, report.success)

rob = run_monte_carlo(report.result.design, report.spec,
                      alphas=report.result.solver_stats.get("alphas"),
                      n_samples=32, seed=0)
print(rob.summary())
```

## Testing

```bash
pytest design_agent/tests/ -q          # full suite (~3-4 min)
DESIGN_AGENT_SKIP_E2E=1 pytest design_agent/tests/ -q   # skip slow e2e
```

## Known limitations

* First-order dispersion model (no Re/coefficient feedback on density scaling).
* Empty-mass model is a Cessna-172-class statistical surrogate — recalibrate
  `MASS_SURROGATE_COEFFS` for other aircraft classes.
* Takeoff roll uses a textbook surrogate; the library check is authoritative.
* Rule-based parser extracts fixed mission fields; unusual phrasings need the
  LLM path (`llm=` argument).
