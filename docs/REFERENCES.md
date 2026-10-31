# Sources to verify and cite

Course source: supplied PDF in `docs/course/`. The approved equations and project title were supplied by the group; the report must find and verify appropriate ecological context and any published model it builds on. Do not cite an unverified AI-generated bibliography entry.

Implementation primary sources (documentation checked during scaffold preparation):

- [SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html): RK45, dense output, terminal events, tolerances and return fields.
- [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html): inverse ODE-fit baseline; read before implementation.
- [JAX automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html) and [installation](https://docs.jax.dev/en/latest/installation.html): `grad`/`vmap` used by the PINN engine; CPU installation.
- [GitHub protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches): review/status gates and plan availability.
- [actions/checkout](https://github.com/actions/checkout) and [actions/setup-python](https://github.com/actions/setup-python): CI configuration.

**WP01 verified entries (24 October 2026)**, all in `report/references.bib`:

| Key | Verified against |
|---|---|
| `courchamp2008allee` | Masaryk University library catalogue record (OUP, Oxford, 2008, ISBN 978-0-19-857030-1) |
| `stephens1999consequences` | Article PDF (Trends Ecol. Evol. 14(10):401-405); also defines the Allee effect at the fitness-component level |
| `dennis1989allee` | Reference list of Kramer et al. (2009) Popul. Ecol. 51:341-354 and arXiv:1405.1686 (Nat. Resour. Model. 3:481-538) |
| `hutchings2004collapses` | UEA ePrints record (BioScience 54(4):297-309) |
| `strogatz2015nonlinear` | Kyushu University library catalogue (Westview Press, Boulder, 2015, ISBN 9780813349107) |
| `brauer2012mathematical` | Bibliotheque nationale de France catalogue (2nd ed., Springer, Texts in Applied Mathematics 40, 2012) |
| `brent1973algorithms`, `scipy_brentq` | Reference section of the installed `scipy.optimize.brentq` docstring and the SciPy documentation page |

WP01/WP02 find a small number of credible primary/textbook sources on strong Allee effects and harvesting. WP04 verifies the original PINN method source and records the particular loss/architecture used. WP05 cites any inference/identifiability method borrowed rather than derived here. Put **verified** entries in `report/references.bib`, cite them where used, and distinguish this group's derivations from adapted material.

## WP03 numerical references checked 28 October 2026

- SciPy [solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html): RK45 5(4), DOP853 and terminal events.
- SciPy [quad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html): QUADPACK integration/error-estimate semantics.
- SciPy [bounded minimisation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize_scalar-bounded.html): independent maxima and its x tolerance.

The mathematical derivations are this project's calculations under the stated models, not empirical ecological validation. Bibliography entries are in report/references.bib.
