# Notebook archive

The project originally lived in two development notebooks:

- a 2,000+ line one-cell notebook;
- a multi-cell experiment notebook used for debugging and follow-up ablations.

They are intentionally **not** the canonical implementation in this repository. The same core implementation has been split into `src/protosleep/` modules so checkpoint reuse, test-set access, timing, and experiment provenance are easier to audit.

If an archival notebook is added later, it should import the package rather than duplicate the full implementation.
