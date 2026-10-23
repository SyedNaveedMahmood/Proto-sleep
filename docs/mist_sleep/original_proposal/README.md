# MIST-Sleep Original Proposal — Source Registry

This folder separates the original user's 13-page research proposal from later MIST-COVE explanation/auditing research.

## Sources

1. [Text mirror](MIST_SLEEP_ORIGINAL_PROPOSAL_TEXT.md): all 13 pages extracted from the original supplied PDF, with page boundaries, for code-agent parsing.
2. **Original binary PDF**: not yet committed; intended path `MIST-Sleep_Original_Research_Proposal.pdf`. It should be added **byte for byte** from the user's original attachment and not regenerated from the extracted text.

Original user PDF:
`MIST-Sleep_ Compute-Efficient Morphology-Invariant Transfer for Cross-Dataset EEG Sleep Staging.docx.pdf`

Expected source fingerprint:

- SHA-256: `4ab676da46b3ef8a49d275d336df5c251cf09cd5aaf98dc8e735cc6a2b1fcadf`
- Length: **201572** bytes
- Pages: **13**

Before committing the original PDF, verify:

```bash
sha256sum docs/mist_sleep/original_proposal/MIST-Sleep_Original_Research_Proposal.pdf
pdfinfo docs/mist_sleep/original_proposal/MIST-Sleep_Original_Research_Proposal.pdf | grep Pages
```

If the PDF is unavailable in the agent's filesystem, use the tracked text mirror for the methods specification and record the missing-binary status; do not invent a source path, substitute a different document, or block independent engineering.

The proposal's primary method is **Morphology-Preserving Invariance (MPI)**, shared morphology prototypes, and compute-efficient **cross-dataset sleep staging**. Exact MIST-WIC finite-circuit audits are later parallel research, not a substitute for testing this proposal.

Keep source text verbatim; any editorial or experimental revision belongs in a new, clearly named artifact.
