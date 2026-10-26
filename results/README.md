# Result evidence

`paper_results.json` is the aggregate record behind the paper's main table.
The values were transcribed from six completed A100/FP16 GPU runs. The file
records their measurement scope.

The final per-sample JSON outputs and adversarial tensors are not included in
this repository. Consequently, `scripts/validate_artifact.py` checks the
internal consistency of the released aggregate evidence but does not replace a
GPU reproduction. A new run should retain its raw `results.json`,
`summary.json`, attack tensors, environment snapshot, and scheduler logs in
large artifact storage.

Verify the frozen inputs and result record with:

```bash
sha256sum -c results/SHA256SUMS
python scripts/validate_artifact.py
```
