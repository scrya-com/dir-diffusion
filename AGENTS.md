# Agent guidelines for dir-diffusion

## Name every GPU process

GPU jobs here run for minutes to hours, often several at once on two cards. nvtop, `ps` and `htop` cut long command lines short, so a job launched from a long path is unreadable. Every script that runs on a GPU, including throwaway scripts in a scratch folder, sets a short process title as its first statement after the imports:

```python
from setproctitle import setproctitle   # in the miniconda env; pip install setproctitle elsewhere

setproctitle("dwd:sgd lr=3e-5 run_v1")   # <project>:<phase> <the arguments that tell runs apart>
```

- **Format:** `project:phase` first (`dir:train`, `dwd:collect`, `nf:stage2`), then the few arguments that tell this run apart from the others. Keep it under about 60 characters.
- **Scripts with phases or subcommands:** set the title after parsing the arguments, so it names the phase.
- **Launching:** `cd` into the script's folder and run it by its bare name (`python3 -u ft_lastk.py …`), not by a long absolute path.
- **One GPU job at a time per card:** follow `lot/gpu_guard.py` and add new GPU scripts to its `GPU_SCRIPTS`.

## Mark positive results with an emoji

Reports here are dense tables of mixed results: eval metrics against a gate, A/B comparisons, held-identity counts. In every report and status update, including a one-line progress note, mark each positive finding with an emoji so it stands out: ✅ for a gate passed or a check that held, 📈 for a metric that improved, 🎉 for a milestone. Leave negative and neutral findings plain, and use the emoji sparingly so they keep their meaning.

```text
soft_mae 0.122 (gate < 0.10)       <- missed: plain
translucent soft_mae 0.086 ✅       <- under the gate
held 15/15 vs 10/15 📈              <- improved
```
