# Finish the running 10K study without restarting generation

Leave Slurm job **13121440** (SLICE request
`5afcfbc24dcd4679b9435afeb697d40f`) running. It is collecting valid answers using
its own source snapshot. Pulling fixes into the main checkout does not change
that running code. Do not edit its source snapshot, cached stages or environment.

The original worker can finish its existing primary-seed pipeline. If its legacy
reporting check fails after fitting, its completed answers, features and fitted
compressors remain usable. The follow-up below requires those three completed,
checksum-verified stages; it refuses incomplete or changed inputs.

From **compsci-login-01**, after pushing the fixes from your Mac:

```bash
cd /usr/xtmp/ms785/SLICE
git pull --rebase origin API
source .venv-native/bin/activate
mkdir -p logs

python -m slice_probe prepare-finalization \
  --job-dir /usr/xtmp/ms785/slice-service/jobs/5afcfbc24dcd4679b9435afeb697d40f \
  --out artifacts/finalize-numina-10k-002
```

This lightweight command snapshots the corrected follow-up code and request into
a new directory. It does no inference or training and submits nothing. It can run
while answer collection continues. Do not run preparation twice with the same
output; the prepared plan can be reused for a later, reconciled retry.

Submit the follow-up once, with a dependency on the existing job:

```bash
sbatch --dependency=afterany:13121440 --kill-on-invalid-dep=yes \
  scripts/finalize-native.sbatch \
  /usr/xtmp/ms785/SLICE/artifacts/finalize-numina-10k-002
```

`afterany` allows reuse of valid cached work even if the original job fails only
at reporting. The follow-up additionally checks scheduler absence and the source
worker lock, so it cannot run concurrently with that worker. If collection or
compressor fitting did not finish, it stops instead of starting fresh generation.

It verifies the saved stages, reuses any matching completed Ridge heads, fits
missing final heads for all three configured seeds, recalibrates newly fitted
heads, recomputes reports using canonical prompt-only features, and publishes
separate immutable versions. It never collects answers or retrains compressors.
The original stages and artifacts remain unchanged. A failed follow-up can reuse
its own fully completed stages on retry; incomplete stages restart from scratch.

The script requests **one A5000, 8 CPUs, 96 GB RAM and a seven-day maximum**.
That maximum is not a runtime estimate. This is an explicit standalone Slurm
follow-up: its resource use is separate from the controller ledger's reservation
for the original job. Record its numeric scheduler ID. If submission times out,
reconcile `slice-finalize` jobs before retrying; concurrent runs sharing the plan
are also blocked by a lock.

Inspect it with the returned ID:

```bash
sacct -j FOLLOWUP_JOB_ID --format=JobID,State,ExitCode,Elapsed
tail -n 60 logs/slice-finalize-FOLLOWUP_JOB_ID.out
cat artifacts/finalize-numina-10k-002/status.json
```

Require `COMPLETED`, `0:0`, and a final status of `completed`. `artifact_paths`
lists all three seeds under `/usr/xtmp/ms785/slice-critics`; the primary version
is `finalized-5afcfbc24dcd4679b9435afeb697d40f`, with `-seed43` and `-seed44`
companions. The summary records per-seed results and mean metrics, not an ensemble.
This follow-up does not convert the original job's controller record into a
three-head run; its own status, provenance and scheduler ID identify the additions.
