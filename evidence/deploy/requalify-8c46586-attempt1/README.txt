Attempt 1 stopped at 'release.sh promote blue': rsb-blue is not ready (state 'absent').
Cause: the previous rsb-blue worker pod rsb-blue-sim-general-worker-krttz had been stuck Terminating for 9 days (its container vanished when Docker Desktop restarted), so the RayCluster never reported ready. rsb-green, from the same image, reached ready. No worker logged an affinity error.
Fix: kubectl delete pod --force --grace-period=0 (project kind cluster); rsb-blue became ready (2/2 workers) within 21 s; requalify rerun from the start in ../requalify-8c46586/.
