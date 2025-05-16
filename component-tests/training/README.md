# Training

Establish the batch size and number of epochs which yield good utilization and time-to-solution.
We do not test solution quality here.

## Submitting

The `run_test.py` script runs within a batch job.

Submit it using the scripts in the `scripts` directory,
which take arguments.

For example, start Polaris with

```
qsub -v args="--training-size 3192" scripts/run-polaris.sh
