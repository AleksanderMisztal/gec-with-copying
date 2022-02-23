import subprocess

def run_errant(in_dir):
  get_ref = f"errant_parallel -orig {in_dir}/orig.txt -cor {in_dir}/corr.txt -out {in_dir}/ref.m2"
  get_hyp = f"errant_parallel -orig {in_dir}/orig.txt -cor {in_dir}/pred.txt -out {in_dir}/hyp.m2"
  compare = f"errant_compare -hyp {in_dir}/hyp.m2 -ref {in_dir}/ref.m2"

  subprocess.run(get_ref)
  subprocess.run(get_hyp)
  subprocess.run(compare)

if __name__ == "__main__":
  run_errant('./out')