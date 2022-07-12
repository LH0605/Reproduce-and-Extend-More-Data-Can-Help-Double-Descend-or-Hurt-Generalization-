from multiprocessing import Pool
from subprocess import Popen, PIPE

def run_linear(dim):
    print(f'dim: {dim}')
    attack = "fgm"
    cmd = f"nice -n 10 python3 linear.py --attack {attack} --ndim {dim} > linear_{attack}_{dim}d.log"
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    res = proc.communicate()
    if proc.returncode != 0:
        print("retcode =", proc.returncode)
        print("res =", res)
        print("stderr =", res[1])

if __name__ == "__main__":
    dims = [1, 2, 5, 7, 10]
    with Pool(5) as pool:
        pool.map(run_linear, dims)
    print("Finished!")
