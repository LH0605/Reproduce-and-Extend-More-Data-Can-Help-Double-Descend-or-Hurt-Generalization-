from multiprocessing import Pool
from subprocess import Popen, PIPE


def run_linreg_dim(dim):
    print(f'dim: {dim}')
    attack = "pgd"
    cmd = f"nice -n 10 python3 linreg_md.py --gaussian --attack {attack} --ndim {dim} > linreg_gaussian_{attack}_{dim}d.log"
    # cmd = f"nice -n 10 python3 linreg_md.py --attack {attack} --ndim {dim} > linreg_poisson_{attack}_{dim}d.log"
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    res = proc.communicate()
    if proc.returncode != 0:
        print("retcode =", proc.returncode)
        print("res =", res)
        print("stderr =", res[1])

def run_linreg_l2(l2):
    dim = 1
    attack = "fgsm"
    type = "gaussian"
    
    print(f'l2: {l2}')
    if type == "gaussian":
        cmd = f"nice -n 10 python3 linreg_md.py --gaussian --attack {attack} --ndim {dim} --l2 {l2} > linreg_gaussian_{attack}_{dim}d_{l2}.log"
    elif type == "poisson":
        cmd = f"nice -n 10 python3 linreg_md.py --attack {attack} --ndim {dim} --l2 {l2} > linreg_poisson_{attack}_{dim}d_{l2}.log"
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    res = proc.communicate()
    if proc.returncode != 0:
        print("retcode =", proc.returncode)
        print("res =", res)
        print("stderr =", res[1])

if __name__ == "__main__":
    # dims = [1, 2, 5, 7, 10]
    l2s = [1, 0.1, 0.01, 0.05, 0.5, 5]
    with Pool(5) as pool:
        # pool.map(run_linreg_dim, dims)
        pool.map(run_linreg_l2, l2s)
    print("Finished!")
