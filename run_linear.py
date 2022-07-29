from multiprocessing import Pool
from subprocess import Popen, PIPE

def run_linear_dim(dim):
    print(f'dim: {dim}')
    attack = "fgm"
    cmd = f"nice -n 10 python3 linear.py --attack {attack} --ndim {dim} > linear_{attack}_{dim}d.log"
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    res = proc.communicate()
    if proc.returncode != 0:
        print("retcode =", proc.returncode)
        print("res =", res)
        print("stderr =", res[1])

def run_linear_l2(l2):
    print(f'l2: {l2}')
    attack = "fgsm"
    cmd = f"nice -n 10 python3 linear.py --attack {attack} --ndim 1 --l2 {l2} > linear_{attack}_1d_{l2}.log"
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    res = proc.communicate()
    if proc.returncode != 0:
        print("retcode =", proc.returncode)
        print("res =", res)
        print("stderr =", res[1])

if __name__ == "__main__":
    dims = [1, 2, 5, 7, 10]
    # l2s = [1, 0.1, 0.01, 0.05, 0.5, 5]
    with Pool(5) as pool:
        pool.map(run_linear_dim, dims)
        # pool.map(run_linear_l2, l2s)
    print("Finished!")
