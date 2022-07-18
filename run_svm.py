from multiprocessing import Pool
from subprocess import Popen, PIPE


def run_svm_dim(dim):
    print(f'dim: {dim}')
    attack = "fgm"
    cmd = f"nice -n 10 python3 svm.py --attack {attack} --ndim {dim} -c 0.1 > svm_{attack}_{dim}d_0.1.log"
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    res = proc.communicate()
    if proc.returncode != 0:
        print("retcode =", proc.returncode)
        print("res =", res)
        print("stderr =", res[1])

def run_svm_lambda(c):
    print(f'lambda: {c}')
    attack = "fgsm"
    cmd = f"nice -n 10 python3 svm.py --attack {attack} --ndim 2 -c {c} > svm_{attack}_2d_{c}.log"
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    res = proc.communicate()
    if proc.returncode != 0:
        print("retcode =", proc.returncode)
        print("res =", res)
        print("stderr =", res[1])

if __name__ == "__main__":
    dims = [2, 5, 7, 10]
    # cs = [0.001, 0.01, 1., 10.]
    with Pool(4) as pool:
        pool.map(run_svm_dim, dims)
        # pool.map(run_svm_lambda, cs)
    print("Finished!")
