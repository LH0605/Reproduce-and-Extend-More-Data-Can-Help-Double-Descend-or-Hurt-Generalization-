import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
# epsilons = [0, 1., 3., 4., 6., 8., 10., 12.] # poisson
# epsilons = [0, 1., 3., 4., 7., 8., 10., 12.] # poisson alt

# del index (4,5,6,13,14)
# epsilons = [0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0] # gaussian

# 3 regimes del index (4,5,6,13,14)
epsilons = [0, 0.1, 0.2, 0.3] # gaussian weak
# epsilons = [0.7, 0.8, 0.9] # gaussian medium
# epsilons = [1.0, 1.5, 2.0] # gaussian strong

test_losses = [[0.00313031,0.00291535,0.00281133,0.00253076,0.00237357,0.00225523
,0.00221489,0.0020347,0.00197454,0.00199823,0.00191329,0.00187712
,0.00178862,0.00175735,0.00178081,0.00162766,0.00166344,0.00171625
,0.00171039,0.00165449,0.00162746,0.00165135,0.0016024,0.0016079
,0.001604,0.00157018,0.00149777,0.00152731,0.00156874,0.0015349,]
,[0.01636444,0.00911234,0.0063334,0.00496681,0.00424727,0.003857
,0.00327337,0.00286593,0.00263911,0.00234617,0.00233384,0.0022243
,0.00188492,0.00185177,0.00167944,0.0016397,0.00147557,0.00143008
,0.00142947,0.00126526,0.00120788,0.00116011,0.00114043,0.00114385
,0.00106989,0.00101025,0.00095394,0.00089233,0.0008765,0.00091319]
,[0.06541536,0.05456132,0.04338063,0.0390014,0.03117482,0.02758771
,0.02284338,0.02040388,0.01683793,0.01507669,0.01381604,0.01193354
,0.01060577,0.0092709,0.00826527,0.0076908,0.00686507,0.00612837
,0.00536463,0.00512177,0.00488681,0.00410248,0.00402792,0.00398773
,0.003726,0.00312097,0.00332721,0.00298827,0.00280709,0.00283094]
,[0.0868453,0.08089539,0.07364348,0.06695969,0.05605862,0.04957446
,0.0478961,0.04280972,0.03809929,0.03568378,0.03254672,0.02920514
,0.02796917,0.02583899,0.02250365,0.02167744,0.02007634,0.01873602
,0.017548,0.0155607,0.01516635,0.01392896,0.01302265,0.01173884
,0.01206706,0.01092625,0.01035003,0.00886365,0.00884738,0.00867564]
,[0.16946802,0.17806799,0.16802474,0.1566166,0.14252436,0.13618701
,0.13296544,0.12788802,0.11415857,0.10724214,0.107254,0.1020537
,0.10367683,0.09675827,0.0969294,0.09474624,0.09813594,0.09115015
,0.09083876,0.08897825,0.09078649,0.08883895,0.0887499,0.08798534
,0.08784186,0.08655252,0.08466766,0.08566569,0.08598238,0.08585634]
,[0.18163996,0.15274297,0.1293256,0.11974441,0.11174394,0.10014512
,0.09722735,0.09690679,0.09554983,0.09566497,0.09442512,0.09557571
,0.0947528,0.09497805,0.09494677,0.09516182,0.09514635,0.0950936
,0.09504953,0.09530516,0.09503995,0.09520368,0.09521461,0.09514375
,0.09548916,0.09545504,0.09533018,0.09562863,0.09550911,0.09558323]
,[0.12717074,0.10322765,0.09816497,0.0974694,0.09729887,0.09723399
,0.09744787,0.09747287,0.09756668,0.09756036,0.09754236,0.09740912
,0.09749246,0.09761348,0.09752093,0.09764083,0.09759905,0.0976363
,0.09760606,0.09766673,0.097605,0.09751309,0.09760445,0.09754339
,0.09766877,0.0976126,0.09754841,0.09763296,0.09759496,0.09757623]
,[0.1014992,0.09810623,0.09808559,0.09818119,0.09829225,0.09827182
,0.09820497,0.09834911,0.09831464,0.09828129,0.09836509,0.09824927
,0.09828569,0.09829335,0.0983442,0.09823007,0.09827763,0.09833782
,0.09822233,0.09824693,0.09821735,0.09834847,0.09831653,0.09819487
,0.09820718,0.0982931,0.0982449,0.09817648,0.09825597,0.09830539]]

# for gaussian
# del(test_losses[4])
# del(test_losses[4])
# del(test_losses[4])
# del(test_losses[10])
# del(test_losses[10])

# for poisson
# del(test_losses[2])
# del(test_losses[4])

# for 3 regimes weak
del(test_losses[4])
del(test_losses[4])
del(test_losses[4])
del(test_losses[4])
del(test_losses[4])
del(test_losses[4])
del(test_losses[4])
del(test_losses[4])
del(test_losses[4])
del(test_losses[4])
del(test_losses[4])
# for 3 regimes medium
# del(test_losses[0])
# del(test_losses[0])
# del(test_losses[0])
# del(test_losses[0])
# del(test_losses[0])
# del(test_losses[0])
# del(test_losses[0])
# del(test_losses[3])
# del(test_losses[3])
# del(test_losses[3])
# del(test_losses[3])
# del(test_losses[3])

# for 3 regimes strong
# del(test_losses[0])
# del(test_losses[0])
# del(test_losses[0])
# del(test_losses[0])
# del(test_losses[0])
# del(test_losses[0])
# del(test_losses[0])
# del(test_losses[0])
# del(test_losses[0])
# del(test_losses[0])
# del(test_losses[3])
# del(test_losses[3])


if len(epsilons) != len(test_losses):
    print(len(epsilons), len(test_losses))
    raise Exception("should be the same, check epsilons")
TRAIN_SIZE = len(test_losses[0])
train_sizes = np.arange(1, TRAIN_SIZE+1)

plt.title("Linear Regression 1D Gaussian with FGM weak")
plt.xlabel("Size of Training Dataset")
plt.ylabel("Test Loss")
# same y for every poisson 10D
plt.ylim(0,0.2)
for i in range(len(epsilons)-1,0,-1):
    print('eps:', epsilons[i])
    ysmoothed = gaussian_filter1d(test_losses[i], sigma=1)
    plt.plot(train_sizes, ysmoothed, label=f"Ɛ = {epsilons[i]}")
plt.legend(loc="right")
plt.savefig(f"linreg_gaussian_fgm_1d_weak.png")
plt.clf()

"""
# 1
plt.title("Linear Regression Gaussian")
plt.xlabel("Size of Training Dataset")
plt.ylabel("Test Loss")
for i in [1,2,3]: # range(len(epsilons)):
    print('eps:', epsilons[i])
    plt.plot(train_sizes, test_losses[i], label=f"Ɛ = {epsilons[i]}")
plt.legend(loc="best")
plt.savefig(f"linreg_gaussian_fgsm_1d_weak.png")
plt.clf()

# 2
for i in [13, 14, 15, 16]:
    print('eps:', epsilons[i])
    plt.plot(train_sizes, test_losses[i], label=f"Ɛ = {epsilons[i]}")
plt.legend(loc="best")
plt.savefig(f"linreg_gaussian_fgsm_1d_strong.png")
"""
