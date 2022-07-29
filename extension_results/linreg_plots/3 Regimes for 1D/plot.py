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

test_losses = [[3.04015719e-01,3.92097986e-02,8.74129925e-03,5.30300955e-03
,3.69659093e-03,2.26195353e-03,1.79074829e-03,1.81961389e-03
,1.57523842e-03,1.47861302e-03,1.08315687e-03,1.15589366e-03
,9.19430121e-04,8.83780306e-04,7.80238104e-04,6.84556971e-04
,6.60720718e-04,6.84295149e-04,5.58569601e-04,6.21615682e-04
,5.28113425e-04,5.28610063e-04,4.46158535e-04,4.46648567e-04
,5.12701711e-04,4.20819988e-04,4.37388313e-04,3.58392779e-04
,3.54763665e-04,2.98303186e-04]
,[5.28577743e-02,1.38398784e-02,4.86111413e-03,4.40160650e-03
,2.55376932e-03,2.27606916e-03,2.06679469e-03,1.65904060e-03
,1.32144206e-03,1.23422995e-03,1.07674197e-03,1.09449062e-03
,9.18796719e-04,9.24061508e-04,8.54857097e-04,8.67096687e-04
,7.86022672e-04,7.37378578e-04,6.94506423e-04,5.97772090e-04
,5.34556404e-04,6.30104157e-04,5.53638602e-04,5.52170276e-04
,4.90725943e-04,4.71806965e-04,4.57635300e-04,4.56629333e-04
,4.32585233e-04,4.38091152e-04]
,[3.02326530e-02,8.30380453e-03,5.71342302e-03,3.39318431e-03
,2.67264539e-03,2.66911374e-03,2.01066288e-03,1.75779000e-03
,1.62411979e-03,1.81664792e-03,1.57804710e-03,1.38084161e-03
,1.21432270e-03,1.16228992e-03,1.12455838e-03,1.17037977e-03
,1.24297658e-03,9.64528231e-04,1.09365829e-03,8.34321130e-04
,8.59316483e-04,8.69854601e-04,8.23032789e-04,7.15690291e-04
,9.21486846e-04,7.36468162e-04,6.78359068e-04,6.76699201e-04
,7.47851696e-04,6.25127492e-04]
,[1.96970483e-02,9.67403655e-03,4.21246127e-03,3.86580658e-03
,3.17332145e-03,2.88773817e-03,2.66019464e-03,2.62330284e-03
,2.21222409e-03,2.09310764e-03,2.08469013e-03,2.33758281e-03
,1.92342712e-03,1.38730197e-03,1.79153709e-03,1.50024873e-03
,1.54203385e-03,1.59571691e-03,1.45647294e-03,1.50252652e-03
,1.45264277e-03,1.28846346e-03,1.28106536e-03,1.29982621e-03
,1.30508381e-03,1.21084237e-03,1.16501325e-03,1.04760991e-03
,1.25339480e-03,1.14710839e-03]
,[1.39695338e-02,7.11337352e-03,4.93965074e-03,4.04423985e-03
,4.08733489e-03,3.30610652e-03,3.31915469e-03,2.94660500e-03
,2.84231699e-03,2.93067703e-03,2.90006695e-03,2.62774599e-03
,2.79115323e-03,2.34641561e-03,2.57101962e-03,2.75699621e-03
,2.28755996e-03,2.40085762e-03,2.20260033e-03,2.21593438e-03
,2.23856188e-03,2.03030143e-03,2.08170965e-03,2.16186473e-03
,2.08757700e-03,1.81926836e-03,1.88406194e-03,2.05160039e-03
,1.73662816e-03,1.92908533e-03]
,[1.14701673e-02,7.14556660e-03,5.27132623e-03,4.78462433e-03
,4.42371350e-03,4.42998096e-03,4.18798122e-03,3.88820571e-03
,3.58692715e-03,4.05549018e-03,3.59844182e-03,3.91354883e-03
,3.28100038e-03,3.26869446e-03,3.39321932e-03,3.55152038e-03
,3.38730640e-03,3.53033319e-03,3.21253492e-03,3.13566107e-03
,3.01818310e-03,3.11119105e-03,3.28946467e-03,3.17732162e-03
,3.08003113e-03,3.05184887e-03,3.22148305e-03,2.64601329e-03
,2.99812918e-03,2.74746252e-03]
,[1.00908388e-02,6.42827634e-03,5.43577267e-03,5.33662337e-03
,4.62560969e-03,4.95971945e-03,4.97847130e-03,4.81292743e-03
,5.26923828e-03,4.70704045e-03,4.54099395e-03,5.00511379e-03
,4.58872573e-03,4.45667824e-03,4.67612773e-03,4.36473424e-03
,4.44585123e-03,4.42395605e-03,4.41589754e-03,4.40329704e-03
,4.12266393e-03,4.49371162e-03,4.00505507e-03,4.18325232e-03
,4.58996222e-03,4.14610298e-03,4.31887488e-03,4.33960132e-03
,4.16612908e-03,4.14243315e-03]
,[9.48233011e-03,6.25078874e-03,6.60083307e-03,6.00905889e-03
,6.06194497e-03,5.89494142e-03,6.18093928e-03,5.91039204e-03
,5.89384360e-03,5.64123113e-03,5.67013220e-03,5.67842093e-03
,5.59427838e-03,6.02756949e-03,5.65198546e-03,5.92426547e-03
,5.85537777e-03,5.38713771e-03,6.00154738e-03,5.49146453e-03
,5.69368711e-03,5.49613663e-03,5.41675368e-03,5.87957967e-03
,5.76007120e-03,5.36775281e-03,5.72588312e-03,5.59781282e-03
,5.48039449e-03,5.54232289e-03]
,[8.86890988e-03,7.04693162e-03,6.64440415e-03,6.57758478e-03
,6.66068423e-03,6.47520692e-03,6.29526842e-03,6.62215706e-03
,5.96057832e-03,6.77115015e-03,6.51327231e-03,6.73280469e-03
,6.99008949e-03,6.79661941e-03,6.98610681e-03,6.62612535e-03
,6.69641864e-03,6.88334916e-03,6.88215648e-03,6.65249447e-03
,6.79026393e-03,7.14170911e-03,6.75798823e-03,7.06473070e-03
,6.95380403e-03,7.07489378e-03,6.86608135e-03,6.60619214e-03
,6.99489645e-03,7.05344502e-03]
,[8.35533071e-03,6.97410905e-03,7.25296497e-03,7.35482558e-03
,7.17965250e-03,7.10373906e-03,7.55078935e-03,7.81687869e-03
,8.00586230e-03,7.60477112e-03,7.41782588e-03,7.72012544e-03
,7.56972987e-03,7.80906472e-03,7.52315676e-03,7.93716043e-03
,7.78558316e-03,7.71236155e-03,7.99867321e-03,7.86448553e-03
,7.92903563e-03,7.78865368e-03,7.99609629e-03,8.04127006e-03
,8.19773796e-03,8.00592329e-03,7.99985111e-03,7.92341646e-03
,8.23680499e-03,8.16572358e-03]
,[8.12067343e-03,7.66443208e-03,7.71670605e-03,7.54305575e-03
,7.51970495e-03,8.06774840e-03,8.15891552e-03,8.29722022e-03
,8.14053301e-03,8.22551921e-03,8.11106621e-03,8.32351032e-03
,8.46403252e-03,8.46578613e-03,8.45584540e-03,8.60350162e-03
,8.56060532e-03,8.47243836e-03,8.58094262e-03,8.75766704e-03
,8.83862325e-03,8.67084164e-03,8.68388041e-03,8.77692729e-03
,8.79078419e-03,8.71304275e-03,8.86927676e-03,8.80700946e-03
,9.07328526e-03,8.89348051e-03]
,[8.86228516e-03,8.82358237e-03,9.16549202e-03,9.48026180e-03
,9.45051810e-03,9.43413491e-03,9.83090353e-03,9.69853514e-03
,9.73683992e-03,9.78438905e-03,9.81951769e-03,9.81196615e-03
,9.78354087e-03,9.88038501e-03,9.74753087e-03,9.88156206e-03
,9.88239057e-03,9.88832848e-03,9.88775316e-03,9.88713611e-03
,9.76671888e-03,9.90124149e-03,9.86200284e-03,9.92811602e-03
,9.97423123e-03,9.86170348e-03,9.95830644e-03,9.94429738e-03
,9.87267157e-03,9.88484425e-03]
,[9.58939844e-03,9.65983304e-03,9.82295423e-03,9.96633182e-03
,9.87329651e-03,9.85946248e-03,9.92837690e-03,9.90226340e-03
,9.96454866e-03,9.95584367e-03,9.98256434e-03,9.95873588e-03
,1.00022808e-02,9.97174565e-03,9.95661654e-03,9.99218006e-03
,9.93671389e-03,9.92427117e-03,9.97636718e-03,9.99106126e-03
,9.94858557e-03,9.96078233e-03,9.98811994e-03,1.00659739e-02
,9.97569023e-03,1.00325561e-02,9.99073269e-03,1.00279303e-02
,9.93959351e-03,1.00482951e-02]
,[9.64141752e-03,9.84613829e-03,9.96375390e-03,9.96533774e-03
,9.99351723e-03,9.96227154e-03,9.98834099e-03,9.97829097e-03
,9.91811788e-03,1.00483039e-02,9.99599119e-03,9.99854314e-03
,9.94681388e-03,9.99889267e-03,9.91158191e-03,1.00289674e-02
,1.00450162e-02,1.00443017e-02,1.00129056e-02,1.00190338e-02
,1.00324245e-02,1.00493581e-02,1.00038433e-02,1.00373941e-02
,9.96686897e-03,9.98186473e-03,9.96701829e-03,1.00110390e-02
,1.00377719e-02,1.00209038e-02]
,[9.92573814e-03,9.99582250e-03,1.00150665e-02,1.00341091e-02
,1.00052063e-02,9.98761026e-03,1.00110288e-02,1.00065524e-02
,9.96783347e-03,1.00189658e-02,9.97530779e-03,9.99395946e-03
,1.00000874e-02,9.98764541e-03,1.00293981e-02,9.99725423e-03
,9.95498298e-03,9.99225953e-03,9.99612129e-03,1.00749461e-02
,9.95683496e-03,1.00267986e-02,9.98104588e-03,1.00545787e-02
,9.97867906e-03,1.00416874e-02,1.00467018e-02,1.00165386e-02
,9.97767244e-03,1.00079738e-02]]

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

plt.title("Linear Regression 1D Gaussian with PGD weak")
plt.xlabel("Size of Training Dataset")
plt.ylabel("Test Loss")
# same y for every poisson 10D
# plt.ylim(0,0.2)
for i in range(len(epsilons)):
    print('eps:', epsilons[i])
    ysmoothed = gaussian_filter1d(test_losses[i], sigma=1)
    plt.plot(train_sizes, ysmoothed, label=f"Ɛ = {epsilons[i]}")
plt.legend(loc="right")
plt.savefig(f"linreg_gaussian_pgd_1d_weak.png")
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
