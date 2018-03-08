from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import seaborn as sns
import statsmodels.api as sm

def coulombs_law_model(x, m):
    u0 = 4 * np.pi * 10**(-7)
    return u0*m*m / (x * x)


def coulombs_law_modified_model(x, G, numerator):
    return numerator / (np.pi * 4 * x ** 2 + G)

# Read in data from file
# df = pd.read_csv("10x10alt.csv", header=None)
# df.columns = ['z', 'force']
# df['z'] = df['z']
#
# # Perform curve-fits
# poptN, pcovN = optimize.curve_fit(coulombs_law_model, df.z.values,
#                                   df.force.values)  # Naive
# poptM, pcovM = optimize.curve_fit(coulombs_law_modified_model, df.z.values,
#                                   df.force.values)  # Modified
#
# m_naive = poptN[0]
#
# poptC, pcovC = optimize.curve_fit(modified_coloumbs_law_model(m_naive), df.z.values,
#                                   df.force.values)  # Coulomb's format
#
# # Make new arrays from the fits
# zValues = np.arange(1, 60, 0.1)
# yNaive = [coulombs_law_model(zz, *poptN) for zz in zValues]
# yMod = [coulombs_law_modified_model(zz, *poptM) for zz in zValues]
# actualCoulombs = modified_coloumbs_law_model(m_naive)  # Fetch the actual coulomb's function
# yCol = [actualCoulombs(zz, *poptC) for zz in zValues]
#
# # Plot the Naive graph and the original data
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(df.z, df.force, 'o', label='FEA')
# plt.plot(zValues, yNaive, label="Coloumb's Law")
# sns.despine()
# plt.xticks([0, 30, 60])
# plt.yticks([0, 15, 30])
# plt.xlabel("$z$")
# plt.ylabel("$\delta_{\mathrm{mag}}(z)$")
# plt.legend()
# setLimits(ax, 0.05, 0.05)
# addCustomTicks(ax, [60], ['60mm'], 'x')
# addCustomTicks(ax, [30], ['30N'], 'y')
# plt.tight_layout()
# plt.savefig('magSpringNaive.pdf', dpi=1200, bbox_inches='tight')
#
# # Plot the Modified graph and the original data
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(df.z, df.force, 'o', label='FEA')
# plt.plot(zValues, yMod, label="Coloumb's Law (modified)")
# sns.despine()
# plt.xticks([0, 30, 60])
# plt.yticks([0, 15, 30])
# plt.xlabel("$z$")
# plt.ylabel("$\delta_{\mathrm{mag}}(z)$")
# plt.legend()
# setLimits(ax, 0.05, 0.05)
#
# addCustomTicks(ax, [60], ['60mm'], 'x')
# addCustomTicks(ax, [30], ['30N'], 'y')
#
# plt.tight_layout()
# plt.savefig('magSpringMod.pdf', dpi=1200, bbox_inches='tight')
#
# # Plot naive, modified graph and the original data on a single set of axes
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(df.z, df.force, 'o', label='FEA')
# plt.plot(zValues, yNaive, '--', label="Coloumb's Law")
# plt.plot(zValues, yMod, label="Coloumb's Law (modified)")
# sns.despine()
# plt.xticks([0, 30, 60])
# plt.yticks([0, 15, 30])
# plt.xlabel("$z$")
# plt.ylabel("$\delta_{\mathrm{mag}}(z)$")
# plt.legend()
# setLimits(ax, 0.05, 0.05)
#
# addCustomTicks(ax, [60], ['60mm'], 'x')
# addCustomTicks(ax, [30], ['30N'], 'y')
#
# plt.tight_layout()
# plt.savefig('magSpringCombined.pdf', bbox_inches='tight')
#
# #locs, labels = plt.xticks()
# plt.show()
#
# u0 = 4 * np.pi * 10**(-7)
# m = m_naive
# #m = '%.2E' % Decimal(str(m))
# G = poptC[0]
#
# print('With naive')
# print('m: ' + str(poptN[0]))
# print('#####')
# print("With format F = (u0 * m^2 ) / (4 * pi * r^2 + G)")
# print("m: " + str(m))
# print("G: " + str(G))
# print()
#
# top = u0*m*m
# top = '%.3E' % Decimal(str(top))
# print(top)
#
# G = '%.3E' % Decimal(str(G))
# print(G)
