from scipy.stats import norm
import numpy as np
class NormalDistribution():
	def __init__(self,mu,std,interval):
		self.mu = mu
		self.std = std
		self.interval = interval
		# print mu
	def p(self,x):
		xl = x - self.interval
		xr = x + self.interval
		yl = norm.cdf(xl, self.mu, self.std)
		yr = norm.cdf(xr, self.mu, self.std)
		# print yl
		# print yr
		return yr-yl

class PolinomialDistribution():
	def __init__(self,z,interval):
		Z =np.polyint(z)
		# x = np.linspace(15, 300)
		# plt.plot(x, z(x), 'g')
		Z = Z/(Z(300))
		self.Z = Z
		self.interval = interval

	def p(self,x):
		xl = x - self.interval
		xr = x + self.interval
		yl = self.Z(xl)
		yr = self.Z(xr)
		# norm.cdf(xr, self.mu, self.std)
		return yr-yl

	# plt.plot(x, Z(x), 'g')

