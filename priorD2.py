
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.stats import norm
from NormalDistribution import NormalDistribution 
from NormalDistribution import PolinomialDistribution



# X,Y=np.meshgrid(np.linspace(0.0,6.0,200),np.linspace(-1.0,5.0,200)) # np.linspace(0,2*np.pi,50))
# mu,sigma=0,1; #suppose that mux=muy=mu=0 and sigmax=sigmay=sigma
# mean = [0, 0]
# cov = [[1, 0], [0, 2]]  
# mean1 = [5, 0]
step_size = 30
def getMeanFromHistogram(values,counts):
	mean = 0
	for i in range(len(values)):
		mean += values[i]*counts[i]
	mean = mean / np.sum(counts)
	return mean
def getStdFromHistogram(x,y,mean):
	std = 0
	for i in range(len(x)):
		std += np.power(x[i]*y[i] - mean,2)
	std = np.sqrt(std/np.sum(y))
	return std
def getSecondaryDDistribution():
	x = np.array([230,125,63,31,15])
	y = np.array([1,27,47,38,30])
	x = x[::-1]
	y = y[::-1]
	# plt.plot(x, y, 'ro')
	# z1 = np.poly1d(np.polyfit(x, y, 2))
	mean = getMeanFromHistogram(x,y)
	std = getStdFromHistogram(x,y,mean)
	std = 55
	x = np.linspace(0, 500, 100)
	p = norm.pdf(x, mean, std)
	pMax = np.max(p)
	# plt.plot(x, p*47/pMax, 'k', linewidth=2)
	# print x
	# print p
	return NormalDistribution(mean,std,step_size)

	# x = np.linspace(1, 300)
	# print z1(x)
	# plt.plot(x, z1(x), 'g')
def getPrimaryDDistribution():
	# plt.axis([0, 6, 0, 20])
	##########
	x = np.array([230,125,63,31,15])
	y = np.array([23,31,35,35.4,37.4])
	x = x[::-1]
	y = y[::-1]

	# plt.plot(x, y, 'ro')
	# mean = getMeanFromHistogram(x,y)
	# std = getStdFromHistogram(x,y,mean)
	z = np.poly1d(np.polyfit(x, y, 2))
	####
	'''
	Z =np.polyint(z)
	x = np.linspace(15, 300)
	plt.plot(x, z(x), 'g')
	Z = Z/(Z(300))
	print Z(0)
	plt.plot(x, Z(x), 'g')
	'''
	###
	return PolinomialDistribution(z,step_size)
	# plt.plot(x, 20.0*z1(x)/(z2(x)+z1(x)), 'r')
	# plt.plot(x, 20.0*z2(x)/(z2(x)+z1(x)), 'b')
	# plt.show() 
# plotZunilD_p()

# pPD = plotPrimaryD_p()
# pSD = plotZunilD_p()
# print pSD.p(200)
# print pPD.p(200)
# print pSD.p(300)
# print pPD.p(300)
# print pSD.p(57)
# print pPD.p(57)

# plt.show() 

# exit()
# class PriorElipticity:
# 	def __init__(self,rMenor,theta):
# 		self.theta = theta
# 		self.rMenor = rMenor
# 		self.mean = np.array([rMenor,theta])
# 		stdX,stdY = 1.0/2,1/2.0
# 		# stdX,stdY = 1.0/(2*4.0),1/(2*3.1415926*4.0)
# 		self.cov = [ [stdX**2, 0], [0,stdY**2]]  
# 	def p(self,x):
# 		rMenor = 1+np.abs(1-self.rMenor)
# 		return 	multivariate_normal.pdf(x, mean=self.mean, cov=self.cov) +\
# 				multivariate_normal.pdf(x, mean=self.mean+(0,np.pi), cov=self.cov) +\
# 				multivariate_normal.pdf(x, mean=self.mean+(rMenor,-np.pi/2), cov=self.cov) +\
# 				multivariate_normal.pdf(x, mean=self.mean+(rMenor,np.pi/2), cov=self.cov)
# class PriorPrimary:
# 	def __init__(self):
# 		# self.theta = theta
# 		# self.rMenor = rMenor
# 		self.mean = 1.0 #np.array([rMenor,theta])
# 		# stdX,stdY = 1.0/2,1/2.0
# 		# stdX,stdY = 1.0/(2*4.0),1/(2*3.1415926*4.0)
# 		self.cov = 0.01 #[ [stdX**2, 0], [0,stdY**2]]  
# 	def p(self,x):
# 		z = x.T[1]
# 		zShape = z.shape
# 		z = z.reshape((zShape[0],zShape[1],1))

# 		return 	multivariate_normal.pdf(z, mean=self.mean, cov=self.cov)
				

# # prior = PriorPrimary() #(0.5,np.pi/2)
# prior = PriorElipticity(0.5,np.pi/2)
# a = prior.p(np.array((Y,X)).T)
# # print 
# # a = multivariate_normal.pdf(np.array((X,Y)).T, mean=mean, cov=cov)+multivariate_normal.pdf(np.array((X,Y)).T, mean=mean1, cov=cov)

# # X = np.linspace(-1,1,10)
# # Y = np.linspace(-1,1,10)
# # print X
# # x = np.linspace(0, 5, 10, endpoint=False)
# # Z = multivariate_normal.pdf(X,Y, mean=mean, cov=cov)


# # diagonal covariance
# # y = multivariate_normal.pdf(x, mean=2.5, cov=0.5);
# # x, y = np.random.multivariate_normal(mean, cov, 5000).T
# # plt.plot(x, y, 'x')
# # plt.axis('equal')
# # plt.show()
# # G=np.exp(-((X-mu)**2 + (Y-mu)**2)/2.0*sigma**2)
# # print G
# fig=plt.figure();
# ax=fig.add_subplot(111,projection='3d')
# surf=ax.plot_wireframe(X,Y,a)
# plt.xlabel('Minor axis')
# plt.ylabel('Theta')
# plt.title('Ellipticity prior for primary craters')
# # plt.zlabel('prior')
# plt.show()