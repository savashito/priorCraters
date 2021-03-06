import matplotlib.pyplot as plt
import numpy as np
from NormalDistribution import NormalDistribution
from scipy.stats import norm

# rng = np.random.RandomState(10)  # deterministic random data
# a = np.hstack((rng.normal(size=1000),
#                rng.normal(loc=5, scale=2, size=1000)))
# print a
# print a.shape
# plt.hist(a, bins='auto')  # plt.hist passes it's arguments to np.histogram
# plt.title("Histogram with 'auto' bins")
# plt.show()

# exit()
f = open('marscraters_1.csv', 'r')
# returns a probability distribution for Rm p(primary|Rm)
interval = 0.01


def getSecondaryRmDistribution():
	i = 0
	array_diameter_major = [] # np.array([]);
	data = np.array([1.0, 1.1927, 1.03152, 1.23834, 1.37319, 1.21848])
	# print array_diameter_major
	# hist, bin_edges = np.histogram(array_diameter_major, bins=40, density=True)
	# Fit a normal distribution to the data:
	mu, std = norm.fit(data)
	return NormalDistribution(mu,std,interval)

	# print hist

def getPrimaryRmDistribution():
	i = 0
	array_diameter_major = [] # np.array([]);
	for line in f.readlines():
		if i>0:
			words = line.split(',')
			# print words[1]
			# print words[2]
			diameter = float(words[5])
			dMajor = float(words[6])

			dMajorRatio = dMajor/diameter
			# np.append(array_diameter_major,dMajorRatio)
			array_diameter_major.append(dMajorRatio)
			# print dMajorRatio
			# print words[7]
		i = i+1
	array_diameter_major = np.array(array_diameter_major)
	# print array_diameter_major
	# hist, bin_edges = np.histogram(array_diameter_major, bins=40, density=True)
	# Fit a normal distribution to the data:
	mu, std = norm.fit(array_diameter_major)
	# print hist
	# print bin_edges
	# plt.plot(bin_edges[1:],hist,'o')

	###
	# xmin, xmax = plt.xlim()
	# x = np.linspace(xmin, xmax, 100)
	# p = norm.pdf(x, mu, std)
	# plt.plot(x, p, 'k', linewidth=2)
	# plt.hist(array_diameter_major, bins='auto') 
	# plt.title("Histogram of primary craters for r_m parameter")
	# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
	# plt.title(title)
	# plt.show()
	return NormalDistribution(mu,std,interval)
# pRmP = getPrimaryRmDistribution(f)
# pRmS = getSecondaryRmDistribution()
# print "Primary"
# print pRmP.p(1.05)
# print pRmP.p(0.8)
# print pRmP.p(1.2)
# print "secondary"
# print pRmS.p(1.05)
# print pRmS.p(0.8)
# print pRmS.p(1.2)
# exit()


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.stats import multivariate_normal

# X,Y=np.meshgrid(np.linspace(0.0,6.0,200),np.linspace(-1.0,5.0,200)) # np.linspace(0,2*np.pi,50))
# # mu,sigma=0,1; #suppose that mux=muy=mu=0 and sigmax=sigmay=sigma
# # mean = [0, 0]
# # cov = [[1, 0], [0, 2]]  
# # mean1 = [5, 0]
# def plotZunilD_p():
# 	x = np.array([230,125,63,31,15])
# 	y = np.array([1,27,47,38,30])
# 	x = x[::-1]
# 	y = y[::-1]
# 	plt.plot(x, y, 'ro')
# 	z1 = np.poly1d(np.polyfit(x, y, 2))

# 	# print z(x)
# 	x = np.linspace(15, 300)
# 	print z1(x)
# 	plt.plot(x, z1(x), 'g')
# 	# plt.axis([0, 6, 0, 20])
# 	##########
# 	x = np.array([230,125,63,31,15])
# 	y = np.array([23,31,35,35.4,37.4])
# 	x = x[::-1]
# 	y = y[::-1]

# 	plt.plot(x, y, 'ro')
# 	z2 = np.poly1d(np.polyfit(x, y, 2))
# 	x = np.linspace(15, 300)
# 	plt.plot(x, z2(x), 'g')
# 	plt.plot(x, 20.0*z1(x)/(z2(x)+z1(x)), 'r')
# 	plt.plot(x, 20.0*z2(x)/(z2(x)+z1(x)), 'b')
# 	plt.show()
# plotZunilD_p()
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

'''
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
import numpy as np
class PriorElipticity:
	def __init__(self,rMenor,theta):
		self.theta = theta
		self.rMenor = rMenor
		self.mean = np.array([rMenor,theta])
		stdX,stdY = 1.0/2,1/2.0
		# stdX,stdY = 1.0/(2*4.0),1/(2*3.1415926*4.0)
		self.cov = [ [stdX**2, 0], [0,stdY**2]]  
	def p(self,x):
		rMenor = 1+np.abs(1-self.rMenor)
		return 	multivariate_normal.pdf(x, mean=self.mean, cov=self.cov) +\
				multivariate_normal.pdf(x, mean=self.mean+(0,np.pi), cov=self.cov) +\
				multivariate_normal.pdf(x, mean=self.mean+(rMenor,-np.pi/2), cov=self.cov) +\
				multivariate_normal.pdf(x, mean=self.mean+(rMenor,np.pi/2), cov=self.cov)

# class PriorPrimary:
# 	def __init__(self):
# 		# self.theta = theta
# 		# self.rMenor = rMenor
# 		self.mean = 1.0 #np.array([rMenor,theta])
# 		# stdX,stdY = 1.0/2,1/2.0
# 		# stdX,stdY = 1.0/(2*4.0),1/(2*3.1415926*4.0)
# 		self.cov = 0.01 #[ [stdX**2, 0], [0,stdY**2]]  
# 	def p(self,x):
# 		z = x.T[0]
# 		zShape = z.shape
# 		print zShape
# 		# z = z.reshape((zShape[0],zShape[1],1))
# 		# z = z.reshape((zShape[0],zShape[1],1))

# 		return 	multivariate_normal.pdf(z, mean=self.mean, cov=self.cov)

class PriorPrimary:
	def __init__(self):
		# self.theta = theta
		# self.rMenor = rMenor
		self.mean = 1.0 #np.array([rMenor,theta])
		# stdX,stdY = 1.0/2,1/2.0
		# stdX,stdY = 1.0/(2*4.0),1/(2*3.1415926*4.0)
		self.cov = 0.01 #[ [stdX**2, 0], [0,stdY**2]]  
	def p(self,x):
		z = x[0]
		# z = x[1]
		# print z
		# print x.T[0].shape
		# print z.shape
		# print z.shape
		# zShape = z.shape
		# z = z.reshape((zShape[0],zShape[1],1))
		z,y = np.meshgrid(multivariate_normal.pdf(z, mean=self.mean, cov=self.cov),x.T[0])
		return 	z


# primary = PriorElipticity(0.8,0.5)
primary = PriorPrimary()
# plot a 3D wireframe like in the example mplot3d/wire3d_demo
X = np.arange(0, 5, 0.1)
Y = np.arange(0, 2*3.141592, 0.25)
X, Y = np.meshgrid(X, Y)
# print X.shape
# print Y.shape
Z = primary.p(X)
# Z = primary.p(np.array([X,Y]))
# print X
# print Z
# print Z.shape
# exit()
fig = plt.figure(figsize=plt.figaspect(0.5))

#===============
#  First subplot
#===============
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')
print X.shape
print Y.shape
print Z.shape
# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)

ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
plt.show()
'''