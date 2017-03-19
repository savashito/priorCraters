import numpy as np
from NormalDistribution import NormalDistribution 
from scipy.stats import norm

interval = 0.005
def getSecondaryHsDistribution():
	i = 0
	array_diameter_major = [] # np.array([]);
	data = np.array([0.345261207406, 0.176531871675, 0.271045526554, 0.206543863779, 0.191287525743, 0.158548824052])
	
	# print array_diameter_major
	# hist, bin_edges = np.histogram(array_diameter_major, bins=40, density=True)
	# Fit a normal distribution to the data:
	mu, std = norm.fit(data)
	# print "stdSecond  "+str(std)
	# print "muSecond  "+str(mu)

	return NormalDistribution(mu,std,interval)

def getPrimaryHsDistribution():
	i = 0
	array_diameter_major = [] # np.array([]);
	data = np.array([0.345261207406, 0.176531871675, 0.271045526554, 0.206543863779, 0.191287525743, 0.158548824052])

	d1   = np.array([502.60, 676.82, 920.99])
	h_s1 = np.array([3.96,10.01,34.91])

	d2   = np.array([571.27,743.04,908.50])
	h_s2 = np.array([12.40, 10.20, 33.34])

	d3 = np.array([523.65,735.33,895.36])
	h_s3 = np.array([18.23, 11.38, 33.35])

	data1 = h_s1/d1
	data2 = h_s2/d2
	data3 = h_s3/d3
	data = np.append(data1,data2)
	data = np.append(data,data3)

	# print data
	
	mu, std = norm.fit(data)
	# print "stdFirst  "+str(std)
	# print "muFirst  "+str(mu)
	# print data1
	# print data2
	# print data3

	# print array_diameter_major
	# hist, bin_edges = np.histogram(array_diameter_major, bins=40, density=True)
	# Fit a normal distribution to the data:
	# mu, std = norm.fit(data)
	return NormalDistribution(mu,std,interval)
# pSHs = getSecondaryHsDistribution()
# pPHs = getPrimaryHsDistribution()
# print pSHs.p(0)
# print pPHs.p(0)
# print pSHs.p(0.1)
# print pPHs.p(0.1)