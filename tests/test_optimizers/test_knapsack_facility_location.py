import scipy
import numpy

try:
	import cupy
except:
	import numpy as cupy

from apricot import FacilityLocationSelection
from apricot.optimizers import *

from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances


from nose.tools import assert_less_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

#	print("[" + ", ".join(map(str, model.ranking)) + "]")
#	print("[" + ", ".join([str(round(gain, 4)) for gain in model.gains]) + "]")

rng = numpy.random.RandomState(0)

digits_data = load_digits()
X_digits = digits_data.data[:250]
X_digits_costs = numpy.abs(rng.randn(250))

X_digits_sparse = scipy.sparse.csr_matrix(X_digits)
X_digits_cupy = cupy.array(X_digits)

digits_ranking = [96, 179, 219, 216, 26, 56, 206, 54, 225, 187, 163, 79, 192, 
	17, 35, 8, 169, 64, 242, 69, 133, 195, 117, 7, 227, 87, 125, 142, 150, 92, 
	154, 137, 27, 71, 10, 233, 114, 98, 201, 209, 30, 13, 226, 243, 57, 67, 1, 
	141, 211, 222, 14, 208, 31, 238, 138, 51, 136, 9, 74, 122, 45, 152, 61, 
	18, 34, 191, 215, 49, 39, 160, 38, 221, 143, 180, 12, 131, 124, 52, 99, 
	78, 129, 59, 106, 231, 94, 203, 90, 120, 126]

digits_gains = [132.3769, 19.3468, 14.247, 4.3593, 7.0206, 8.4383, 2.6395, 
	2.6087, 3.4037, 2.2681, 2.8624, 1.3601, 0.5692, 2.7581, 2.0511, 1.3346, 
	3.6155, 1.3742, 1.0221, 0.3618, 0.1726, 0.8898, 0.7893, 0.7392, 0.9948, 
	0.6893, 0.51, 0.5096, 0.1732, 0.506, 0.2326, 0.4857, 0.4344, 0.2623, 
	0.2858, 0.7386, 0.4207, 0.1898, 0.3478, 0.1395, 0.2196, 0.168, 0.2155, 
	0.3933, 0.3766, 0.5458, 0.4164, 0.4137, 0.266, 0.37, 0.3328, 0.1958, 
	0.2645, 0.2956, 0.2661, 0.2467, 0.3663, 0.252, 0.2453, 0.1533, 0.2195, 
	0.371, 0.1653, 0.1424, 0.1558, 0.1183, 0.1771, 0.0917, 0.13, 0.2125, 
	0.1625, 0.1436, 0.263, 0.1454, 0.3037, 0.2128, 0.2727, 0.2004, 0.1544, 
	0.1161, 0.1492, 0.1326, 0.1472, 0.1741, 0.1245, 0.2288, 0.1378, 0.1216, 
	0.1267]

digits_greedi_ranking =  [96, 179, 216, 219, 54, 26, 56, 206, 225, 187, 163, 
	79, 192, 69, 92, 35, 7, 133, 17, 242, 195, 227, 64, 125, 117, 8, 169, 142, 
	150, 27, 87, 10, 137, 243, 30, 71, 154, 226, 13, 209, 233, 211, 98, 114, 
	57, 9, 1, 74, 131, 67, 51, 122, 31, 222, 18, 191, 215, 49, 39, 208, 201, 
	14, 244, 12, 65, 99, 78, 38, 129, 59, 231, 203, 238, 138, 106, 45, 120, 
	34, 152, 15, 52, 94, 134, 221, 77, 61, 217, 232, 241, 141]

digits_greedi_gains = [132.3769, 19.3468, 5.5078, 13.0984, 3.052, 7.0012, 
	8.0181, 2.6359, 3.4037, 2.2681, 2.8624, 1.3601, 0.5692, 0.4066, 2.6452, 
	2.0193, 1.2513, 0.1726, 2.233, 1.6882, 0.8898, 1.0965, 1.2725, 0.4306, 
	0.7893, 0.363, 1.8682, 0.5096, 0.1663, 0.4344, 0.6893, 0.2858, 0.4857, 
	0.3933, 0.2196, 0.2623, 0.2326, 0.2215, 0.2098, 0.1395, 0.709, 0.266, 
	0.1715, 0.4207, 0.3766, 0.252, 0.4164, 0.2799, 0.3581, 0.5458, 0.2467, 
	0.1533, 0.2645, 0.37, 0.1424, 0.1183, 0.1771, 0.0917, 0.13, 0.1958, 
	0.3132, 0.3328, 0.13, 0.3037, 0.2199, 0.1544, 0.1161, 0.1625, 0.1492, 
	0.1326, 0.1741, 0.2288, 0.2956, 0.2661, 0.1472, 0.2195, 0.1216, 0.1558, 
	0.367, 0.0949, 0.231, 0.1245, 0.162, 0.1436, 0.1452, 0.1347, 0.0929, 
	0.1611, 0.0928, 0.2684]

digits_approx_ranking = [96, 179, 219, 216, 26, 56, 206, 54, 225, 187, 163, 
	79, 192, 17, 92, 35, 242, 64, 69, 169, 195, 133, 117, 7, 227, 87, 8, 142, 
	125, 154, 150, 137, 27, 71, 10, 233, 114, 98, 13, 201, 209, 30, 226, 243, 
	57, 67, 1, 141, 211, 222, 14, 31, 208, 238, 138, 51, 136, 9, 74, 122, 45, 
	152, 61, 18, 180, 34, 191, 215, 49, 39, 160, 38, 221, 143, 12, 131, 124, 
	52, 99, 203, 78, 129, 59, 106, 231, 94, 90, 120, 126]

digits_approx_gains = [132.3769, 19.3468, 14.247, 4.3593, 7.0206, 8.4383, 
	2.6395, 2.6087, 3.4037, 2.2681, 2.8624, 1.3601, 0.5692, 2.7581, 2.6452, 
	2.0511, 1.6882, 1.3742, 0.3618, 1.9058, 0.8898, 0.1726, 0.7893, 0.7392, 
	0.9948, 0.6893, 0.363, 0.5096, 0.393, 0.2326, 0.1663, 0.4857, 0.4344, 
	0.2623, 0.2858, 0.7386, 0.4207, 0.1898, 0.168, 0.3478, 0.1395, 0.2196, 
	0.2155, 0.3933, 0.3766, 0.5458, 0.4164, 0.4137, 0.266, 0.37, 0.3328, 
	0.2645, 0.1958, 0.2956, 0.2661, 0.2467, 0.3663, 0.252, 0.2453, 0.1533, 
	0.2195, 0.371, 0.1653, 0.1424, 0.1454, 0.1558, 0.1183, 0.1771, 0.0917, 
	0.13, 0.2125, 0.1625, 0.1436, 0.263, 0.3037, 0.2128, 0.2727, 0.2004, 
	0.1544, 0.2288, 0.1161, 0.1492, 0.1326, 0.1472, 0.1741, 0.1245, 0.1378, 
	0.1216, 0.1267]

digits_stochastic_ranking = [225, 104, 60, 12, 213, 142, 45, 236, 53, 144, 
	30, 226, 202, 232, 140, 95, 2, 150, 240, 75, 62, 185, 135, 37, 168, 61, 
	241, 212, 121, 124]

digits_stochastic_gains = [2023.6035, 12.477, 28.4395, 2.1767, 10.596, 31.2631, 
	1.8466, 3.2723, 1.4403, 0.266, 33.6989, 3.1455, 2.548, 2.5224, 0.1445, 
	0.5862, 2.4432, 2.1424, 0.4515, 0.4854, 0.8598, 0.1719, 8.4095, 0.8081, 
	1.938, 3.427, 2.0227, 1.1632, 0.7072, 1.5839]

digits_sample_ranking = [179, 216, 206, 219, 26, 56, 54, 163, 187, 79, 242, 17,
	35, 64, 192, 169, 69, 133, 195, 117, 7, 87, 227, 171, 8, 125, 142, 209, 
	137, 27, 149, 71, 10, 18, 114, 13, 201, 30, 226, 98, 243, 57, 67, 1, 141, 
	211, 222, 39, 14, 208, 31, 138, 38, 228, 51, 136, 74, 9, 129, 143, 61, 
	180, 34, 191, 215, 49, 160, 221, 233, 238, 12, 131, 124, 52, 78, 59, 106, 
	99, 94, 90, 120, 196, 50, 126]

digits_sample_gains = [135.1415, 12.9734, 9.23, 12.6779, 6.9931, 8.4351, 2.624, 
	4.6758, 2.4125, 1.3601, 2.2744, 3.2146, 2.0511, 2.3048, 0.4903, 3.921, 
	0.4243, 0.1726, 0.8898, 0.7893, 0.7392, 0.8592, 1.0594, 1.368, 0.363, 
	0.51, 0.5096, 0.2371, 0.4857, 0.4344, 0.9881, 0.2623, 0.2858, 0.5983, 
	0.4506, 0.1896, 0.3478, 0.2196, 0.2215, 0.1715, 0.3933, 0.3766, 0.5458, 
	0.4227, 0.4137, 0.266, 0.37, 0.2294, 0.3328, 0.1958, 0.2645, 0.2661, 
	0.259, 0.3979, 0.2467, 0.3663, 0.2453, 0.2369, 0.2239, 0.3055, 0.1653, 
	0.1608, 0.1558, 0.1183, 0.1771, 0.0917, 0.2125, 0.1436, 0.1724, 0.1718, 
	0.3037, 0.2128, 0.2727, 0.2004, 0.1161, 0.1326, 0.1472, 0.141, 0.1245, 
	0.1378, 0.1216, 0.2449, 0.2774, 0.1281]

digits_modular_ranking = [96, 179, 216, 206, 54, 133, 219, 26, 192, 187, 69, 
	79, 225, 150, 56, 209, 8, 154, 242, 163, 13, 98, 71, 10, 125, 35, 226, 
	195, 117, 142, 30, 87, 64, 17, 7, 27, 92, 137, 49, 114, 227, 229, 208, 
	201, 244, 122, 39, 191, 243, 171, 221, 18, 138, 61, 180, 57, 211, 78, 
	169, 94, 59, 120, 34, 74, 15, 1, 217, 200, 55, 99, 129, 126, 38, 9, 233, 
	90, 65, 51, 14, 238, 178, 159, 149, 106, 241, 213, 82, 45, 31, 160, 222, 
	215, 141, 52]

digits_modular_gains = [132.3769, 19.3468, 5.5078, 3.8928, 3.7407, 7.4877, 
	4.418, 6.5995, 0.5051, 2.4398, 0.4647, 1.3601, 4.9198, 0.2969, 6.3063, 
	0.3112, 1.2141, 0.6595, 1.0254, 2.7387, 0.5686, 0.1775, 1.3718, 0.2858, 
	0.8841, 0.9211, 0.5179, 0.8666, 0.782, 0.5096, 0.2196, 0.9902, 1.045, 
	2.0233, 0.7392, 0.4986, 1.3647, 0.4857, 0.0917, 0.4207, 0.923, 0.1395, 
	0.1498, 0.3478, 0.13, 0.1533, 0.4446, 0.1681, 0.3933, 0.1928, 0.1442, 
	0.1424, 0.2661, 0.1653, 0.0949, 0.3766, 0.2162, 0.1161, 1.5536, 0.1605, 
	0.1326, 0.1961, 0.1558, 0.2493, 0.0949, 0.4164, 0.0929, 0.0948, 0.0844, 
	0.141, 0.1492, 0.1259, 0.1625, 0.2601, 0.6224, 0.1378, 0.3119, 0.2467, 
	0.3328, 0.1718, 0.064, 0.0777, 0.1232, 0.1472, 0.0928, 0.1178, 0.0607, 
	0.2195, 0.2645, 0.1358, 0.334, 0.1326, 0.4131, 0.2004]

# Test all optimizers

def test_digits_naive():
	model = FacilityLocationSelection(25, 'cosine', optimizer='naive')
	model.fit(X_digits_cupy, sample_cost=X_digits_costs)
	assert_array_equal(model.ranking, digits_ranking)
	assert_array_almost_equal(model.gains, digits_gains, 4)
	assert_less_equal(sum(X_digits_costs[model.ranking]), 25)

def test_digits_lazy():
	model = FacilityLocationSelection(25, 'cosine', optimizer='lazy')
	model.fit(X_digits_cupy, sample_cost=X_digits_costs)
	assert_array_equal(model.ranking, digits_ranking)
	assert_array_almost_equal(model.gains, digits_gains, 4)
	assert_less_equal(sum(X_digits_costs[model.ranking]), 25)

def test_digits_two_stage():
	model = FacilityLocationSelection(25, 'cosine', optimizer='two-stage')
	model.fit(X_digits_cupy, sample_cost=X_digits_costs)
	assert_array_equal(model.ranking[:50], digits_ranking[:50])
	assert_array_almost_equal(model.gains[:50], digits_gains[:50], 4)
	assert_less_equal(sum(X_digits_costs[model.ranking]), 25)

def test_digits_greedi_nn():
	model = FacilityLocationSelection(25, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits_cupy, sample_cost=X_digits_costs)
	assert_array_equal(model.ranking, digits_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_greedi_gains, 4)
	assert_less_equal(sum(X_digits_costs[model.ranking]), 25)

def test_digits_greedi_ll():
	model = FacilityLocationSelection(25, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits_cupy, sample_cost=X_digits_costs)
	assert_array_equal(model.ranking, digits_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_greedi_gains, 4)
	assert_less_equal(sum(X_digits_costs[model.ranking]), 25)

def test_digits_greedi_ln():
	model = FacilityLocationSelection(25, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits_cupy, sample_cost=X_digits_costs)
	assert_array_equal(model.ranking, digits_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_greedi_gains, 4)
	assert_less_equal(sum(X_digits_costs[model.ranking]), 25)

def test_digits_greedi_nl():
	model = FacilityLocationSelection(25, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits_cupy, sample_cost=X_digits_costs)
	assert_array_equal(model.ranking, digits_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_greedi_gains, 4)
	assert_less_equal(sum(X_digits_costs[model.ranking]), 25)

def test_digits_approximate():
	model = FacilityLocationSelection(25, 'cosine', optimizer='approximate-lazy')
	model.fit(X_digits_cupy, sample_cost=X_digits_costs)
	assert_array_equal(model.ranking, digits_approx_ranking)
	assert_array_almost_equal(model.gains, digits_approx_gains, 4)
	assert_less_equal(sum(X_digits_costs[model.ranking]), 25)

def test_digits_stochastic():
	model = FacilityLocationSelection(25, 'cosine', optimizer='stochastic',
		random_state=0)
	model.fit(X_digits_cupy, sample_cost=X_digits_costs)
	assert_array_equal(model.ranking, digits_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_stochastic_gains, 4)
	assert_less_equal(sum(X_digits_costs[model.ranking]), 25)

def test_digits_sample():
	model = FacilityLocationSelection(25, 'cosine', optimizer='sample',
		random_state=0)
	model.fit(X_digits_cupy, sample_cost=X_digits_costs)
	assert_array_equal(model.ranking, digits_sample_ranking)
	assert_array_almost_equal(model.gains, digits_sample_gains, 4)
	assert_less_equal(sum(X_digits_costs[model.ranking]), 25)

def test_digits_modular():
	model = FacilityLocationSelection(25, 'cosine', optimizer='modular',
		random_state=0)
	model.fit(X_digits_cupy, sample_cost=X_digits_costs)
	assert_array_equal(model.ranking, digits_modular_ranking)
	assert_array_almost_equal(model.gains, digits_modular_gains, 4)
	assert_less_equal(sum(X_digits_costs[model.ranking]), 25)
