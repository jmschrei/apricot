import scipy
import numpy

try:
	import cupy
except:
	import numpy as cupy

from apricot import FacilityLocationSelection

from sklearn.datasets import load_digits

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

digits_data = load_digits()
X_digits = digits_data.data

norm = lambda x: numpy.sqrt((x*x).sum(axis=1)).reshape(x.shape[0], 1)
cosine = lambda x: numpy.dot(x, x.T) / (norm(x).dot(norm(x).T))

X_digits_sparse = scipy.sparse.csr_matrix(cosine(X_digits))

X_digits_corr_cupy = cupy.corrcoef(cupy.array(X_digits), rowvar=True) ** 2
X_digits_cosine_cupy = cupy.array(cosine(X_digits))

digits_corr_ranking = [424, 1647, 396, 339, 1030, 331, 983, 1075, 1482, 
	1539, 1282, 493, 885, 823, 1051, 236, 537, 1161, 345, 1788, 1432, 1634, 
	1718, 1676, 146, 1286, 655, 1292, 556, 533, 1545, 520, 1711, 1428, 620, 
	1276, 305, 438, 1026, 183, 2, 384, 1012, 798, 213, 1291, 162, 1206, 227, 
	1655, 233, 1508, 410, 1295, 1312, 1350, 514, 938, 579, 1066, 82, 164, 948, 
	1588, 1294, 1682, 943, 517, 959, 1429, 762, 898, 1556, 881, 1470, 1549, 1325, 
	1568, 937, 347, 1364, 126, 732, 1168, 241, 573, 731, 815, 864, 1639, 1570, 411, 
	1086, 696, 870, 1156, 353, 160, 1381, 326]

digits_corr_gains = [736.794, 114.2782, 65.4154, 61.3037, 54.5428, 38.7506, 34.097, 
	32.6649, 30.2813, 27.8395, 24.1843, 17.568, 16.6615, 15.2973, 13.2629, 9.7685, 
	9.5183, 7.9484, 7.8094, 7.0224, 6.623, 6.061, 6.0469, 5.323, 5.167, 5.0563, 
	4.8848, 4.7694, 4.4766, 4.4577, 4.3198, 3.9347, 3.5501, 3.3284, 3.0123, 2.9994, 
	2.9739, 2.8233, 2.7572, 2.558, 2.5281, 2.4412, 2.4328, 2.3452, 2.2498, 2.2457, 
	2.2127, 2.1542, 2.1416, 2.0876, 2.0715, 2.0482, 2.0053, 1.9996, 1.9912, 1.973, 
	1.8029, 1.7865, 1.7284, 1.7255, 1.7201, 1.7169, 1.6614, 1.6445, 1.6147, 1.5874, 
	1.5827, 1.5822, 1.5784, 1.5164, 1.4876, 1.4319, 1.4288, 1.3736, 1.3485, 1.3039, 
	1.2872, 1.2771, 1.2587, 1.2391, 1.2279, 1.2006, 1.1654, 1.1491, 1.1445, 1.137, 
	1.1122, 1.0785, 1.0771, 1.0402, 1.0321, 1.0192, 1.0158, 0.9734, 0.9627, 0.9612, 
	0.9401, 0.9291, 0.912, 0.8924]

digits_euclidean_ranking = [945, 392, 1507, 793, 1417, 1039, 97, 1107, 1075, 
	867, 360, 186, 1584, 1422, 885, 1084, 1327, 1696, 991, 146, 181, 765, 
	175, 1513, 1120, 877, 1201, 1764, 1711, 1447, 1536, 1286, 438, 612, 6, 
	514, 410, 1545, 384, 1053, 1485, 983, 310, 51, 654, 1312, 708, 157, 259, 
	1168, 117, 1634, 1537, 1188, 1364, 1713, 579, 582, 69, 200, 1678, 798, 183, 
	520, 1011, 1295, 1291, 938, 1276, 501, 696, 948, 925, 558, 269, 1066, 573, 
	762, 1294, 1588, 732, 1387, 1568, 1026, 1156, 79, 1222, 1414, 864, 1549, 
	1236, 213, 411, 151, 233, 924, 126, 345, 1421, 1562]

digits_euclidean_gains = [7448636.0, 384346.0, 250615.0, 224118.0, 166266.0, 
	127456.0, 122986.0, 109483.0, 93463.0, 67173.0, 55997.0, 54721.0, 51497.0, 
	47765.0, 45073.0, 33857.0, 30100.0, 25043.0, 22260.0, 19700.0, 19135.0, 
	17545.0, 17000.0, 15462.0, 15315.0, 14996.0, 14819.0, 13244.0, 12529.0, 
	12474.0, 11702.0, 11639.0, 11612.0, 11266.0, 11187.0, 9722.0, 9244.0, 
	8645.0, 8645.0, 8461.0, 8404.0, 8115.0, 7998.0, 7351.0, 7153.0, 6992.0, 
	6956.0, 6919.0, 6711.0, 6684.0, 6526.0, 6348.0, 6099.0, 5969.0, 5460.0, 
	5433.0, 5163.0, 5141.0, 5090.0, 4900.0, 4842.0, 4683.0, 4165.0, 4104.0, 
	4099.0, 4099.0, 3998.0, 3959.0, 3912.0, 3807.0, 3703.0, 3675.0, 3670.0, 
	3636.0, 3564.0, 3407.0, 3395.0, 3196.0, 3188.0, 3168.0, 3156.0, 3144.0, 
	3093.0, 3078.0, 3059.0, 2997.0, 2944.0, 2891.0, 2886.0, 2865.0, 2804.0, 
	2779.0, 2756.0, 2748.0, 2709.0, 2696.0, 2651.0, 2637.0, 2619.0, 2602.0]

digits_cosine_ranking = [424, 615, 1545, 1385, 1399, 1482, 1539, 1075, 331, 493, 
	885, 236, 345, 1282, 1051, 823, 537, 1788, 1549, 834, 1634, 1009, 1718, 655, 
	1474, 1292, 1185, 396, 1676, 2, 183, 533, 1536, 438, 1276, 305, 1353, 620, 
	1026, 983, 162, 1012, 384, 91, 227, 798, 1291, 1655, 1485, 1206, 410, 556, 
	1161, 29, 1320, 1295, 164, 514, 1294, 1711, 579, 938, 517, 1682, 1325, 1222, 
	82, 959, 520, 1066, 943, 1556, 762, 898, 732, 1086, 881, 1588, 1470, 1568, 1678, 
	948, 1364, 62, 937, 1156, 1168, 241, 573, 347, 908, 1628, 1442, 126, 815, 411, 
	1257, 151, 23, 696]

digits_cosine_gains = [1418.7103, 47.8157, 25.4947, 21.0313, 19.7599, 19.0236, 
	16.3013, 13.5381, 11.811, 9.0032, 6.2765, 5.9886, 5.2185, 4.6696, 4.1744, 
	4.0718, 3.0075, 2.8132, 2.5777, 2.2983, 2.2391, 2.2223, 2.0622, 1.9568, 
	1.9192, 1.7356, 1.7038, 1.6463, 1.6003, 1.5979, 1.3458, 1.3415, 1.288, 
	1.1595, 1.0048, 0.9198, 0.8886, 0.8454, 0.8446, 0.829, 0.8162, 0.799, 
	0.7805, 0.7723, 0.7717, 0.7681, 0.7533, 0.7227, 0.7017, 0.6899, 0.6895, 
	0.6448, 0.6397, 0.6334, 0.6014, 0.5881, 0.5677, 0.5628, 0.5534, 0.5527, 
	0.5428, 0.5415, 0.5384, 0.5249, 0.5232, 0.498, 0.4944, 0.4877, 0.4799, 
	0.4788, 0.4775, 0.4663, 0.4641, 0.4589, 0.4447, 0.4437, 0.4408, 0.4382, 
	0.4312, 0.4266, 0.4238, 0.4184, 0.4168, 0.4058, 0.4, 0.3983, 0.3892, 
	0.3855, 0.3837, 0.3818, 0.3765, 0.3524, 0.3519, 0.3471, 0.3331, 0.3289, 
	0.3268, 0.324, 0.3197, 0.3173]

def test_digits_corr_small_greedy():
	model = FacilityLocationSelection(10, 'corr', 10)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking[:10])
	assert_array_almost_equal(model.gains, digits_corr_gains[:10], 4)

def test_digits_corr_small_greedy_rank_initialized():
	model = FacilityLocationSelection(10, 'corr', 10, initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking[5:15])
	assert_array_almost_equal(model.gains, digits_corr_gains[5:15], 4)

def test_digits_corr_small_greedy_bool_initialized():
	mask = numpy.zeros(X_digits.shape[0], dtype=bool)
	mask[digits_corr_ranking[:5]] = True
	model = FacilityLocationSelection(10, 'corr', 10, initial_subset=mask)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking[5:15])
	assert_array_almost_equal(model.gains, digits_corr_gains[5:15], 4)

def test_digits_corr_small_pivot():
	model = FacilityLocationSelection(10, 'corr', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking[:10])
	assert_array_almost_equal(model.gains, digits_corr_gains[:10], 4)

def test_digits_corr_small_pivot_rank_initialized():
	model = FacilityLocationSelection(10, 'corr', 5, initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking[5:15])
	assert_array_almost_equal(model.gains, digits_corr_gains[5:15], 4)

def test_digits_corr_small_pivot_bool_initialized():
	mask = numpy.zeros(X_digits.shape[0], dtype=bool)
	mask[digits_corr_ranking[:5]] = True
	model = FacilityLocationSelection(10, 'corr', 5, initial_subset=mask)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking[5:15])
	assert_array_almost_equal(model.gains, digits_corr_gains[5:15], 4)

def test_digits_corr_small_pq():
	model = FacilityLocationSelection(10, 'corr', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking[:10])
	assert_array_almost_equal(model.gains, digits_corr_gains[:10], 4)

def test_digits_corr_small_pq_rank_initialized():
	model = FacilityLocationSelection(10, 'corr', 1, initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking[5:15])
	assert_array_almost_equal(model.gains, digits_corr_gains[5:15], 4)

def test_digits_corr_small_pq_bool_initialized():
	mask = numpy.zeros(X_digits.shape[0], dtype=bool)
	mask[digits_corr_ranking[:5]] = True
	model = FacilityLocationSelection(10, 'corr', 1, initial_subset=mask)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking[5:15])
	assert_array_almost_equal(model.gains, digits_corr_gains[5:15], 4)

def test_digits_corr_small_truncated():
	model = FacilityLocationSelection(15, 'corr', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_corr_ranking[:10])
	assert_array_almost_equal(model.gains[:10], digits_corr_gains[:10], 4)

def test_digits_corr_small_truncated_pivot():
	model = FacilityLocationSelection(15, 'corr', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_corr_ranking[:10])
	assert_array_almost_equal(model.gains[:10], digits_corr_gains[:10], 4)

def test_digits_corr_large_greedy():
	model = FacilityLocationSelection(100, 'corr', 100)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_corr_large_pivot():
	model = FacilityLocationSelection(100, 'corr', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_corr_large_pq():
	model = FacilityLocationSelection(100, 'corr', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_corr_large_truncated():
	model = FacilityLocationSelection(150, 'corr', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_corr_ranking)
	assert_array_almost_equal(model.gains[:100], digits_corr_gains, 4)

def test_digits_corr_large_truncated_pivot():
	model = FacilityLocationSelection(150, 'corr', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_corr_ranking)
	assert_array_almost_equal(model.gains[:100], digits_corr_gains, 4)

def test_digits_euclidean_small_greedy():
	model = FacilityLocationSelection(10, 'euclidean', 10)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking[:10])
	assert_array_almost_equal(model.gains, digits_euclidean_gains[:10], 4)

def test_digits_euclidean_small_pivot():
	model = FacilityLocationSelection(10, 'euclidean', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking[:10])
	assert_array_almost_equal(model.gains, digits_euclidean_gains[:10], 4)

def test_digits_euclidean_small_pq():
	model = FacilityLocationSelection(10, 'euclidean', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking[:10])
	assert_array_almost_equal(model.gains, digits_euclidean_gains[:10], 4)

def test_digits_euclidean_small_truncated():
	model = FacilityLocationSelection(15, 'euclidean', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_euclidean_ranking[:10])
	assert_array_almost_equal(model.gains[:10], digits_euclidean_gains[:10], 4)

def test_digits_euclidean_small_truncated_pivot():
	model = FacilityLocationSelection(15, 'euclidean', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_euclidean_ranking[:10])
	assert_array_almost_equal(model.gains[:10], digits_euclidean_gains[:10], 4)

def test_digits_euclidean_large_greedy():
	model = FacilityLocationSelection(100, 'euclidean', 100)
	model.fit(X_digits)

	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)
	
	# There is one swap in the resulting ranking because two points have
	# the same distance. Thus, the gains are the same but this swap means
	# that the ranking aren't the same.
	assert_array_equal(model.ranking[:25], digits_euclidean_ranking[:25])
	assert_array_equal(model.ranking[-25:], digits_euclidean_ranking[-25:])

def test_digits_euclidean_large_pivot():
	model = FacilityLocationSelection(100, 'euclidean', 25)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)

def test_digits_euclidean_large_pq():
	model = FacilityLocationSelection(100, 'euclidean', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)

def test_digits_euclidean_large_truncated():
	model = FacilityLocationSelection(150, 'euclidean', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_euclidean_ranking)
	assert_array_almost_equal(model.gains[:100], digits_euclidean_gains, 4)

def test_digits_euclidean_large_truncated_pivot():
	model = FacilityLocationSelection(150, 'euclidean', 10)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_euclidean_ranking)
	assert_array_almost_equal(model.gains[:100], digits_euclidean_gains, 4)

def test_digits_cosine_small_greedy():
	model = FacilityLocationSelection(10, 'cosine', 10)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking[:10])
	assert_array_almost_equal(model.gains, digits_cosine_gains[:10], 4)

def test_digits_cosine_small_pivot():
	model = FacilityLocationSelection(10, 'cosine', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking[:10])
	assert_array_almost_equal(model.gains, digits_cosine_gains[:10], 4)

def test_digits_cosine_small_pq():
	model = FacilityLocationSelection(10, 'cosine', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking[:10])
	assert_array_almost_equal(model.gains, digits_cosine_gains[:10], 4)

def test_digits_cosine_small_truncated():
	model = FacilityLocationSelection(15, 'cosine', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_cosine_ranking[:10])
	assert_array_almost_equal(model.gains[:10], digits_cosine_gains[:10], 4)

def test_digits_cosine_small_truncated_pivot():
	model = FacilityLocationSelection(15, 'cosine', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_cosine_ranking[:10])
	assert_array_almost_equal(model.gains[:10], digits_cosine_gains[:10], 4)

def test_digits_cosine_large_greedy():
	model = FacilityLocationSelection(100, 'cosine', 100)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_large_pivot():
	model = FacilityLocationSelection(100, 'cosine', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_large_pq():
	model = FacilityLocationSelection(100, 'cosine', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_large_truncated():
	model = FacilityLocationSelection(150, 'cosine', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_cosine_ranking)
	assert_array_almost_equal(model.gains[:100], digits_cosine_gains, 4)

def test_digits_cosine_large_truncated_pivot():
	model = FacilityLocationSelection(150, 'cosine', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_cosine_ranking)
	assert_array_almost_equal(model.gains[:100], digits_cosine_gains, 4)

def test_digits_cosine_small_precomputed_greedy():
	model = FacilityLocationSelection(10, 'precomputed', 10)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking[:10])
	assert_array_almost_equal(model.gains, digits_cosine_gains[:10], 4)

def test_digits_cosine_small_precomputed_pivot():
	model = FacilityLocationSelection(10, 'precomputed', 5)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking[:10])
	assert_array_almost_equal(model.gains, digits_cosine_gains[:10], 4)

def test_digits_cosine_small_precomputed_pq():
	model = FacilityLocationSelection(10, 'precomputed', 1)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking[:10])
	assert_array_almost_equal(model.gains, digits_cosine_gains[:10], 4)

def test_digits_cosine_small_precomputed_truncated():
	model = FacilityLocationSelection(15, 'precomputed', 1)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking[:10], digits_cosine_ranking[:10])
	assert_array_almost_equal(model.gains[:10], digits_cosine_gains[:10], 4)

def test_digits_cosine_small_precomputed_truncated_pivot():
	model = FacilityLocationSelection(15, 'precomputed', 5)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking[:10], digits_cosine_ranking[:10])
	assert_array_almost_equal(model.gains[:10], digits_cosine_gains[:10], 4)

def test_digits_cosine_large_precomputed_greedy():
	model = FacilityLocationSelection(100, 'precomputed', 100)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_large_precomputed_pivot():
	model = FacilityLocationSelection(100, 'precomputed', 50)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_large_precomputed_pq():
	model = FacilityLocationSelection(100, 'precomputed', 1)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_large_precomputed_truncated():
	model = FacilityLocationSelection(150, 'precomputed', 1)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking[:100], digits_cosine_ranking)
	assert_array_almost_equal(model.gains[:100], digits_cosine_gains, 4)

def test_digits_cosine_large_precomputed_truncated_pivot():
	model = FacilityLocationSelection(150, 'precomputed', 50)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking[:100], digits_cosine_ranking)
	assert_array_almost_equal(model.gains[:100], digits_cosine_gains, 4)

def test_digits_corr_small_precomputed_greedy():
	model = FacilityLocationSelection(10, 'precomputed', 10)
	model.fit(X_digits_corr_cupy)
	assert_array_equal(model.ranking, digits_corr_ranking[:10])
	assert_array_almost_equal(model.gains, digits_corr_gains[:10], 4)

def test_digits_corr_large_precomputed_greedy():
	model = FacilityLocationSelection(100, 'precomputed', 100)
	model.fit(X_digits_corr_cupy)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_cosine_small_precomputed_greedy():
	model = FacilityLocationSelection(10, 'precomputed', 10)
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking, digits_cosine_ranking[:10])
	assert_array_almost_equal(model.gains, digits_cosine_gains[:10], 4)

def test_digits_cosine_large_precomputed_greedy():
	model = FacilityLocationSelection(100, 'precomputed', 100)
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
