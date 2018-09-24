import scipy
import numpy

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

digits_corr_ranking = [424, 1647, 396, 339, 1030, 331, 983, 1075, 1482, 
	1539, 1282, 493, 885, 823, 1051, 236, 537, 1161, 345, 1788, 1432, 1634, 
	1718, 1676, 146, 1286, 655, 1292, 556, 533, 1545, 520, 1711, 1428, 620, 
	1276, 305, 438, 1026, 183, 2, 384, 1012, 798, 213, 1291, 162, 1206, 227, 
	1655, 233, 1508, 410, 1295, 1312, 1350, 514, 938, 579, 1066, 82, 164, 948, 
	1588, 1294, 1682, 943, 517, 959, 1429, 762, 898, 1556, 881, 1470, 1549, 1325, 
	1568, 937, 347, 1364, 126, 732, 1168, 241, 573, 731, 815, 864, 1639, 1570, 411, 
	1086, 696, 870, 1156, 353, 160, 1381, 326]

digits_euclidean_ranking = [945, 392, 1507, 793, 1417, 1039, 97, 1107, 1075, 
	867, 360, 186, 1584, 1422, 885, 1084, 1327, 1696, 991, 146, 181, 765, 
	175, 1513, 1120, 877, 1201, 1764, 1711, 1447, 1536, 1286, 438, 612, 6, 
	514, 410, 1545, 384, 1053, 1485, 983, 310, 51, 654, 1312, 708, 157, 259, 
	1168, 117, 1634, 1537, 1188, 1364, 1713, 579, 582, 69, 200, 1678, 798, 183, 
	520, 1011, 1295, 1291, 938, 1276, 501, 696, 948, 925, 558, 269, 1066, 573, 
	762, 1294, 1588, 732, 1387, 1568, 1026, 1156, 79, 1222, 1414, 864, 1549, 
	1236, 213, 411, 151, 233, 924, 126, 345, 1421, 1562]

digits_cosine_ranking = [424, 615, 1545, 1385, 1399, 1482, 1539, 1075, 331, 493, 
	885, 236, 345, 1282, 1051, 823, 537, 1788, 1549, 834, 1634, 1009, 1718, 655, 
	1474, 1292, 1185, 396, 1676, 2, 183, 533, 1536, 438, 1276, 305, 1353, 620, 
	1026, 983, 162, 1012, 384, 91, 227, 798, 1291, 1655, 1485, 1206, 410, 556, 
	1161, 29, 1320, 1295, 164, 514, 1294, 1711, 579, 938, 517, 1682, 1325, 1222, 
	82, 959, 520, 1066, 943, 1556, 762, 898, 732, 1086, 881, 1588, 1470, 1568, 1678, 
	948, 1364, 62, 937, 1156, 1168, 241, 573, 347, 908, 1628, 1442, 126, 815, 411, 
	1257, 151, 23, 696]

def test_digits_corr_small_greedy():
	model = FacilityLocationSelection(10, 'corr', 10)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking[:10])

def test_digits_corr_small_pivot():
	model = FacilityLocationSelection(10, 'corr', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking[:10])

def test_digits_corr_small_pq():
	model = FacilityLocationSelection(10, 'corr', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking[:10])

def test_digits_corr_small_truncated():
	model = FacilityLocationSelection(15, 'corr', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_corr_ranking[:10])

def test_digits_corr_small_truncated_pivot():
	model = FacilityLocationSelection(15, 'corr', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_corr_ranking[:10])

def test_digits_corr_large_greedy():
	model = FacilityLocationSelection(100, 'corr', 100)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)

def test_digits_corr_large_pivot():
	model = FacilityLocationSelection(100, 'corr', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)

def test_digits_corr_large_pq():
	model = FacilityLocationSelection(100, 'corr', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)

def test_digits_corr_large_truncated():
	model = FacilityLocationSelection(150, 'corr', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_corr_ranking)

def test_digits_corr_large_truncated_pivot():
	model = FacilityLocationSelection(150, 'corr', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_corr_ranking)

def test_digits_euclidean_small_greedy():
	model = FacilityLocationSelection(10, 'euclidean', 10)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking[:10])

def test_digits_euclidean_small_pivot():
	model = FacilityLocationSelection(10, 'euclidean', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking[:10])

def test_digits_euclidean_small_pq():
	model = FacilityLocationSelection(10, 'euclidean', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking[:10])

def test_digits_euclidean_small_truncated():
	model = FacilityLocationSelection(15, 'euclidean', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_euclidean_ranking[:10])

def test_digits_euclidean_small_truncated_pivot():
	model = FacilityLocationSelection(15, 'euclidean', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_euclidean_ranking[:10])

def test_digits_euclidean_large_greedy():
	model = FacilityLocationSelection(100, 'euclidean', 100)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:25], digits_euclidean_ranking[:25])
	assert_array_equal(model.ranking[-25:], digits_euclidean_ranking[-25:])

def test_digits_euclidean_large_pivot():
	model = FacilityLocationSelection(100, 'euclidean', 25)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)

def test_digits_euclidean_large_pq():
	model = FacilityLocationSelection(100, 'euclidean', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)

def test_digits_euclidean_large_truncated():
	model = FacilityLocationSelection(150, 'euclidean', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_euclidean_ranking)

def test_digits_euclidean_large_truncated_pivot():
	model = FacilityLocationSelection(150, 'euclidean', 10)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_euclidean_ranking)

def test_digits_cosine_small_greedy():
	model = FacilityLocationSelection(10, 'cosine', 10)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking[:10])

def test_digits_cosine_small_pivot():
	model = FacilityLocationSelection(10, 'cosine', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking[:10])

def test_digits_cosine_small_pq():
	model = FacilityLocationSelection(10, 'cosine', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking[:10])

def test_digits_cosine_small_truncated():
	model = FacilityLocationSelection(15, 'cosine', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_cosine_ranking[:10])

def test_digits_cosine_small_truncated_pivot():
	model = FacilityLocationSelection(15, 'cosine', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_cosine_ranking[:10])

def test_digits_cosine_large_greedy():
	model = FacilityLocationSelection(100, 'cosine', 100)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)

def test_digits_cosine_large_pivot():
	model = FacilityLocationSelection(100, 'cosine', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)

def test_digits_cosine_large_pq():
	model = FacilityLocationSelection(100, 'cosine', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)

def test_digits_cosine_large_truncated():
	model = FacilityLocationSelection(150, 'cosine', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_cosine_ranking)

def test_digits_cosine_large_truncated_pivot():
	model = FacilityLocationSelection(150, 'cosine', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_cosine_ranking)

def test_digits_cosine_small_precomputed_greedy():
	model = FacilityLocationSelection(10, 'precomputed', 10)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking[:10])

def test_digits_cosine_small_precomputed_pivot():
	model = FacilityLocationSelection(10, 'precomputed', 5)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking[:10])

def test_digits_cosine_small_precomputed_pq():
	model = FacilityLocationSelection(10, 'precomputed', 1)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking[:10])

def test_digits_cosine_small_precomputed_truncated():
	model = FacilityLocationSelection(15, 'precomputed', 1)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking[:10], digits_cosine_ranking[:10])

def test_digits_cosine_small_precomputed_truncated_pivot():
	model = FacilityLocationSelection(15, 'precomputed', 5)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking[:10], digits_cosine_ranking[:10])

def test_digits_cosine_large_precomputed_greedy():
	model = FacilityLocationSelection(100, 'precomputed', 100)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)

def test_digits_cosine_large_precomputed_pivot():
	model = FacilityLocationSelection(100, 'precomputed', 50)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)

def test_digits_cosine_large_precomputed_pq():
	model = FacilityLocationSelection(100, 'precomputed', 1)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)

def test_digits_cosine_large_precomputed_truncated():
	model = FacilityLocationSelection(150, 'precomputed', 1)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking[:100], digits_cosine_ranking)

def test_digits_cosine_large_precomputed_truncated_pivot():
	model = FacilityLocationSelection(150, 'precomputed', 50)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking[:100], digits_cosine_ranking)
	