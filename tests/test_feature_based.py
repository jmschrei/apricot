import scipy
import numpy

from apricot import FeatureBasedSelection

from sklearn.datasets import load_digits

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

digits_data = load_digits()
X_digits = digits_data.data

digits_sqrt_ranking = [818, 1296, 732, 988, 629, 1747, 951, 235, 1375, 1205, 
	1572, 1766, 178, 1657, 898, 1271, 513, 591, 160, 736, 1070, 185, 1113, 491, 
	1793, 1017, 283, 221, 1493, 688, 538, 423, 919, 1796, 163, 1022, 1176, 208, 
	890, 565, 693, 313, 1009, 1317, 956, 502, 1043, 1082, 1273, 1313, 1030, 33, 
	756, 263, 768, 1487, 586, 1086, 854, 430, 615, 805, 1193, 1393, 1704, 352, 
	757, 873, 666, 979, 457, 424, 481, 1437, 1342, 407, 1349, 1668, 1470, 1021, 
	77, 1191, 673, 1305, 453, 786, 851, 1186, 168, 500, 1012, 548, 1071, 451, 
	1260, 436, 1106, 655, 372, 1263]

digits_log_ranking = [818, 1296, 732, 988, 629, 1657, 1375, 1572, 1271, 1070, 
	163, 873, 1086, 1176, 502, 591, 757, 1205, 1273, 919, 1043, 1313, 1264, 
	1576, 283, 586, 951, 756, 1022, 1493, 1113, 956, 673, 565, 1017, 1305, 
	235, 1321, 851, 87, 689, 1317, 1248, 33, 1754, 538, 263, 1191, 160, 1260, 
	1109, 317, 1487, 1277, 1263, 1012, 979, 1778, 1009, 178, 313, 998, 430, 
	1082, 889, 947, 655, 1393, 494, 786, 1437, 1193, 372, 767, 1293, 800, 436, 
	1635, 1106, 566, 513, 447, 1718, 1793, 609, 1342, 1302, 900, 898, 77, 1470, 
	517, 1290, 1023, 457, 1094, 488, 185, 788, 1033]

digits_inverse_ranking = [505, 732, 1273, 988, 1375, 1264, 502, 1594, 1070, 1271, 
	873, 757, 956, 87, 1576, 1086, 1313, 1043, 919, 163, 566, 1305, 673, 1176, 
	1657, 586, 800, 1572, 1248, 1321, 756, 1165, 1022, 1277, 947, 1754, 327, 447, 
	1293, 900, 1259, 689, 1263, 1778, 629, 889, 998, 1735, 1191, 317, 851, 591, 
	897, 494, 1393, 33, 1012, 1341, 1302, 1094, 1260, 1109, 467, 1023, 1742, 
	283, 1618, 1274, 655, 609, 1215, 1718, 1033, 263, 951, 982, 1635, 413, 1205,
	1600, 771, 1589, 599, 1001, 1017, 1290, 1268, 1009, 517, 488, 1115, 1731, 
	1493, 176, 430, 734, 1708, 1211, 538, 1487]

def test_digits_sqrt_small_greedy():
	model = FeatureBasedSelection(10, 'sqrt', 10)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking[:10])

def test_digits_sqrt_small_pivot():
	model = FeatureBasedSelection(10, 'sqrt', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking[:10])

def test_digits_sqrt_small_pq():
	model = FeatureBasedSelection(10, 'sqrt', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking[:10])

def test_digits_sqrt_small_truncated():
	model = FeatureBasedSelection(15, 'sqrt', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_sqrt_ranking[:10])

def test_digits_sqrt_small_truncated_pivot():
	model = FeatureBasedSelection(15, 'sqrt', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_sqrt_ranking[:10])

def test_digits_sqrt_large_greedy():
	model = FeatureBasedSelection(100, 'sqrt', 100)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking)

def test_digits_sqrt_large_pivot():
	model = FeatureBasedSelection(100, 'sqrt', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking)

def test_digits_sqrt_large_pq():
	model = FeatureBasedSelection(100, 'sqrt', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking)

def test_digits_sqrt_large_truncated():
	model = FeatureBasedSelection(150, 'sqrt', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_sqrt_ranking)

def test_digits_sqrt_large_truncated_pivot():
	model = FeatureBasedSelection(150, 'sqrt', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_sqrt_ranking)

def test_digits_log_small_greedy():
	model = FeatureBasedSelection(10, 'log', 10)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_log_ranking[:10])

def test_digits_log_small_pivot():
	model = FeatureBasedSelection(10, 'log', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_log_ranking[:10])

def test_digits_log_small_pq():
	model = FeatureBasedSelection(10, 'log', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_log_ranking[:10])

def test_digits_log_small_truncated():
	model = FeatureBasedSelection(15, 'log', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_log_ranking[:10])

def test_digits_log_small_truncated_pivot():
	model = FeatureBasedSelection(15, 'log', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_log_ranking[:10])

def test_digits_log_large_greedy():
	model = FeatureBasedSelection(100, 'log', 100)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_log_ranking)

def test_digits_log_large_pivot():
	model = FeatureBasedSelection(100, 'log', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_log_ranking)

def test_digits_log_large_pq():
	model = FeatureBasedSelection(100, 'log', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_log_ranking)

def test_digits_log_large_truncated():
	model = FeatureBasedSelection(150, 'log', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_log_ranking)

def test_digits_log_large_truncated_pivot():
	model = FeatureBasedSelection(150, 'log', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_log_ranking)

def test_digits_inverse_small_greedy():
	model = FeatureBasedSelection(10, 'inverse', 10)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_inverse_ranking[:10])

def test_digits_inverse_small_pivot():
	model = FeatureBasedSelection(10, 'inverse', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_inverse_ranking[:10])

def test_digits_inverse_small_pq():
	model = FeatureBasedSelection(10, 'inverse', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_inverse_ranking[:10])

def test_digits_inverse_small_truncated():
	model = FeatureBasedSelection(15, 'inverse', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_inverse_ranking[:10])

def test_digits_inverse_small_truncated_pivot():
	model = FeatureBasedSelection(15, 'inverse', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_inverse_ranking[:10])

def test_digits_inverse_large_greedy():
	model = FeatureBasedSelection(100, 'inverse', 100)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_inverse_ranking)

def test_digits_inverse_large_pivot():
	model = FeatureBasedSelection(100, 'inverse', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_inverse_ranking)

def test_digits_inverse_large_pq():
	model = FeatureBasedSelection(100, 'inverse', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_inverse_ranking)

def test_digits_inverse_large_truncated():
	model = FeatureBasedSelection(150, 'inverse', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_inverse_ranking)

def test_digits_inverse_large_truncated_pivot():
	model = FeatureBasedSelection(150, 'inverse', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_inverse_ranking)

