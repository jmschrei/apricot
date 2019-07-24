import scipy
import numpy

from apricot import FeatureBasedSelection

from sklearn.datasets import load_digits

from numpy.testing import assert_equal
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

digits_sqrt_gains = [124.8187, 59.9654, 47.7603, 37.88, 34.3933, 29.819, 27.4785, 
	25.2363, 23.6127, 22.6002, 21.1175, 20.1914, 19.1849, 18.5015, 17.9476, 17.3043, 
	16.6255, 16.0984, 15.7533, 15.1851, 14.8136, 14.4975, 14.1907, 13.8717, 13.4887, 
	13.2287, 13.0037, 12.6406, 12.4373, 12.1814, 11.9999, 11.7982, 11.6161, 11.4169, 
	11.2585, 11.1032, 10.9829, 10.8366, 10.6663, 10.5211, 10.4042, 10.2704, 10.1146, 
	10.0116, 9.8487, 9.7181, 9.6466, 9.529, 9.4284, 9.3388, 9.2417, 9.1192, 9.0285, 
	8.9639, 8.8704, 8.7846, 8.7234, 8.6348, 8.5746, 8.4954, 8.4284, 8.349, 8.28, 8.1906, 
	8.1304, 8.0608, 8.0066, 7.9458, 7.8825, 7.8351, 7.7756, 7.7179, 7.6538, 7.591, 7.544, 
	7.491, 7.4324, 7.3709, 7.3239, 7.2695, 7.2282, 7.1841, 7.1514, 7.1167, 7.0633, 7.009, 
	6.9693, 6.9245, 6.8838, 6.8409, 6.8031, 6.7691, 6.7295, 6.6905, 6.655, 6.6156, 6.5894, 
	6.5419, 6.5137, 6.4752]

digits_log_ranking = [818, 1296, 732, 988, 629, 1657, 1375, 1572, 1271, 1070, 
	163, 873, 1086, 1176, 502, 591, 757, 1205, 1273, 919, 1043, 1313, 1264, 
	1576, 283, 586, 951, 756, 1022, 1493, 1113, 956, 673, 565, 1017, 1305, 
	235, 1321, 851, 87, 689, 1317, 1248, 33, 1754, 538, 263, 1191, 160, 1260, 
	1109, 317, 1487, 1277, 1263, 1012, 979, 1778, 1009, 178, 313, 998, 430, 
	1082, 889, 947, 655, 1393, 494, 786, 1437, 1193, 372, 767, 1293, 800, 436, 
	1635, 1106, 566, 513, 447, 1718, 1793, 609, 1342, 1302, 900, 898, 77, 1470, 
	517, 1290, 1023, 457, 1094, 488, 185, 788, 1033]

digits_log_gains = [91.5816, 34.6074, 25.542, 16.4452, 13.9838, 11.1475, 9.1666, 
	7.6363, 6.8872, 5.7784, 5.3453, 4.7717, 4.5818, 4.1378, 3.8602, 3.5665, 3.4442, 
	3.1993, 2.9802, 2.8595, 2.7123, 2.6332, 2.5552, 2.419, 2.3094, 2.2069, 2.1207, 
	2.0497, 1.9825, 1.8702, 1.8015, 1.7358, 1.6978, 1.6348, 1.5802, 1.5312, 1.4802, 
	1.4477, 1.4025, 1.3824, 1.3361, 1.3128, 1.2653, 1.2326, 1.2089, 1.1864, 1.1684, 
	1.1311, 1.104, 1.0827, 1.0572, 1.0433, 1.0235, 0.9939, 0.9825, 0.9685, 0.9532, 
	0.9373, 0.9189, 0.8972, 0.8808, 0.869, 0.8477, 0.8338, 0.8175, 0.808, 0.7975, 
	0.785, 0.7693, 0.7611, 0.7532, 0.7334, 0.7213, 0.7139, 0.706, 0.6899, 0.6817, 
	0.6738, 0.665, 0.6595, 0.6501, 0.6383, 0.6318, 0.6265, 0.6206, 0.6104, 0.6003, 
	0.5945, 0.5864, 0.578, 0.5718, 0.5681, 0.5598, 0.554, 0.5482, 0.5434, 0.5383, 
	0.5306, 0.5265, 0.5191]

digits_inverse_ranking = [505, 732, 1273, 988, 1375, 1264, 502, 1594, 1070, 1271, 
	873, 757, 956, 87, 1576, 1086, 1313, 1043, 919, 163, 566, 1305, 673, 1176, 
	1657, 586, 800, 1572, 1248, 1321, 756, 1165, 1022, 1277, 947, 1754, 327, 447, 
	1293, 900, 1259, 689, 1263, 1778, 629, 889, 998, 1735, 1191, 317, 851, 591, 
	897, 494, 1393, 33, 1012, 1341, 1302, 1094, 1260, 1109, 467, 1023, 1742, 
	283, 1618, 1274, 655, 609, 1215, 1718, 1033, 263, 951, 982, 1635, 413, 1205,
	1600, 771, 1589, 599, 1001, 1017, 1290, 1268, 1009, 517, 488, 1115, 1731, 
	1493, 176, 430, 734, 1708, 1211, 538, 1487]

digits_inverse_gains = [35.1677, 8.8456, 4.5601, 3.2231, 1.7348, 1.3904, 0.8558, 
	0.5118, 0.4719, 0.3108, 0.291, 0.223, 0.2062, 0.1997, 0.159, 0.1267, 0.1138, 
	0.1042, 0.0739, 0.0614, 0.0593, 0.0547, 0.0499, 0.0455, 0.0361, 0.0339, 0.0294, 
	0.0284, 0.0271, 0.0237, 0.021, 0.0187, 0.0176, 0.0171, 0.0166, 0.0149, 0.0139, 
	0.0128, 0.0127, 0.0118, 0.0113, 0.0102, 0.0101, 0.0096, 0.0095, 0.0086, 0.0082, 
	0.0077, 0.0073, 0.0071, 0.0067, 0.0065, 0.0064, 0.0058, 0.0056, 0.0054, 0.0052, 
	0.005, 0.0047, 0.0046, 0.0042, 0.004, 0.004, 0.0037, 0.0036, 0.0035, 0.0034, 
	0.0032, 0.0031, 0.003, 0.0029, 0.0028, 0.0028, 0.0027, 0.0026, 0.0025, 0.0024, 
	0.0023, 0.0023, 0.0022, 0.0021, 0.0021, 0.002, 0.002, 0.0019, 0.0018, 0.0018, 
	0.0018, 0.0018, 0.0017, 0.0017, 0.0016, 0.0016, 0.0016, 0.0015, 0.0015, 0.0014, 
	0.0014, 0.0014, 0.0013]

def test_digits_sqrt_select_biggest_first():
	model = FeatureBasedSelection(10, 'sqrt', 10)
	model.fit(X_digits)
	assert_equal(model.ranking[0], X_digits.sum(axis=1).argmax())

def test_digits_sqrt_select_biggest_first_pq():
	model = FeatureBasedSelection(10, 'sqrt', 1)
	model.fit(X_digits)
	assert_equal(model.ranking[0], X_digits.sum(axis=1).argmax())

def test_digits_log_select_biggest_first():
	model = FeatureBasedSelection(10, 'log', 10)
	model.fit(X_digits)
	assert_equal(model.ranking[0], X_digits.sum(axis=1).argmax())

def test_digits_log_select_biggest_first_pq():
	model = FeatureBasedSelection(10, 'log', 1)
	model.fit(X_digits)
	assert_equal(model.ranking[0], X_digits.sum(axis=1).argmax())

def test_digits_sqrt_small_greedy():
	model = FeatureBasedSelection(10, 'sqrt', 10)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking[:10])
	assert_array_almost_equal(model.gains, digits_sqrt_gains[:10], 4)

def test_digits_sqrt_small_pivot():
	model = FeatureBasedSelection(10, 'sqrt', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking[:10])
	assert_array_almost_equal(model.gains, digits_sqrt_gains[:10], 4)

def test_digits_sqrt_small_pq():
	model = FeatureBasedSelection(10, 'sqrt', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking[:10])
	assert_array_almost_equal(model.gains, digits_sqrt_gains[:10], 4)

def test_digits_sqrt_small_truncated():
	model = FeatureBasedSelection(15, 'sqrt', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_sqrt_ranking[:10])
	assert_array_almost_equal(model.gains[:10], digits_sqrt_gains[:10], 4)

def test_digits_sqrt_small_truncated_pivot():
	model = FeatureBasedSelection(15, 'sqrt', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_sqrt_ranking[:10])
	assert_array_almost_equal(model.gains[:10], digits_sqrt_gains[:10], 4)

def test_digits_sqrt_large_greedy():
	model = FeatureBasedSelection(100, 'sqrt', 100)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_gains, 4)

def test_digits_sqrt_large_pivot():
	model = FeatureBasedSelection(100, 'sqrt', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_gains, 4)

def test_digits_sqrt_large_pq():
	model = FeatureBasedSelection(100, 'sqrt', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_gains, 4)

def test_digits_sqrt_large_truncated():
	model = FeatureBasedSelection(150, 'sqrt', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_sqrt_ranking)
	assert_array_almost_equal(model.gains[:100], digits_sqrt_gains, 4)

def test_digits_sqrt_large_truncated_pivot():
	model = FeatureBasedSelection(150, 'sqrt', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_sqrt_ranking)
	assert_array_almost_equal(model.gains[:100], digits_sqrt_gains, 4)

def test_digits_log_small_greedy():
	model = FeatureBasedSelection(10, 'log', 10)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_log_ranking[:10])
	assert_array_almost_equal(model.gains, digits_log_gains[:10], 4)

def test_digits_log_small_pivot():
	model = FeatureBasedSelection(10, 'log', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_log_ranking[:10])
	assert_array_almost_equal(model.gains, digits_log_gains[:10], 4)

def test_digits_log_small_pq():
	model = FeatureBasedSelection(10, 'log', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_log_ranking[:10])
	assert_array_almost_equal(model.gains, digits_log_gains[:10], 4)

def test_digits_log_small_truncated():
	model = FeatureBasedSelection(15, 'log', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_log_ranking[:10])
	assert_array_almost_equal(model.gains[:10], digits_log_gains[:10], 4)

def test_digits_log_small_truncated_pivot():
	model = FeatureBasedSelection(15, 'log', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_log_ranking[:10])
	assert_array_almost_equal(model.gains[:10], digits_log_gains[:10], 4)

def test_digits_log_large_greedy():
	model = FeatureBasedSelection(100, 'log', 100)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_log_ranking)
	assert_array_almost_equal(model.gains, digits_log_gains, 4)

def test_digits_log_large_pivot():
	model = FeatureBasedSelection(100, 'log', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_log_ranking)
	assert_array_almost_equal(model.gains, digits_log_gains, 4)

def test_digits_log_large_pq():
	model = FeatureBasedSelection(100, 'log', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_log_ranking)
	assert_array_almost_equal(model.gains, digits_log_gains, 4)

def test_digits_log_large_truncated():
	model = FeatureBasedSelection(150, 'log', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_log_ranking)
	assert_array_almost_equal(model.gains[:100], digits_log_gains, 4)

def test_digits_log_large_truncated_pivot():
	model = FeatureBasedSelection(150, 'log', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_log_ranking)
	assert_array_almost_equal(model.gains[:100], digits_log_gains, 4)

def test_digits_inverse_small_greedy():
	model = FeatureBasedSelection(10, 'inverse', 10)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_inverse_ranking[:10])
	assert_array_almost_equal(model.gains, digits_inverse_gains[:10], 4)

def test_digits_inverse_small_pivot():
	model = FeatureBasedSelection(10, 'inverse', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_inverse_ranking[:10])
	assert_array_almost_equal(model.gains, digits_inverse_gains[:10], 4)

def test_digits_inverse_small_pq():
	model = FeatureBasedSelection(10, 'inverse', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_inverse_ranking[:10])
	assert_array_almost_equal(model.gains, digits_inverse_gains[:10], 4)

def test_digits_inverse_small_truncated():
	model = FeatureBasedSelection(15, 'inverse', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_inverse_ranking[:10])
	assert_array_almost_equal(model.gains[:10], digits_inverse_gains[:10], 4)

def test_digits_inverse_small_truncated_pivot():
	model = FeatureBasedSelection(15, 'inverse', 5)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:10], digits_inverse_ranking[:10])
	assert_array_almost_equal(model.gains[:10], digits_inverse_gains[:10], 4)

def test_digits_inverse_large_greedy():
	model = FeatureBasedSelection(100, 'inverse', 100)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_inverse_ranking)
	assert_array_almost_equal(model.gains, digits_inverse_gains, 4)

def test_digits_inverse_large_pivot():
	model = FeatureBasedSelection(100, 'inverse', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_inverse_ranking)
	assert_array_almost_equal(model.gains, digits_inverse_gains, 4)

def test_digits_inverse_large_pq():
	model = FeatureBasedSelection(100, 'inverse', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_inverse_ranking)
	assert_array_almost_equal(model.gains, digits_inverse_gains, 4)

def test_digits_inverse_large_truncated():
	model = FeatureBasedSelection(150, 'inverse', 1)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_inverse_ranking)
	assert_array_almost_equal(model.gains[:100], digits_inverse_gains, 4)

def test_digits_inverse_large_truncated_pivot():
	model = FeatureBasedSelection(150, 'inverse', 50)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:100], digits_inverse_ranking)
	assert_array_almost_equal(model.gains[:100], digits_inverse_gains, 4)

