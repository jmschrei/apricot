import scipy
import numpy

try:
	import cupy
except:
	import numpy as cupy

from apricot import MixtureSelection
from apricot import FeatureBasedSelection
from apricot.optimizers import *

from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

digits_data = load_digits()
X_digits = digits_data.data

X_digits_sparse = scipy.sparse.csr_matrix(X_digits)
X_digits_cupy = cupy.array(X_digits)

digits_ranking = [818, 1296, 732, 988, 629, 951, 1747, 235, 1375, 1205, 1572, 
	1657, 1271, 898, 178, 1766, 591, 160, 513, 1070, 1113, 185, 1017, 1793, 736, 
	283, 491, 538, 919, 423, 688, 163, 1176, 1022, 1493, 1796, 221, 565, 502, 
	208, 1273, 1009, 890, 1043, 313, 1313, 693, 1317, 956, 1086, 756, 263, 1082, 
	33, 586, 854, 1030, 873, 757, 1487, 768, 805, 430, 1393, 1704, 615, 1193, 
	979, 666, 457, 1437, 352, 481, 1342, 1305, 1191, 673, 1021, 424, 407, 1470, 
	851, 1349, 77, 453, 1012, 1668, 786, 372, 87, 1260, 1263, 1186, 655, 168, 
	1106, 436, 548, 500, 1071]

digits_gains = [152.2932, 70.3476, 55.4229, 42.8135, 38.5884, 32.8221, 
	29.7563, 27.2078, 25.8361, 24.3585, 22.8313, 21.6289, 20.6903, 19.7554, 
	18.9661, 18.2044, 17.4879, 17.0986, 16.5219, 16.1346, 15.6849, 15.1929, 
	14.8071, 14.4614, 14.1263, 13.8427, 13.6364, 13.1858, 13.0251, 12.7098, 
	12.5099, 12.2925, 12.1759, 11.9375, 11.8008, 11.5694, 11.4054, 11.2093, 
	11.0464, 10.9119, 10.7966, 10.6753, 10.5141, 10.3925, 10.3039, 10.1926, 
	10.0551, 9.9719, 9.8187, 9.6708, 9.5892, 9.4608, 9.3775, 9.2822, 9.1917, 
	9.0711, 8.9826, 8.9076, 8.847, 8.7676, 8.6784, 8.603, 8.5362, 8.4719, 
	8.3972, 8.3124, 8.252, 8.1715, 8.1113, 8.0456, 7.9853, 7.9312, 7.8523, 
	7.794, 7.7451, 7.6839, 7.6567, 7.5988, 7.5465, 7.4879, 7.4444, 7.3839, 
	7.3437, 7.2931, 7.2339, 7.1909, 7.1453, 7.1032, 7.0634, 7.0083, 6.9775, 
	6.9542, 6.8977, 6.8592, 6.827, 6.7886, 6.7578, 6.7162, 6.6772, 6.6408]

digits_greedi_ranking = [818, 1296, 732, 988, 629, 951, 1747, 235, 1375, 1205, 
	1572, 1657, 1271, 898, 178, 1766, 591, 160, 513, 1070, 1113, 185, 1017, 
	1793, 736, 283, 491, 538, 919, 423, 688, 163, 1176, 1022, 1493, 1796, 221, 
	565, 502, 208, 1273, 1009, 890, 1043, 313, 1313, 693, 1317, 956, 1086, 756, 
	263, 1082, 33, 586, 854, 1030, 873, 757, 1487, 768, 805, 430, 1393, 1704, 
	615, 1193, 979, 666, 457, 1437, 352, 481, 1342, 1305, 1191, 673, 1021, 
	424, 407, 1470, 851, 1349, 77, 453, 1012, 1668, 786, 372, 87, 1260, 1263, 
	1186, 655, 168, 1106, 436, 548, 500, 1071]

digits_greedi_gains = [152.2932, 70.3476, 55.4229, 42.8135, 38.5884, 32.8221, 
	29.7563, 27.2078, 25.8361, 24.3585, 22.8313, 21.6289, 20.6903, 19.7554, 
	18.9661, 18.2044, 17.4879, 17.0986, 16.5219, 16.1346, 15.6849, 15.1929, 
	14.8071, 14.4614, 14.1263, 13.8427, 13.6364, 13.1858, 13.0251, 12.7098, 
	12.5099, 12.2925, 12.1759, 11.9375, 11.8008, 11.5694, 11.4054, 11.2093, 
	11.0464, 10.9119, 10.7966, 10.6753, 10.5141, 10.3925, 10.3039, 10.1926, 
	10.0551, 9.9719, 9.8187, 9.6708, 9.5892, 9.4608, 9.3775, 9.2822, 9.1917, 
	9.0711, 8.9826, 8.9076, 8.847, 8.7676, 8.6784, 8.603, 8.5362, 8.4719,
	8.3972, 8.3124, 8.252, 8.1715, 8.1113, 8.0456, 7.9853, 7.9312, 7.8523, 
	7.794, 7.7451, 7.6839, 7.6567, 7.5988, 7.5465, 7.4879, 7.4444, 7.3839, 
	7.3437, 7.2931, 7.2339, 7.1909, 7.1453, 7.1032, 7.0634, 7.0083, 6.9775, 
	6.9542, 6.8977, 6.8592, 6.827, 6.7886, 6.7578, 6.7162, 6.6772, 6.6408]

digits_approx_ranking = [818, 1296, 732, 1375, 988, 951, 1747, 629, 1572, 
	1793, 1657, 235, 1205, 1273, 898, 1766, 178, 1070, 591, 1271, 513, 185, 
	491, 1493, 1022, 1017, 1113, 736, 263, 919, 423, 1176, 283, 160, 538, 
	1796, 163, 502, 565, 666, 586, 688, 221, 208, 1009, 1313, 313, 1086, 
	1317, 756, 1704, 890, 1043, 693, 1487, 1082, 33, 1030, 615, 956, 430, 
	1012, 1437, 481, 1106, 372, 873, 655, 1260, 77, 1263, 768, 854, 424, 
	1393, 757, 457, 979, 1349, 407, 1781, 1109, 1305, 352, 805, 87, 1186, 
	851, 1342, 459, 1193, 1470, 1191, 453, 451, 317, 168, 786, 673, 1021]

digits_approx_gains = [152.2932, 70.3476, 55.4229, 40.662, 39.006, 33.449, 
	29.9424, 28.0319, 25.7394, 24.0863, 22.9212, 21.7298, 20.6456, 19.3092, 
	19.0731, 18.2018, 17.5479, 16.8794, 16.7182, 15.8711, 15.8008, 15.248, 
	14.7117, 14.2284, 14.0479, 13.8035, 13.8965, 13.4434, 12.695, 12.8843, 
	12.5063, 12.2843, 12.1381, 12.1398, 11.8313, 11.6022, 11.5103, 11.1856, 
	11.1116, 10.7301, 10.7392, 10.8269, 10.6069, 10.453, 10.2373, 10.0994, 
	10.059, 9.8912, 9.8362, 9.6562, 9.479, 9.6119, 9.6617, 9.4232, 9.0999, 
	9.0473, 9.068, 8.9251, 8.7993, 8.8327, 8.7163, 8.4407, 8.4107, 8.4169, 
	8.276, 8.1522, 8.3762, 8.0707, 7.9703, 7.958, 7.9603, 7.9682, 7.9183, 
	7.8082, 7.7928, 7.892, 7.6979, 7.6573, 7.5355, 7.4719, 7.2557, 7.3026, 
	7.3903, 7.3379, 7.2994, 7.1366, 7.1479, 7.0997, 7.1551, 6.976, 7.0948,
	6.974, 6.9205, 6.857, 6.785, 6.8166, 6.7522, 6.7384, 6.8115, 6.6876]

digits_stochastic_ranking = [1081, 1014, 1386, 770, 567, 137, 723, 1491, 
	1274, 1492, 1728, 1456, 186, 1448, 386, 148, 891, 1759, 1424, 587, 191, 
	629, 1507, 1084, 1473, 946, 518, 638, 1739, 502, 1537, 1227, 689, 88, 
	238, 1667, 1785, 1067, 1461, 1222, 1099, 607, 364, 1572, 1195, 217, 
	1034, 208, 84, 1128, 425, 345, 626, 843, 1070, 83, 1449, 1071, 1644, 
	1392, 1415, 449, 802, 1348, 1553, 175, 1455, 1770, 1395, 1032, 879, 
	1220, 1137, 129, 754, 1695, 1459, 782, 549, 1069, 260, 834, 517, 919, 
	1622, 700, 424, 1685, 245, 1339, 1152, 1212, 1425, 937, 1665, 291, 
	1535, 701, 1508, 1219]

digits_stochastic_gains = [121.7429, 38.299, 40.6373, 51.1555, 28.4007, 
	24.8129, 30.5944, 23.784, 29.3665, 22.5999, 23.256, 22.3397, 19.6173, 
	18.9918, 19.0722, 18.4918, 13.6348, 18.2636, 13.1233, 11.306, 13.4296, 
	17.0155, 12.0848, 12.0533, 10.7912, 14.023, 10.762, 11.3215, 12.6178, 
	16.4564, 11.9374, 9.8052, 17.0825, 10.1284, 11.0922, 11.09, 10.5038, 
	11.1906, 9.9223, 8.7334, 7.9894, 8.0544, 9.1596, 15.8808, 8.293, 8.7925, 
	9.8181, 11.2449, 10.4297, 7.3253, 9.0816, 8.5007, 9.5166, 8.1465, 12.6806, 
	6.8592, 7.0207, 9.5185, 6.2303, 7.6187, 6.6266, 6.8528, 7.7183, 8.1687, 
	5.9507, 7.0074, 7.6181, 6.877, 7.9805, 5.9543, 5.9006, 8.5146, 6.2211, 
	5.6803, 7.6504, 8.0842, 6.3355, 6.8525, 6.2785, 7.7865, 5.3526, 7.0893, 
	8.2436, 10.0573, 6.293, 6.8794, 7.7733, 7.0383, 5.9675, 5.2374, 5.5081, 
	5.4276, 5.9783, 6.4971, 5.1889, 5.6313, 5.8053, 6.5889, 5.0918, 5.1209]

digits_sample_ranking = [818, 1296, 732, 988, 629, 951, 1747, 235, 1205, 1313, 
	898, 1657, 283, 1271, 160, 591, 1070, 1766, 178, 1113, 185, 491, 736, 
	1017, 1793, 1022, 221, 1493, 1176, 1009, 919, 163, 1796, 538, 423, 693, 
	208, 890, 502, 565, 1043, 1273, 956, 1317, 263, 313, 1082, 430, 1393, 
	1086, 756, 586, 757, 33, 1030, 805, 873, 1704, 854, 768, 1193, 615, 1191, 
	352, 1487, 673, 979, 1021, 481, 1342, 407, 1305, 424, 1263, 77, 1349, 
	851, 1470, 655, 1668, 786, 1437, 453, 548, 1012, 168, 1186, 1109, 689, 
	372, 87, 1071, 1106, 500, 767, 436, 1576, 172, 451, 317]

digits_sample_gains = [152.2932, 70.3476, 55.4229, 42.8135, 38.5884, 32.8221,
	29.7563, 27.2078, 25.2167, 23.5628, 22.374, 21.4788, 20.4943, 19.8581, 
	18.9381, 18.1915, 17.7252, 17.1965, 16.6098, 16.0786, 15.6007, 15.1446, 
	14.8123, 14.3777, 14.1483, 13.8825, 13.461, 13.2176, 12.9521, 12.7478, 
	12.5743, 12.3747, 12.1752, 11.9979, 11.7362, 11.5816, 11.3711, 11.1938, 
	11.0639, 10.946, 10.7851, 10.7243, 10.4895, 10.3896, 10.3093, 10.125, 
	9.9581, 9.8459, 9.7677, 9.6534, 9.5863, 9.508, 9.3817, 9.2986, 9.198, 
	9.0883, 9.0038, 8.9179, 8.8321, 8.7494, 8.6602, 8.5976, 8.5145, 8.4323, 
	8.3656, 8.3018, 8.2424, 8.1621, 8.0957, 8.0337, 7.9818, 7.9204, 7.8514, 
	7.7973, 7.7334, 7.6862, 7.6332, 7.5786, 7.5323, 7.4613, 7.4109, 7.3694, 
	7.3199, 7.2791, 7.2389, 7.1823, 7.1232, 7.0875, 7.046, 7.0172, 6.9806, 
	6.9359, 6.9055, 6.8493, 6.8181, 6.7789, 6.7478, 6.7228, 6.6745, 6.6534]

digits_modular_ranking = [818, 1766, 491, 178, 768, 185, 513, 1747, 160, 208, 
	423, 898, 1793, 854, 352, 424, 890, 1796, 505, 402, 978, 459, 148, 500, 
	138, 457, 666, 1030, 1342, 693, 370, 509, 417, 235, 452, 1296, 309, 481, 
	615, 578, 1082, 1709, 407, 805, 1186, 55, 168, 1379, 453, 1759, 736, 332, 
	565, 1205, 126, 644, 221, 1021, 1493, 1668, 451, 831, 209, 1276, 420, 396, 
	1009, 1703, 1704, 913, 1071, 1317, 1193, 646, 1069, 1393, 688, 1774, 951, 
	1736, 985, 1349, 1105, 72, 1545, 1113, 1705, 1015, 33, 899, 786, 1620, 
	1470, 1310, 514, 1676, 1106, 439, 1474, 1781]

digits_modular_gains = [152.2932, 59.4281, 54.5905, 39.7922, 32.8763, 29.5361, 
	27.967, 26.596, 23.784, 21.5854, 21.4519, 22.8164, 19.3186, 17.6405, 17.8932, 
	16.4197, 17.0048, 16.5204, 14.8051, 14.3766, 14.3648, 14.0213, 13.0132, 
	13.7872, 12.7561, 16.864, 12.9037, 12.7349, 12.4368, 12.8589, 11.2171, 
	11.2693, 11.082, 12.5391, 10.8932, 13.6737, 10.1224, 10.8944, 10.5785, 
	10.4644, 10.6246, 9.9776, 10.013, 10.164, 9.8172, 9.2902, 9.534, 9.2717, 
	9.3514, 9.222, 10.9156, 8.5255, 9.3958, 10.2975, 8.1919, 8.2635, 9.2393, 
	9.7903, 8.9236, 8.5461, 8.2657, 8.0681, 7.8935, 7.881, 7.6749, 7.7052, 
	11.5209, 7.5021, 8.1381, 7.1416, 7.8203, 7.8894, 7.8344, 7.4424, 7.2331, 
	9.2211, 8.1051, 7.1892, 8.7177, 6.76, 7.7628, 7.3084, 7.1069, 6.7299, 
	6.5033, 8.4939, 6.5589, 6.5227, 8.1911, 6.2681, 6.8672, 6.606, 6.6948, 
	6.7548, 6.0001, 6.5057, 6.5458, 6.1066, 6.5208, 6.5072]

# Test some concave functions

def test_digits_naive():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer='naive')
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking[:30], digits_ranking[:30])
	assert_array_equal(model.ranking[-30:], digits_ranking[-30:])
	assert_array_almost_equal(model.gains, digits_gains, 4)

def test_digits_lazy():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer='lazy')
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking, digits_ranking)
	assert_array_almost_equal(model.gains, digits_gains, 4)

def test_digits_two_stage():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer='two-stage')
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking, digits_ranking)
	assert_array_almost_equal(model.gains, digits_gains, 4)

# Test with initialization

def test_digits_naive_init():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer='naive', initial_subset=digits_ranking[:5])
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking[:20], digits_ranking[5:25])
	assert_array_almost_equal(model.gains[:20], digits_gains[5:25], 4)

def test_digits_lazy_init():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer='lazy', initial_subset=digits_ranking[:5])
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking[:-5], digits_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_gains[5:], 4)

def test_digits_two_stage_init():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer='two-stage', initial_subset=digits_ranking[:5])
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking[:-5], digits_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_gains[5:], 4)

# Test all optimizers

def test_digits_greedi_nn():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer='two-stage', optimizer_kwds={'optimizer1': 'naive', 
		'optimizer2': 'naive'}, random_state=0)
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking[:85], digits_greedi_ranking[:85])
	assert_array_almost_equal(model.gains[:85], digits_greedi_gains[:85], 4)

def test_digits_greedi_ll():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer='two-stage', optimizer_kwds={'optimizer1': 'lazy', 
		'optimizer2': 'lazy'}, random_state=0)
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking[:85], digits_greedi_ranking[:85])
	assert_array_almost_equal(model.gains[:85], digits_greedi_gains[:85], 4)

def test_digits_greedi_ln():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer='two-stage', optimizer_kwds={'optimizer1': 'lazy', 
		'optimizer2': 'naive'}, random_state=0)
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking, digits_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_greedi_gains, 4)

def test_digits_greedi_nl():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer='two-stage', optimizer_kwds={'optimizer1': 'naive', 
		'optimizer2': 'lazy'}, random_state=0)
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking[:85], digits_greedi_ranking[:85])
	assert_array_almost_equal(model.gains[:85], digits_greedi_gains[:85], 4)

def test_digits_approximate():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer='approximate-lazy', random_state=0)
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking, digits_approx_ranking)
	assert_array_almost_equal(model.gains, digits_approx_gains, 4)

def test_digits_stochastic():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer='stochastic', random_state=0)
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking, digits_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_stochastic_gains, 4)

def test_digits_sample():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer='sample', random_state=0)
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking, digits_sample_ranking)
	assert_array_almost_equal(model.gains, digits_sample_gains, 4)

def test_digits_modular():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer='modular', random_state=0)
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking, digits_modular_ranking)
	assert_array_almost_equal(model.gains, digits_modular_gains, 4)


# Using Optimizer Objects

def test_digits_naive_object():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer=NaiveGreedy(random_state=0))
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking, digits_ranking)
	assert_array_almost_equal(model.gains, digits_gains, 4)

def test_digits_lazy_object():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer=LazyGreedy(random_state=0))
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking, digits_ranking)
	assert_array_almost_equal(model.gains, digits_gains, 4)

def test_digits_two_stage_object():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		optimizer=TwoStageGreedy(random_state=0))
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking, digits_ranking)
	assert_array_almost_equal(model.gains, digits_gains, 4)

def test_digits_greedi_nn_object():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer=GreeDi(optimizer1='naive', optimizer2='naive', 
		 	random_state=0))
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking[:85], digits_greedi_ranking[:85])
	assert_array_almost_equal(model.gains[:85], digits_greedi_gains[:85], 4)

def test_digits_greedi_ll_object():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer=GreeDi(optimizer1='lazy', optimizer2='lazy', 
		 	random_state=0))
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking[:85], digits_greedi_ranking[:85])
	assert_array_almost_equal(model.gains[:85], digits_greedi_gains[:85], 4)

def test_digits_greedi_ln_object():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer=GreeDi(optimizer1='lazy', optimizer2='naive', 
		 	random_state=0))
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking[:85], digits_greedi_ranking[:85])
	assert_array_almost_equal(model.gains[:85], digits_greedi_gains[:85], 4)

def test_digits_greedi_nl_object():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer=GreeDi(optimizer1='naive', optimizer2='lazy', 
		 	random_state=0))
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking[:85], digits_greedi_ranking[:85])
	assert_array_almost_equal(model.gains[:85], digits_greedi_gains[:85], 4)

def test_digits_approximate_object():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer=ApproximateLazyGreedy())
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking, digits_approx_ranking)
	assert_array_almost_equal(model.gains, digits_approx_gains, 4)

def test_digits_stochastic_object():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer=StochasticGreedy(random_state=0))
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking, digits_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_stochastic_gains, 4)

def test_digits_sample_object():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer=SampleGreedy(random_state=0))
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking, digits_sample_ranking)
	assert_array_almost_equal(model.gains, digits_sample_gains, 4)

def test_digits_modular_object():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer=ModularGreedy(random_state=0))
	model.fit(X_digits_cupy)
	assert_array_equal(model.ranking, digits_modular_ranking)
	assert_array_almost_equal(model.gains, digits_modular_gains, 4)

# Test all optimizers on sparse data

def test_digits_naive_sparse():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer='naive')
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_ranking)
	assert_array_almost_equal(model.gains, digits_gains, 4)

def test_digits_lazy_sparse():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer='lazy')
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_ranking)
	assert_array_almost_equal(model.gains, digits_gains, 4)

def test_digits_two_stage_sparse():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer='two-stage')
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_ranking)
	assert_array_almost_equal(model.gains, digits_gains, 4)

def test_digits_greedi_nn_sparse():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer='greedi', optimizer_kwds={'optimizer1': 'naive', 
		 'optimizer2': 'naive'}, random_state=0)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking[:85], digits_greedi_ranking[:85])
	assert_array_almost_equal(model.gains[:85], digits_greedi_gains[:85], 4)

def test_digits_greedi_ll_sparse():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer='greedi', optimizer_kwds={'optimizer1': 'lazy', 
		 'optimizer2': 'lazy'}, random_state=0)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking[:85], digits_greedi_ranking[:85])
	assert_array_almost_equal(model.gains[:85], digits_greedi_gains[:85], 4)

def test_digits_greedi_ln_sparse():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer='greedi', optimizer_kwds={'optimizer1': 'lazy', 
		 'optimizer2': 'naive'}, random_state=0)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking[:85], digits_greedi_ranking[:85])
	assert_array_almost_equal(model.gains[:85], digits_greedi_gains[:85], 4)

def test_digits_greedi_nl_sparse():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer='greedi', optimizer_kwds={'optimizer1': 'naive', 
		 'optimizer2': 'lazy'}, random_state=0)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking[:85], digits_greedi_ranking[:85])
	assert_array_almost_equal(model.gains[:85], digits_greedi_gains[:85], 4)

def test_digits_approximate_sparse():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer='approximate-lazy', random_state=0)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_approx_ranking)
	assert_array_almost_equal(model.gains, digits_approx_gains, 4)

def test_digits_stochastic_sparse():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer='stochastic', random_state=0)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_stochastic_gains, 4)

def test_digits_sample_sparse():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer='sample', random_state=0)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_sample_ranking)
	assert_array_almost_equal(model.gains, digits_sample_gains, 4)

def test_digits_modular_sparse():
	model1 = FeatureBasedSelection(100, 'sqrt')
	model2 = FeatureBasedSelection(100, 'log')
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		 optimizer='modular', random_state=0)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_modular_ranking)
	assert_array_almost_equal(model.gains, digits_modular_gains, 4)
