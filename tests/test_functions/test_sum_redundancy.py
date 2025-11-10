import scipy
import numpy

try:
	import cupy
except:
	import numpy as cupy

from apricot import SumRedundancySelection
from apricot.optimizers import NaiveGreedy, LazyGreedy, TwoStageGreedy, GreeDi, StochasticGreedy, SampleGreedy, ModularGreedy

from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

digits_data = load_digits()
X_digits = digits_data.data

X_digits_cosine_sparse = scipy.sparse.csr_matrix((1 - pairwise_distances(
	X_digits, metric='cosine')) ** 2)
X_digits_corr_cupy = cupy.array((1 - pairwise_distances(
	X_digits, metric='correlation')) ** 2)
X_digits_cosine_cupy = cupy.array((1 - pairwise_distances(
	X_digits, metric='cosine')) ** 2)

# fmt: off
digits_corr_ranking = [0, 1631, 1308, 1259, 1024, 673, 734, 1595, 75, 1078, 
	958, 447, 1195, 1221, 576, 1660, 998, 1274, 632, 1671, 1572, 1180, 1742, 
	1779, 713, 1514, 813, 988, 1576, 757, 1000, 1025, 1419, 1499, 133, 1551, 
	1580, 1316, 689, 1589, 756, 1462, 889, 413, 599, 1311, 1275, 1163, 54, 
	1657, 1400, 1001, 1264, 1495, 1302, 1154, 1473, 317, 1488, 194, 527, 1283, 
	675, 1604, 1077, 215, 766, 946, 1165, 403, 1115, 972, 1079, 1754, 914, 617, 
	57, 1182, 751, 1717, 876, 31, 103, 941, 1467, 218, 1440, 637, 982, 1708, 
	1272, 7, 966, 981, 163, 1562, 440, 341, 1301, 1216]

digits_corr_gains = [-1.0, -1.0053, -1.2226, -1.2786, -1.5351, -1.5826, 
	-1.7436, -2.2005, -2.2884, -2.5087, -2.7581, -2.8347, -3.0216, -3.3001, 
	-3.3921, -4.1251, -4.2028, -4.4885, -4.8339, -4.8857, -5.1816, -5.3797, 
	-5.7385, -6.03, -6.0674, -6.4577, -6.8554, -6.9494, -7.0362, -7.4869, 
	-7.6198, -8.117, -8.5624, -8.7402, -8.9086, -9.2393, -9.372, -9.7991, 
	-10.003, -10.3158, -10.7446, -10.8657, -11.3165, -11.518, -11.8186, 
	-11.8664, -12.24, -12.5102, -12.865, -13.1717, -13.5104, -13.611, -14.0063, 
	-14.3125, -14.4315, -14.9093, -15.4115, -15.5949, -15.8784, -15.9565, 
	-16.2537, -16.4321, -16.8201, -17.01, -17.5617, -17.6755, -18.0406, 
	-18.2885, -19.0408, -19.1925, -19.3148, -19.7779, -19.8668, -20.3401, 
	-20.4092, -21.0325, -21.2272, -21.472, -21.9653, -22.1756, -22.5028, 
	-22.8417, -23.1781, -23.34, -23.5919, -24.1087, -24.2592, -24.4434, 
	-24.7991, -25.0295, -25.2185, -25.8405, -26.3942, -26.5413, -26.8281, 
	-27.0728, -27.2003, -27.7439, -27.9688, -28.2785]

digits_euclidean_ranking = [0, 623, 163, 1572, 215, 241, 988, 1635, 1505, 766, 
	673, 67, 1296, 283, 947, 1589, 172, 832, 1001, 1259, 1495, 1111, 919, 1576, 
	141, 1710, 1437, 1106, 1308, 1595, 982, 317, 1274, 131, 1375, 204, 998, 788, 
	732, 756, 1462, 916, 155, 1086, 77, 851, 1574, 680, 263, 1436, 1205, 609, 
	1660, 1221, 1275, 1035, 1514, 953, 1754, 1604, 853, 1562, 1094, 1457, 1264, 
	1685, 218, 447, 1492, 689, 951, 235, 1272, 734, 1657, 494, 517, 1317, 210, 
	629, 1467, 1587, 1290, 926, 1735, 565, 1419, 1522, 801, 660, 1499, 1220, 
	757, 1000, 312, 1470, 135, 1070, 889, 1426]

digits_euclidean_gains = [-5935.0, -9777.0, -13543.0, -17943.0, -22857.0, 
	-27221.0, -33987.0, -37289.0, -42655.0, -44649.0, -50719.0, -54275.0, 
	-61769.0, -67425.0, -69983.0, -75243.0, -78809.0, -86093.0, -91183.0, 
	-95109.0, -100791.0, -106791.0, -111741.0, -116985.0, -121935.0, 
	-128081.0, -132819.0, -138007.0, -142801.0, -148589.0, -153975.0, 
	-157323.0, -164121.0, -168921.0, -174335.0, -179963.0, -184405.0, 
	-190255.0, -196089.0, -201213.0, -206047.0, -210323.0, -215441.0, 
	-221839.0, -227461.0, -233779.0, -238253.0, -243317.0, -248155.0, 
	-254819.0, -259613.0, -263847.0, -270369.0, -274331.0, -281473.0, 
	-286875.0, -292657.0, -296343.0, -303373.0, -308247.0, -314769.0, 
	-317825.0, -324991.0, -330459.0, -336565.0, -342469.0, -347659.0, 
	-353903.0, -358113.0, -365203.0, -370033.0, -376423.0, -379817.0, 
	-386027.0, -391541.0, -397755.0, -402413.0, -406833.0, -413851.0, 
	-420295.0, -425313.0, -430743.0, -437069.0, -442009.0, -445999.0, 
	-453759.0, -460717.0, -467225.0, -470579.0, -478611.0, -481381.0, 
	-486279.0, -492717.0, -501067.0, -507697.0, -511129.0, -517415.0, 
	-523749.0, -527895.0, -533187.0]

digits_cosine_ranking = [0, 1626, 1308, 1259, 1078, 673, 1595, 958, 75, 
	447, 1660, 1221, 1180, 734, 1589, 1462, 632, 1671, 1779, 1717, 133, 
	966, 914, 1274, 1514, 1000, 1035, 1580, 1373, 1311, 403, 1272, 946, 
	766, 215, 1611, 813, 876, 1572, 1316, 1631, 713, 1467, 1499, 1508, 
	941, 1264, 1275, 194, 1024, 527, 1079, 998, 1473, 1182, 990, 131, 
	1585, 204, 617, 675, 67, 1001, 1200, 31, 413, 341, 1025, 1213, 1301, 
	1576, 1400, 218, 317, 442, 1734, 192, 1551, 801, 1495, 1708, 889, 561, 
	1604, 4, 751, 704, 1314, 994, 155, 151, 981, 1125, 498, 57, 1480, 1742, 
	1562, 1195, 988]

digits_cosine_gains = [-1.0, -1.2608, -1.8589, -2.2945, -2.9785, -3.273, 
	-4.0554, -4.5342, -5.1995, -5.9721, -6.2496, -6.8113, -7.5169, -8.1488, 
	-9.0023, -9.2591, -10.006, -10.618, -11.3895, -11.7181, -12.7751, -13.2362, 
	-14.0317, -14.7067, -15.1397, -15.7079, -16.3536, -17.0946, -17.5568, 
	-18.2507, -19.4527, -20.1545, -20.7871, -21.4178, -21.7413, -22.5283, 
	-23.3034, -23.7622, -24.2979, -25.3123, -25.7659, -26.4916, -27.6042, 
	-28.0639, -28.5733, -29.3231, -30.0786, -30.4752, -31.2483, -31.8044, 
	-32.8744, -33.6345, -34.0621, -34.7931, -35.4228, -36.1701, -36.9265, 
	-37.5654, -38.2502, -38.9728, -39.8254, -40.4355, -41.1174, -41.7559, 
	-42.4764, -43.3304, -43.9796, -44.5423, -45.3739, -46.0711, -46.7337, 
	-47.2139, -47.7222, -48.8554, -49.4269, -50.3968, -51.0522, -51.5147, 
	-52.5101, -52.9929, -53.7409, -54.115, -55.3053, -55.9741, -56.6847, 
	-57.6243, -58.176, -58.5817, -59.7514, -60.3535, -60.7568, -61.6063, 
	-62.5969, -63.1069, -64.0203, -64.638, -65.5255, -66.1212, -66.8677, 
	-67.4475]

digits_cosine_greedi_ranking = [16, 617, 1660, 673, 1000, 1589, 1308, 1221, 
	1595, 1078, 447, 1626, 1514, 1259, 632, 133, 1671, 1717, 914, 75, 946, 
	734, 1779, 766, 1462, 1580, 1272, 1473, 1311, 1611, 1180, 1572, 876, 1024, 
	1631, 1274, 215, 1035, 403, 958, 1316, 966, 1182, 1373, 1508, 1025, 1467, 
	1499, 194, 1275, 1264, 813, 998, 67, 527, 1079, 131, 204, 990, 1585, 675, 
	713, 1200, 1301, 413, 192, 1001, 1400, 1576, 218, 1213, 981, 1734, 31, 341, 
	317, 1495, 889, 1708, 801, 1125, 442, 751, 1059, 1314, 4, 12, 1604, 1562, 
	704, 107, 561, 163, 988, 498, 1551, 1195, 1480, 994, 151]

digits_cosine_greedi_gains = [-1.0, -1.3478, -1.9394, -2.3069, -3.1291, -3.4523, 
	-3.811, -4.7084, -5.3596, -5.7415, -6.1306, -6.9337, -7.5382, -7.9192, 
	-8.7388, -9.5881, -10.0552, -10.5168, -11.4115, -12.0015, -12.4559, -13.359, 
	-13.957, -14.6866, -14.849, -15.9155, -16.5696, -17.298, -18.0227, -18.5558, 
	-19.1504, -19.9612, -20.7278, -21.2489, -22.0421, -22.5731, -23.1782, -23.8589, 
	-24.283, -25.1361, -25.8376, -26.5521, -27.0485, -28.0133, -28.5404, -29.4877, 
	-30.0386, -30.6349, -31.0334, -32.2181, -32.7685, -33.3918, -34.193, -34.753, 
	-35.6558, -36.0367, -36.7902, -37.8717, -38.4027, -38.9991, -39.478, -40.4198, 
	-41.1552, -41.7612, -42.363, -43.2004, -44.1362, -44.6401, -45.3337, -45.9022, 
	-46.5817, -47.0738, -48.3477, -48.9314, -49.5341, -50.1541, -51.1846, -51.5936, 
	-52.3633, -52.9651, -53.8957, -54.671, -55.369, -55.8204, -56.6435, -57.4165, 
	-58.2005, -59.0945, -59.66, -60.1562, -61.0494, -62.0454, -62.5839, -63.0563, 
	-64.1159, -64.557, -65.2267, -66.0529, -66.849, -67.6248]

digits_cosine_stochastic_ranking = [1081, 1014, 1386, 770, 567, 137, 723, 1491, 
	1274, 1492, 1728, 1456, 186, 1448, 386, 148, 891, 1759, 1424, 587, 191, 629, 
	1507, 1084, 1473, 946, 518, 638, 1739, 502, 1537, 1227, 689, 88, 238, 1667, 
	1785, 1067, 1461, 1222, 1099, 607, 364, 1572, 1195, 217, 1034, 208, 84, 
	1128, 425, 345, 626, 843, 1070, 83, 1449, 1071, 1644, 1392, 1415, 449, 802, 
	1348, 1553, 175, 1455, 1770, 1395, 1032, 879, 1220, 1137, 129, 754, 1695, 
	1459, 782, 549, 1069, 260, 834, 517, 919, 1622, 700, 424, 1685, 245, 1339, 
	1152, 1212, 1425, 937, 1665, 291, 1535, 701, 1508, 1219]

digits_cosine_stochastic_gains = [-1.0, -2.5269, -2.7995, -3.2931, -5.5216, 
	-6.0637, -7.7541, -8.1408, -7.2975, -10.4072, -11.2111, -10.3373, -10.6228, 
	-12.2424, -13.1709, -18.8132, -14.5518, -19.0894, -18.2777, -18.8472, 
	-17.3115, -23.1732, -23.7351, -26.0553, -18.8892, -17.7293, -29.2051, 
	-23.7703, -27.484, -27.9689, -34.4357, -33.4325, -23.4861, -31.3646, 
	-30.627, -35.9673, -36.9382, -38.8354, -37.5974, -33.8208, -38.1725, 
	-43.8267, -39.2434, -31.1408, -34.4416, -44.662, -43.0794, -51.7522, 
	-45.9076, -45.5261, -53.8562, -54.4264, -51.2669, -53.0412, -44.3672, 
	-52.5772, -53.4106, -66.2475, -61.2402, -58.9847, -59.9399, -65.6418, 
	-67.2847, -61.22, -63.0781, -66.7774, -76.5662, -67.3135, -69.2438, 
	-66.9263, -77.0899, -69.448, -57.6908, -71.3056, -69.0238, -82.4863, 
	-82.4559, -78.3968, -79.779, -96.6816, -69.961, -84.4731, -70.0465, 
	-69.5608, -81.6036, -94.3322, -109.4369, -80.056, -72.2626, -79.6127, 
	-80.3194, -86.0693, -95.2923, -89.2698, -82.4462, -84.8242, -99.9732, 
	-102.3375, -72.4036, -91.7773]

digits_cosine_sample_ranking = [0, 1626, 1308, 1589, 673, 734, 914, 75, 1660, 
	447, 1717, 1671, 1078, 133, 1221, 958, 1514, 1373, 1580, 766, 966, 1272, 
	1462, 1035, 1275, 1311, 632, 1180, 1631, 1576, 946, 1024, 215, 1274, 403, 
	527, 876, 1611, 617, 813, 998, 1499, 1508, 1079, 1473, 194, 317, 1264, 
	1316, 67, 1562, 1585, 204, 990, 675, 1314, 1301, 889, 218, 941, 413, 994, 
	1604, 4, 1025, 801, 341, 31, 131, 704, 442, 689, 155, 192, 751, 1163, 1400, 
	1552, 57, 1731, 561, 151, 163, 1734, 1648, 1495, 1587, 498, 972, 728, 1195, 
	988, 1125, 1304, 800, 1721, 7, 19, 1754, 981]

digits_cosine_sample_gains = [-1.0, -1.2608, -1.8589, -2.3718, -2.9235, 
	-3.3502, -4.1682, -4.7961, -5.3775, -5.8901, -6.4477, -6.8991, -7.7067, 
	-8.1207, -8.9752, -9.6247, -10.2187, -10.4458, -11.617, -12.2125, -12.9059, 
	-13.5264, -14.323, -14.9052, -15.3206, -16.1262, -16.5538, -17.6517, 
	-18.1862, -18.9705, -19.4022, -20.3438, -20.9218, -21.26, -22.1514, 
	-22.8178, -23.5192, -24.0593, -24.7916, -25.6778, -26.448, -27.0072, 
	-27.5813, -28.1622, -28.9791, -29.6623, -30.212, -30.8247, -31.4657, 
	-32.7805, -33.1168, -33.913, -34.5045, -35.0171, -35.7662, -36.9909, 
	-37.4624, -38.2663, -38.7272, -39.5656, -40.1986, -40.7718, -41.6487, 
	-42.6081, -43.0435, -43.8426, -44.2991, -45.0666, -45.9207, -46.4336, 
	-47.4731, -47.8693, -48.9462, -49.5734, -50.278, -51.0253, -51.7427, 
	-52.3166, -53.3424, -54.1884, -54.6391, -55.5147, -55.9356, -56.7033, 
	-57.3959, -58.1553, -58.8327, -59.6321, -60.1963, -60.9938, -61.5477, 
	-62.3124, -62.7948, -63.7351, -64.9801, -65.5436, -66.2565, -66.9823, 
	-67.8899, -68.5351]

digits_cosine_modular_ranking = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
	10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
	20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
	30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
	40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
	50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
	60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
	70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
	80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
	90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

digits_cosine_modular_gains = [-1.0, -1.5389, -3.0365, -3.5809, -4.1258, -6.3708, -7.4814, -5.2822, -9.816, -10.1087,
	-11.2945, -12.0195, -10.2326, -14.4426, -15.4059, -13.5579, -15.712, -18.8125, -18.1894, -16.5568,
	-21.4572, -22.6813, -20.5902, -21.7776, -22.6762, -22.7138, -28.8823, -26.2444, -31.125, -29.2979,
	-27.6456, -27.191, -32.4208, -35.4842, -33.8871, -37.3775, -40.0967, -36.2334, -36.487, -41.201,
	-46.9144, -44.8115, -41.252, -40.5032, -41.1934, -44.3236, -47.6659, -46.5503, -47.7433, -48.3655,
	-42.9863, -45.7986, -52.5101, -57.7592, -46.5784, -63.3521, -56.248, -52.9443, -56.6673, -59.9108,
	-57.0929, -66.185, -66.4162, -69.7889, -63.3601, -60.7774, -64.9483, -53.2438, -64.4567, -63.1722,
	-69.1823, -66.228, -78.5513, -71.3569, -84.6291, -60.0869, -84.3693, -69.2598, -71.6762, -79.8334,
	-79.2383, -85.0744, -84.2064, -79.5252, -75.8808, -84.791, -80.2349, -86.4289, -89.9171, -89.6284,
	-91.0734, -95.421, -94.9176, -97.0423, -93.7396, -101.5086, -100.2028, -99.8039, -100.3847, -102.4461]
# fmt: on

# Test some similarity functions

def test_digits_euclidean_naive():
	model = SumRedundancySelection(100, 'euclidean', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_lazy():
	model = SumRedundancySelection(100, 'euclidean', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_two_stage():
	model = SumRedundancySelection(100, 'euclidean', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_naive():
	model = SumRedundancySelection(100, 'corr', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_lazy():
	model = SumRedundancySelection(100, 'corr', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_two_stage():
	model = SumRedundancySelection(100, 'corr', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_naive():
	model = SumRedundancySelection(100, 'cosine', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_lazy():
	model = SumRedundancySelection(100, 'cosine', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_two_stage():
	model = SumRedundancySelection(100, 'cosine', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_precomputed_naive():
	model = SumRedundancySelection(100, 'precomputed', optimizer='naive')
	model.fit(X_digits_corr_cupy)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_precomputed_lazy():
	model = SumRedundancySelection(100, 'precomputed', optimizer='lazy')
	model.fit(X_digits_corr_cupy)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_precomputed_two_stage():
	model = SumRedundancySelection(100, 'precomputed', optimizer='two-stage')
	model.fit(X_digits_corr_cupy)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

# Test with initialization

def test_digits_euclidean_naive_init():
	model = SumRedundancySelection(100, 'euclidean', optimizer='naive', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:20], digits_euclidean_ranking[5:25])
	assert_array_almost_equal(model.gains[:20], digits_euclidean_gains[5:25], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_lazy_init():
	model = SumRedundancySelection(100, 'euclidean', optimizer='lazy', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_euclidean_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_euclidean_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_two_stage_init():
	model = SumRedundancySelection(100, 'euclidean', optimizer='two-stage', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_euclidean_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_euclidean_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_naive_init():
	model = SumRedundancySelection(100, 'corr', optimizer='naive', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_lazy_init():
	model = SumRedundancySelection(100, 'corr', optimizer='lazy', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_two_stage_init():
	model = SumRedundancySelection(100, 'corr', optimizer='two-stage', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_naive_init():
	model = SumRedundancySelection(100, 'cosine', optimizer='naive', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_lazy_init():
	model = SumRedundancySelection(100, 'cosine', optimizer='lazy', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_two_stage_init():
	model = SumRedundancySelection(100, 'cosine', optimizer='two-stage', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_precomputed_naive_init():
	model = SumRedundancySelection(100, 'precomputed', optimizer='naive', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_precomputed_lazy_init():
	model = SumRedundancySelection(100, 'precomputed', optimizer='lazy', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_precomputed_two_stage_init():
	model = SumRedundancySelection(100, 'precomputed', optimizer='two-stage', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

# Test all optimizers

def test_digits_cosine_greedi_nn():
	model = SumRedundancySelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ll():
	model = SumRedundancySelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ln():
	model = SumRedundancySelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_nl():
	model = SumRedundancySelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_approximate():
	pass
	#model = SumRedundancySelection(100, 'cosine', optimizer='approximate-lazy')
	#model.fit(X_digits)
	#assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	#assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)

def test_digits_cosine_stochastic():
	model = SumRedundancySelection(100, 'cosine', optimizer='stochastic',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_sample():
	model = SumRedundancySelection(100, 'cosine', optimizer='sample',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_modular():
	model = SumRedundancySelection(100, 'cosine', optimizer='modular',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

# Using the partial_fit method

def test_digits_cosine_sieve_batch():
	return
	# model = SumRedundancySelection(100, 'cosine', random_state=0, 
	# 	reservoir=X_digits)
	# model.partial_fit(X_digits)
	# print("[" + ", ".join(map(str, model.ranking)) + "]")
	# print("[" + ", ".join([str(round(gain, 4)) for gain in model.gains]) + "]")
	# assert_array_equal(model.ranking, digits_cosine_sieve_ranking)
	# assert_array_almost_equal(model.gains, digits_cosine_sieve_gains, 4)
	# assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_sieve_minibatch():
	return
	# model = SumRedundancySelection(100, 'cosine', random_state=0, 
	# 	reservoir=X_digits)
	# model.partial_fit(X_digits[:300])
	# model.partial_fit(X_digits[300:500])
	# model.partial_fit(X_digits[500:])
	# assert_array_equal(model.ranking, digits_cosine_sieve_ranking)
	# assert_array_almost_equal(model.gains, digits_cosine_sieve_gains, 4)
	# assert_array_almost_equal(model.subset, X_digits[model.ranking])

# Using Optimizer Objects

def test_digits_cosine_naive_object():
	model = SumRedundancySelection(100, 'cosine', optimizer=NaiveGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_lazy_object():
	model = SumRedundancySelection(100, 'cosine', optimizer=LazyGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_two_stage_object():
	model = SumRedundancySelection(100, 'cosine', optimizer=TwoStageGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_nn_object():
	model = SumRedundancySelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='naive', optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ll_object():
	model = SumRedundancySelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='lazy', optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ln_object():
	model = SumRedundancySelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='lazy', optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_nl_object():
	model = SumRedundancySelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='naive', optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_approximate_object():
	#model = SumRedundancySelection(100, 'cosine', 
	#	optimizer=ApproximateLazyGreedy())
	#model.fit(X_digits)
	#assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	#assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)
	pass

def test_digits_cosine_stochastic_object():
	model = SumRedundancySelection(100, 'cosine', 
		optimizer=StochasticGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_sample_object():
	model = SumRedundancySelection(100, 'cosine', 
		optimizer=SampleGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_modular_object():
	model = SumRedundancySelection(100, 'cosine', 
		optimizer=ModularGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

# Test all optimizers on sparse data

def test_digits_cosine_naive_sparse():
	model = SumRedundancySelection(100, 'precomputed', optimizer='naive')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_lazy_sparse():
	model = SumRedundancySelection(100, 'precomputed', optimizer='lazy')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_two_stage_sparse():
	model = SumRedundancySelection(100, 'precomputed', optimizer='two-stage')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_greedi_nn_sparse():
	model = SumRedundancySelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_ll_sparse():
	model = SumRedundancySelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_ln_sparse():
	model = SumRedundancySelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_nl_sparse():
	model = SumRedundancySelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_approximate_sparse():
	pass
	# Approximate doesn't work with supermodular functions.

	#model = SumRedundancySelection(100, 'precomputed', optimizer='approximate-lazy')
	#model.fit(X_digits_cosine_sparse)
	#assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	#assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)

def test_digits_cosine_stochastic_sparse():
	model = SumRedundancySelection(100, 'precomputed', optimizer='stochastic',
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)

def test_digits_cosine_sample_sparse():
	model = SumRedundancySelection(100, 'precomputed', optimizer='sample',
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)

def test_digits_sqrt_modular_sparse():
	model = SumRedundancySelection(100, 'precomputed', optimizer='modular',
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)
