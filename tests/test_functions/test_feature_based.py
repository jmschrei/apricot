import scipy
import numpy

from apricot import FeatureBasedSelection
from apricot.optimizers import *

from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

#	print("[" + ", ".join(map(str, model.ranking)) + "]")
#	print("[" + ", ".join([str(round(gain, 4)) for gain in model.gains]) + "]")

digits_data = load_digits()
X_digits = digits_data.data
X_digits_sparse = scipy.sparse.csr_matrix(X_digits)

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

digits_sigmoid_ranking = [505, 732, 1273, 988, 1375, 1264, 502, 1594, 1070, 1271, 
	873, 757, 956, 87, 1576, 1086, 1313, 1043, 919, 163, 566, 1305, 673, 1176, 
	1657, 586, 800, 1572, 1248, 1321, 756, 1165, 1022, 1277, 947, 1754, 327, 447, 
	1293, 900, 1259, 689, 1263, 1778, 629, 889, 998, 1735, 1191, 317, 851, 591, 
	897, 494, 1393, 33, 1012, 1341, 1302, 1094, 1260, 1109, 467, 1023, 1742, 
	283, 1618, 1274, 655, 609, 1215, 1718, 1033, 263, 951, 982, 1635, 413, 1205,
	1600, 771, 1589, 599, 1001, 1017, 1290, 1268, 1009, 517, 488, 1115, 1731, 
	1493, 176, 430, 734, 1708, 1211, 538, 1487]

digits_sigmoid_gains = [35.1677, 8.8456, 4.5601, 3.2231, 1.7348, 1.3904, 0.8558, 
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

digits_sqrt_greedi_ranking = [818, 1296, 732, 988, 629, 1747, 951, 235, 1375, 
	1205, 1572, 1766, 178, 1657, 898, 1271, 513, 591, 160, 736, 1070, 185, 1113, 
	491, 1793, 1017, 283, 221, 1493, 688, 538, 423, 919, 1796, 163, 1022, 1176, 
	208, 890, 565, 693, 313, 1009, 1317, 956, 502, 1043, 1082, 1273, 1313, 1030, 
	33, 756, 263, 768, 1487, 586, 1086, 854, 430, 615, 805, 1193, 1393, 1704, 
	352, 757, 873, 666, 979, 457, 424, 481, 1437, 1342, 407, 1349, 1668, 1470, 
	1021, 77, 1191, 673, 1305, 453, 786, 851, 1186, 168, 500, 1012, 548, 1071, 
	451, 1260, 436, 1106, 655, 372, 1105]

digits_sqrt_greedi_gains = [124.8187, 59.9654, 47.7603, 37.88, 34.3933, 29.819, 
	27.4785, 25.2363, 23.6127, 22.6002, 21.1175, 20.1914, 19.1849, 18.5015, 
	17.9476, 17.3043, 16.6255, 16.0984, 15.7533, 15.1851, 14.8136, 14.4975, 
	14.1907, 13.8717, 13.4887, 13.2287, 13.0037, 12.6406, 12.4373, 12.1814, 
	11.9999, 11.7982, 11.6161, 11.4169, 11.2585, 11.1032, 10.9829, 10.8366, 
	10.6663, 10.5211, 10.4042, 10.2704, 10.1146, 10.0116, 9.8487, 9.7181, 
	9.6466, 9.529, 9.4284, 9.3388, 9.2417, 9.1192, 9.0285, 8.9639, 8.8704, 
	8.7846, 8.7234, 8.6348, 8.5746, 8.4954, 8.4284, 8.349, 8.28, 8.1906, 
	8.1304, 8.0608, 8.0066, 7.9458, 7.8825, 7.8351, 7.7756, 7.7179, 7.6538, 
	7.591, 7.544, 7.491, 7.4324, 7.3709, 7.3239, 7.2695, 7.2282, 7.1841, 7.1514, 
	7.1167, 7.0633, 7.009, 6.9693, 6.9245, 6.8838, 6.8409, 6.8031, 6.7691, 
	6.7295, 6.6905, 6.655, 6.6156, 6.5894, 6.5419, 6.5137, 6.4719]

digits_sqrt_approx_ranking = [818, 1296, 732, 951, 988, 629, 1747, 1375, 1572, 
	1793, 1657, 235, 1205, 491, 898, 1273, 1766, 1022, 1493, 956, 160, 185, 
	1086, 423, 1070, 178, 736, 513, 591, 163, 565, 263, 1113, 688, 221, 890, 
	208, 538, 1313, 1317, 693, 283, 502, 919, 1017, 1796, 1030, 1271, 313, 
	1043, 1176, 1009, 1470, 854, 1082, 756, 352, 548, 1193, 768, 615, 805, 
	33, 757, 586, 786, 979, 689, 1106, 1437, 1393, 430, 666, 457, 1487, 424, 
	407, 873, 481, 1021, 1349, 1709, 851, 1191, 1305, 1088, 929, 317, 1704, 
	1342, 87, 1668, 1260, 500, 453, 168, 451, 1111, 402, 1071]

digits_sqrt_approx_gains = [124.8187, 59.9654, 47.7603, 36.5877, 34.585, 
	30.8293, 27.5687, 24.7392, 23.6423, 22.5006, 21.0966, 20.4486, 19.3707, 
	18.284, 17.9394, 16.808, 16.814, 15.608, 15.5028, 14.5844, 14.9441, 
	14.4841, 13.8934, 13.5299, 13.507, 13.2489, 13.3447, 12.923, 12.684, 
	12.3187, 11.8248, 11.7822, 11.8639, 11.6505, 11.3763, 11.1356, 11.0365, 
	10.7705, 10.6115, 10.4958, 10.4803, 10.3406, 10.0809, 10.2725, 10.0921, 
	9.8785, 9.5831, 9.5634, 9.4422, 9.4675, 9.2119, 9.2461, 9.0002, 8.9668, 
	8.9111, 8.8331, 8.6894, 8.4808, 8.4961, 8.4728, 8.433, 8.3223, 8.3195, 
	8.1904, 8.2816, 7.9816, 7.9634, 7.6974, 7.8092, 7.7831, 7.8066, 7.757, 
	7.6813, 7.607, 7.5631, 7.5253, 7.4628, 7.5083, 7.358, 7.2888, 7.2419, 
	7.1631, 7.1315, 7.1121, 7.0837, 6.796, 6.7103, 6.7474, 6.985, 6.8964, 
	6.6507, 6.809, 6.7182, 6.7353, 6.7056, 6.6408, 6.6041, 6.5328, 6.4955, 
	6.4845]

digits_sqrt_stochastic_ranking = [1081, 1014, 1386, 770, 567, 137, 723, 1491, 
	1274, 1492, 1728, 1456, 186, 1448, 386, 148, 891, 1759, 1424, 587, 191, 
	629, 1507, 1084, 1473, 946, 518, 638, 1739, 502, 1537, 1227, 689, 88, 238, 
	1667, 1785, 1067, 1461, 1222, 1099, 607, 364, 1572, 1195, 217, 1034, 208, 
	84, 1128, 425, 345, 626, 843, 1070, 83, 1449, 1071, 1644, 1392, 1415, 449, 
	802, 1348, 1553, 175, 1455, 1770, 1395, 1032, 879, 1220, 1137, 129, 754, 
	1695, 1459, 782, 549, 1069, 260, 834, 517, 919, 1622, 700, 424, 1685, 
	245, 1339, 1152, 1212, 1425, 937, 1665, 291, 1535, 701, 1508, 1219]

digits_sqrt_stochastic_gains = [99.7066, 33.2522, 35.1028, 43.8482, 25.3456, 
	22.3207, 27.4697, 21.5974, 26.0808, 20.7894, 21.1324, 20.6968, 18.3521, 
	17.6346, 17.864, 17.4868, 12.7709, 17.2483, 12.4284, 10.7492, 12.7326, 
	15.9778, 11.509, 11.4864, 10.2599, 12.9274, 10.2816, 10.7871, 12.043, 
	15.0108, 11.4361, 9.4542, 15.5201, 9.7471, 10.6219, 10.6349, 10.1038, 
	10.7344, 9.5296, 8.3925, 7.6976, 7.7776, 8.8363, 14.6291, 7.9523, 8.5095, 
	9.4513, 10.8671, 10.0301, 7.0837, 8.7835, 8.2457, 9.1762, 7.8972, 11.8294, 
	6.6201, 6.815, 9.2472, 6.0684, 7.3866, 6.4261, 6.6696, 7.5083, 7.9376, 
	5.7945, 6.7975, 7.4113, 6.6602, 7.7565, 5.7888, 5.7582, 8.2307, 6.0439, 
	5.5457, 7.4292, 7.8641, 6.1818, 6.6728, 6.1108, 7.5972, 5.2215, 6.9156, 
	7.8731, 9.4223, 6.1425, 6.7087, 7.5943, 6.8337, 5.7614, 5.119, 5.3402, 
	5.2962, 5.8339, 6.3244, 5.0581, 5.4821, 5.6646, 6.4401, 4.9711, 5.0013]

digits_sqrt_sample_ranking = [818, 1296, 732, 988, 629, 1747, 951, 235, 1205, 
	898, 1313, 1766, 1657, 283, 178, 1271, 160, 591, 1070, 736, 185, 1113, 
	491, 1793, 1017, 221, 1022, 423, 1493, 693, 890, 1796, 538, 919, 1009, 
	565, 163, 1176, 208, 313, 263, 1317, 1043, 956, 502, 1030, 1082, 1086, 
	430, 1393, 586, 756, 33, 854, 805, 615, 768, 757, 1704, 1193, 1273, 352, 
	1487, 481, 424, 1191, 407, 873, 1349, 673, 979, 1342, 1021, 1668, 453, 
	77, 1470, 1305, 548, 168, 1263, 1437, 655, 786, 1071, 1186, 451, 851, 
	1012, 500, 172, 436, 767, 1106, 372, 689, 1709, 1105, 1130, 1117]

digits_sqrt_sample_gains = [124.8187, 59.9654, 47.7603, 37.88, 34.3933, 29.819, 
	27.4785, 25.2363, 23.3612, 21.8656, 20.7177, 19.9711, 19.0521, 18.4552, 
	17.8137, 17.2874, 16.5854, 16.2293, 15.7069, 15.2899, 14.87, 14.5026, 
	14.1959, 13.7814, 13.4552, 13.1936, 12.9014, 12.6701, 12.4045, 12.2143, 
	11.9625, 11.7778, 11.6071, 11.4762, 11.3381, 11.1117, 10.9774, 10.8607, 
	10.6766, 10.4814, 10.3056, 10.2264, 10.051, 9.9809, 9.887, 9.7386, 9.6435, 
	9.4956, 9.4086, 9.3191, 9.2254, 9.1569, 9.0615, 8.9634, 8.8806, 8.7956, 
	8.7102, 8.6244, 8.5677, 8.4807, 8.4288, 8.3312, 8.256, 8.1811, 8.1037, 
	8.037, 7.9838, 7.9265, 7.8527, 7.8004, 7.7752, 7.7201, 7.6412, 7.5886, 
	7.5193, 7.4676, 7.4258, 7.3639, 7.3103, 7.2682, 7.224, 7.1757, 7.1467, 
	7.0998, 7.0498, 7.0015, 6.9572, 6.9109, 6.8789, 6.8376, 6.7858, 6.7495, 
	6.7202, 6.6808, 6.6485, 6.619, 6.5719, 6.5329, 6.504, 6.4681]

digits_sqrt_modular_ranking = [818, 1766, 491, 178, 185, 768, 1747, 513, 160, 
	208, 423, 898, 1793, 854, 424, 890, 352, 1796, 505, 402, 978, 148, 459, 
	138, 500, 457, 1030, 666, 693, 1342, 235, 509, 370, 417, 452, 1296, 615, 
	407, 1709, 309, 481, 736, 578, 1082, 168, 805, 55, 1186, 1379, 453, 1759, 
	1205, 332, 126, 565, 1668, 221, 644, 1021, 451, 1493, 209, 1276, 831, 420, 
	1071, 396, 1704, 1009, 688, 1703, 1774, 913, 1069, 1393, 1317, 646, 1193, 
	951, 1349, 985, 1736, 72, 1705, 1545, 1113, 1105, 899, 1470, 1015, 33, 
	1474, 1310, 514, 1620, 1676, 786, 1106, 439, 1781]

digits_sqrt_modular_gains = [124.8187, 51.9061, 47.4915, 35.6125, 29.7708, 
	27.3961, 26.4327, 24.1408, 22.1667, 20.2845, 20.0954, 21.1915, 18.2166, 
	16.7368, 16.2063, 16.7485, 15.7232, 15.6179, 14.0931, 13.7756, 13.6743, 
	12.7662, 13.1461, 12.4817, 12.9374, 15.6196, 12.4405, 12.1912, 12.4937, 
	11.7731, 12.5722, 10.859, 10.4607, 10.5201, 10.5302, 12.899, 10.5381, 
	10.313, 10.013, 9.4011, 10.09, 11.2713, 9.7648, 9.9619, 9.3907, 9.6243, 
	8.9277, 9.2394, 8.9121, 8.982, 8.8631, 9.9972, 8.2035, 8.0633, 8.8855, 
	8.6107, 8.9425, 7.8859, 9.1915, 8.1208, 8.5308, 7.7537, 7.6988, 7.7163, 
	7.481, 7.8962, 7.4548, 7.9412, 10.5212, 8.2094, 7.1568, 7.2931, 6.8229, 
	7.0808, 8.7026, 7.518, 7.1572, 7.3819, 8.3246, 7.2349, 7.4443, 6.5081, 
	6.62, 6.5171, 6.3782, 8.1463, 6.7833, 6.2106, 6.7071, 6.3, 7.7459, 6.6087, 
	6.5699, 5.9068, 6.3729, 6.3294, 6.4772, 6.3678, 5.9507, 6.354]

digits_sqrt_sieve_ranking = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
	15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 
	36, 37, 38, 39, 40, 41, 44, 48, 49, 52, 55, 58, 61, 64, 66, 72, 73, 76, 77, 
	79, 84, 87, 98, 126, 135, 138, 160, 163, 168, 172, 178, 185, 208, 221, 235, 
	263, 283, 313, 317, 423, 430, 436, 491, 494, 502, 538, 586, 591, 629, 673, 
	732, 757, 818, 873, 988, 1043, 1070, 1086, 1176, 1205, 1271, 1296, 1313, 
	1375, 1493, 1572]

digits_sqrt_sieve_gains = [97.4318, 56.261, 41.2277, 28.303, 22.4519, 
	26.0864, 20.4726, 25.2477, 21.9949, 19.2037, 19.0145, 16.1839, 16.4639, 
	18.937, 17.1049, 18.4655, 14.7195, 14.2448, 10.6285, 11.6475, 13.6245, 
	12.8581, 10.1707, 11.3949, 9.3721, 13.8714, 12.2688, 10.2994, 11.2193, 
	13.2881, 9.4804, 13.0039, 12.9107, 9.8257, 10.7652, 9.9669, 9.839, 9.8269, 
	9.6268, 9.8585, 10.6952, 9.1965, 9.0226, 10.4219, 10.352, 9.0895, 9.4044, 
	9.1408, 9.5094, 9.6559, 8.7728, 8.9756, 10.122, 8.9807, 9.5481, 8.7555, 
	8.6382, 8.6878, 8.6501, 8.736, 9.5372, 9.3277, 8.7554, 8.7882, 9.1946, 
	8.9874, 8.7261, 8.6805, 8.9986, 11.8718, 10.1197, 8.595, 9.8908, 8.3371, 
	8.4011, 8.5834, 8.2552, 8.3212, 9.5605, 8.3984, 8.3562, 8.395, 8.3125, 
	8.7659, 8.9045, 8.4036, 8.0924, 9.0439, 9.965, 7.7882, 7.9432, 8.8214, 
	8.1663, 7.5898, 7.6332, 7.6048, 7.6056, 7.7276, 7.2711, 7.6604]


# Test some concave functions

def test_digits_log_naive():
	model = FeatureBasedSelection(100, 'log', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_log_ranking[:30])
	assert_array_equal(model.ranking[-30:], digits_log_ranking[-30:])
	assert_array_almost_equal(model.gains, digits_log_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_log_lazy():
	model = FeatureBasedSelection(100, 'log', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_log_ranking)
	assert_array_almost_equal(model.gains, digits_log_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_log_two_stage():
	model = FeatureBasedSelection(100, 'log', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_log_ranking)
	assert_array_almost_equal(model.gains, digits_log_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sigmoid_naive():
	model = FeatureBasedSelection(100, 'sigmoid', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sigmoid_ranking)
	assert_array_almost_equal(model.gains, digits_sigmoid_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sigmoid_lazy():
	model = FeatureBasedSelection(100, 'sigmoid', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sigmoid_ranking)
	assert_array_almost_equal(model.gains, digits_sigmoid_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sigmoid_two_stage():
	model = FeatureBasedSelection(100, 'sigmoid', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sigmoid_ranking)
	assert_array_almost_equal(model.gains, digits_sigmoid_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_naive():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_lazy():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_two_stage():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

# Test with initialization

def test_digits_log_naive_init():
	model = FeatureBasedSelection(100, 'log', optimizer='naive', 
		initial_subset=digits_log_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:20], digits_log_ranking[5:25])
	assert_array_almost_equal(model.gains[:20], digits_log_gains[5:25], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_log_lazy_init():
	model = FeatureBasedSelection(100, 'log', optimizer='lazy', 
		initial_subset=digits_log_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_log_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_log_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_log_two_stage_init():
	model = FeatureBasedSelection(100, 'log', optimizer='two-stage', 
		initial_subset=digits_log_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_log_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_log_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sigmoid_naive_init():
	model = FeatureBasedSelection(100, 'sigmoid', optimizer='naive', 
		initial_subset=digits_sigmoid_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_sigmoid_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_sigmoid_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sigmoid_lazy_init():
	model = FeatureBasedSelection(100, 'sigmoid', optimizer='lazy', 
		initial_subset=digits_sigmoid_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_sigmoid_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_sigmoid_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sigmoid_two_stage_init():
	model = FeatureBasedSelection(100, 'sigmoid', optimizer='two-stage', 
		initial_subset=digits_sigmoid_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_sigmoid_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_sigmoid_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_naive_init():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='naive', 
		initial_subset=digits_sqrt_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_sqrt_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_sqrt_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_lazy_init():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='lazy', 
		initial_subset=digits_sqrt_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_sqrt_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_sqrt_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_two_stage_init():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='two-stage', 
		initial_subset=digits_sqrt_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_sqrt_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_sqrt_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

# Test all optimizers

def test_digits_sqrt_greedi_nn():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:50], digits_sqrt_greedi_ranking[:50])
	assert_array_almost_equal(model.gains[:50], digits_sqrt_greedi_gains[:50], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_greedi_ll():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_sqrt_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_sqrt_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_greedi_ln():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_greedi_nl():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_sqrt_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_sqrt_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_approximate():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='approximate-lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_approx_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_approx_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_stochastic():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='stochastic',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_stochastic_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_sample():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='sample',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_sample_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_sample_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_modular():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='modular',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_modular_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_modular_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

# Using the partial_fit method

def test_digits_sqrt_sieve_batch():
	model = FeatureBasedSelection(100, 'sqrt', random_state=0)
	model.partial_fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_sieve_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_sieve_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_sieve_minibatch():
	model = FeatureBasedSelection(100, 'sqrt', random_state=0)
	model.partial_fit(X_digits[:300])
	model.partial_fit(X_digits[300:500])
	model.partial_fit(X_digits[500:])
	assert_array_equal(model.ranking, digits_sqrt_sieve_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_sieve_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])


def test_digits_sqrt_sieve_batch_sparse():
	model = FeatureBasedSelection(100, 'sqrt', random_state=0)
	model.partial_fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_sqrt_sieve_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_sieve_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_sieve_minibatch_sparse():
	model = FeatureBasedSelection(100, 'sqrt', random_state=0)
	model.partial_fit(X_digits_sparse[:300])
	model.partial_fit(X_digits_sparse[300:500])
	model.partial_fit(X_digits_sparse[500:])
	assert_array_equal(model.ranking, digits_sqrt_sieve_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_sieve_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

# Using Optimizer Objects

def test_digits_sqrt_naive_object():
	model = FeatureBasedSelection(100, 'sqrt', optimizer=NaiveGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_lazy_object():
	model = FeatureBasedSelection(100, 'sqrt', optimizer=LazyGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_two_stage_object():
	model = FeatureBasedSelection(100, 'sqrt', optimizer=TwoStageGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_greedi_nn_object():
	model = FeatureBasedSelection(100, 'sqrt', optimizer=GreeDi(
		optimizer1='naive', optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_greedi_ll_object():
	model = FeatureBasedSelection(100, 'sqrt', optimizer=GreeDi(
		optimizer1='lazy', optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_sqrt_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_sqrt_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_greedi_ln_object():
	model = FeatureBasedSelection(100, 'sqrt', optimizer=GreeDi(
		optimizer1='lazy', optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_greedi_nl_object():
	model = FeatureBasedSelection(100, 'sqrt', optimizer=GreeDi(
		optimizer1='naive', optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_sqrt_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_sqrt_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_approximate_object():
	model = FeatureBasedSelection(100, 'sqrt', 
		optimizer=ApproximateLazyGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_approx_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_approx_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_stochastic_object():
	model = FeatureBasedSelection(100, 'sqrt', 
		optimizer=StochasticGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_stochastic_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_sample_object():
	model = FeatureBasedSelection(100, 'sqrt', 
		optimizer=SampleGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_sample_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_sample_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_modular_object():
	model = FeatureBasedSelection(100, 'sqrt', 
		optimizer=ModularGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_sqrt_modular_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_modular_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

# Test all optimizers on sparse data

def test_digits_sqrt_naive_sparse():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='naive')
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_sqrt_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_gains, 4)
	assert_array_almost_equal(model.subset, X_digits_sparse[model.ranking].toarray())

def test_digits_sqrt_lazy_sparse():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='lazy')
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_sqrt_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_gains, 4)
	assert_array_almost_equal(model.subset, X_digits_sparse[model.ranking].toarray())

def test_digits_sqrt_two_stage_sparse():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='two-stage')
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_sqrt_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_gains, 4)
	assert_array_almost_equal(model.subset, X_digits_sparse[model.ranking].toarray())

def test_digits_sqrt_greedi_nn_sparse():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_sqrt_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits_sparse[model.ranking].toarray())

def test_digits_sqrt_greedi_ll_sparse():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking[:30], digits_sqrt_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_sqrt_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits_sparse[model.ranking].toarray())

def test_digits_sqrt_greedi_ln_sparse():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_sqrt_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits_sparse[model.ranking].toarray())

def test_digits_sqrt_greedi_nl_sparse():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking[:30], digits_sqrt_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_sqrt_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits_sparse[model.ranking].toarray())

def test_digits_sqrt_approximate_sparse():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='approximate-lazy')
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_sqrt_approx_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_approx_gains, 4)
	assert_array_almost_equal(model.subset, X_digits_sparse[model.ranking].toarray())

def test_digits_sqrt_stochastic_sparse():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='stochastic',
		random_state=0)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_sqrt_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_stochastic_gains, 4)
	assert_array_almost_equal(model.subset, X_digits_sparse[model.ranking].toarray())

def test_digits_sqrt_sample_sparse():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='sample',
		random_state=0)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_sqrt_sample_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_sample_gains, 4)
	assert_array_almost_equal(model.subset, X_digits_sparse[model.ranking].toarray())

def test_digits_sqrt_modular_sparse():
	model = FeatureBasedSelection(100, 'sqrt', optimizer='modular',
		random_state=0)
	model.fit(X_digits_sparse)
	assert_array_equal(model.ranking, digits_sqrt_modular_ranking)
	assert_array_almost_equal(model.gains, digits_sqrt_modular_gains, 4)
	assert_array_almost_equal(model.subset, X_digits_sparse[model.ranking].toarray())
