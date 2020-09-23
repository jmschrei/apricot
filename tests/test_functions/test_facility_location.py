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

digits_cosine_ranking = [424, 615, 1545, 1385, 1482, 112, 1539, 1075, 331, 
	493, 885, 345, 1282, 823, 1432, 1051, 537, 1788, 1549, 1622, 834, 1634, 
	1718, 1474, 1185, 655, 1292, 396, 1711, 1676, 2, 183, 1536, 983, 438, 
	1276, 305, 1353, 620, 384, 1026, 162, 1012, 798, 213, 227, 1291, 533, 
	1655, 1485, 410, 1206, 157, 1387, 556, 29, 1588, 1320, 1295, 164, 514, 
	938, 948, 517, 579, 1325, 1682, 1222, 82, 959, 943, 520, 762, 1066, 898, 
	1227, 1568, 1086, 881, 1678, 1470, 1364, 732, 937, 1168, 347, 241, 573, 
	1156, 1570, 908, 1584, 126, 1639, 815, 411, 151, 846, 696, 233]

digits_cosine_gains = [1126.631, 76.1809, 43.868, 35.4606, 33.088, 32.6405, 
	27.5345, 23.2181, 20.0442, 15.384, 10.8113, 9.5884, 8.3655, 7.7554, 7.6675, 
	7.3457, 5.2902, 5.0039, 4.6488, 4.2388, 4.2165, 4.1855, 3.7814, 3.4738, 
	3.1826, 3.1456, 3.1258, 3.0707, 2.9035, 2.8744, 2.788, 2.4376, 2.3306, 
	1.9438, 1.8711, 1.8376, 1.7047, 1.6453, 1.5747, 1.5644, 1.5371, 1.4879, 
	1.4603, 1.4256, 1.4233, 1.4098, 1.3889, 1.3861, 1.3012, 1.2986, 1.278, 
	1.2779, 1.2775, 1.2059, 1.1914, 1.1437, 1.1221, 1.1031, 1.0829, 1.0708, 
	1.0427, 1.0116, 1.0099, 0.9978, 0.9943, 0.98, 0.9717, 0.945, 0.927, 0.9064, 
	0.8899, 0.8746, 0.8651, 0.8649, 0.859, 0.8526, 0.84, 0.8284, 0.8269, 0.8067, 
	0.8066, 0.7645, 0.7517, 0.7409, 0.7396, 0.7383, 0.7237, 0.7209, 0.7043, 
	0.7021, 0.6945, 0.6806, 0.6585, 0.6539, 0.6198, 0.6161, 0.6078, 0.5963, 
	0.5945, 0.5899]

digits_cosine_greedi_ranking = [424, 1030, 468, 1111, 1385, 41, 201, 81, 396, 
	186, 537, 938, 1009, 655, 1071, 1736, 345, 1482, 331, 1788, 1536, 543, 1104, 
	1120, 1535, 1201, 1053, 236, 1282, 339, 1718, 148, 115, 1046, 1203, 530, 
	1185, 494, 155, 621, 1542, 280, 732, 561, 900, 1276, 191, 554, 1051, 275, 
	1134, 1223, 504, 410, 1624, 982, 1364, 241, 692, 1047, 1428, 312, 1709, 
	160, 816, 768, 402, 131, 1265, 1483, 635, 1554, 1614, 923, 552, 188, 1410, 
	493, 32, 264, 1532, 1478, 1244, 675, 976, 235, 262, 132, 809, 668, 1555, 
	1332, 924, 138, 1295, 1533, 165, 822, 1184, 463]

digits_cosine_greedi_gains = [1126.631, 73.1793, 46.4627, 29.5523, 34.0967, 
	22.7932, 21.2345, 19.1614, 37.8457, 10.9945, 6.7468, 10.4479, 8.1564, 
	3.4669, 9.0943, 6.8193, 9.0551, 7.4868, 7.4788, 4.6044, 5.3755, 2.0625, 
	3.1779, 4.5603, 3.5798, 2.3906, 1.7633, 3.2745, 3.2062, 2.0717, 2.8894, 
	2.3138, 1.7417, 1.4141, 1.6926, 1.0587, 2.5893, 1.7456, 1.0624, 1.278, 
	1.4662, 0.8194, 0.7861, 1.0377, 1.0657, 1.2889, 0.5415, 1.1421, 1.082, 
	1.0706, 2.0516, 1.4782, 0.4165, 2.6358, 1.2067, 0.9377, 0.7095, 0.9775, 
	1.3211, 0.758, 1.699, 0.8115, 0.6292, 1.8336, 0.54, 0.6586, 0.4637, 0.813, 
	0.3427, 0.2793, 0.8045, 0.1796, 0.9699, 0.7823, 1.132, 0.8785, 0.5449, 
	1.0747, 1.3588, 0.3345, 0.3211, 0.6306, 0.4073, 0.139, 0.668, 0.2053, 
	0.3902, 0.8324, 0.1153, 0.5048, 1.1628, 0.4711, 0.8476, 0.8886, 0.6292, 
	0.6492, 0.6842, 0.639, 0.2356, 0.2605]

digits_cosine_approx_ranking = [424, 615, 1545, 1385, 983, 1482, 1539, 1075, 331, 
	493, 885, 345, 823, 1282, 1051, 236, 537, 1788, 1161, 655, 195, 1634, 438, 
	1718, 1474, 1676, 1292, 533, 396, 1185, 183, 1711, 2, 640, 1353, 1276, 305, 
	384, 1442, 1026, 1012, 213, 162, 197, 798, 227, 1291, 410, 1485, 1655, 1206, 
	29, 1588, 556, 1682, 1536, 1325, 1295, 517, 514, 579, 1461, 1320, 1294, 948, 
	567, 1222, 898, 943, 762, 550, 1556, 1066, 1350, 1568, 881, 689, 1678, 1470, 
	347, 82, 1364, 1168, 1549, 1156, 959, 241, 126, 1628, 149, 652, 411, 1639, 937, 
	216, 1250, 1257, 815, 151, 846]

digits_cosine_approx_gains = [1126.631, 76.1809, 43.868, 35.4606, 32.6168, 
	33.0709, 28.6118, 23.568, 20.2298, 15.2965, 10.8862, 9.3698, 8.3121, 8.3655, 
	7.3398, 6.4778, 5.2902, 5.0039, 4.7001, 4.4474, 4.1603, 4.1855, 3.8457, 3.7814, 
	3.4738, 3.0503, 2.9736, 2.8744, 3.0707, 2.8541, 2.7273, 2.6049, 2.4984, 2.3956, 
	2.138, 1.8376, 1.7047, 1.5309, 1.4743, 1.5371, 1.5433, 1.4296, 1.5101, 1.4607, 
	1.4256, 1.4098, 1.3889, 1.281, 1.2986, 1.3012, 1.2779, 1.1437, 1.1221, 1.1914, 
	1.0978, 1.0958, 1.0816, 0.987, 0.9978, 1.0427, 0.9943, 0.9759, 1.0199, 1.0072, 
	0.9435, 0.8643, 0.9121, 0.859, 0.8899, 0.8651, 0.8293, 0.8497, 0.8649, 0.7972, 
	0.7962, 0.7702, 0.7473, 0.7945, 0.8066, 0.7383, 0.7807, 0.7645, 0.734, 0.7296, 
	0.7043, 0.728, 0.7237, 0.6585, 0.6398, 0.6063, 0.6387, 0.6161, 0.6539, 0.6395, 
	0.5825, 0.5886, 0.6205, 0.6031, 0.6078, 0.5963]

digits_cosine_stochastic_ranking = [1081, 1014, 1386, 770, 567, 137, 723, 1491, 
	1274, 1492, 1728, 1456, 186, 1448, 386, 148, 891, 1759, 1424, 587, 191, 629, 
	1507, 1084, 1473, 946, 518, 638, 1739, 502, 1537, 1227, 689, 88, 238, 1667, 
	1785, 1067, 1461, 1222, 1099, 607, 364, 1572, 1195, 217, 1034, 208, 84, 1128, 
	425, 345, 626, 843, 1070, 83, 1449, 1071, 1644, 1392, 1415, 449, 802, 1348, 
	1553, 175, 1455, 1770, 1395, 1032, 879, 1220, 1137, 129, 754, 1695, 1459, 782, 
	549, 1069, 260, 834, 517, 919, 1622, 700, 424, 1685, 245, 1339, 1152, 1212, 
	1425, 937, 1665, 291, 1535, 701, 1508, 1219]

digits_cosine_stochastic_gains = [889.4143, 20.4744, 103.4191, 46.2267, 4.3305, 
	18.5211, 54.3538, 35.5077, 6.551, 5.7077, 5.6756, 37.63, 17.5565, 34.278, 
	42.5224, 35.5992, 1.2107, 19.8488, 3.8072, 3.4409, 2.7019, 1.3549, 1.5089, 
	1.0551, 14.8608, 2.5029, 0.4053, 1.4568, 4.6182, 1.0255, 2.0205, 3.7699, 
	2.0694, 12.8589, 2.3583, 3.3821, 3.8403, 1.1118, 1.6831, 4.3867, 1.2917, 
	1.8872, 0.5247, 1.1549, 1.0258, 6.2302, 10.167, 3.4695, 4.1669, 1.0842, 
	2.3784, 11.029, 1.6765, 0.7525, 0.7335, 0.454, 2.307, 4.0194, 0.7664, 0.3671, 
	0.9314, 3.1446, 1.2759, 2.6267, 0.4132, 0.6536, 0.5142, 0.4766, 0.7555, 
	1.1265, 0.1483, 0.5388, 0.6339, 0.4821, 3.1597, 0.6725, 2.7096, 0.2288, 
	0.335, 0.8554, 2.0427, 0.6971, 4.5681, 2.0573, 0.6791, 0.703, 1.0025, 
	0.3316, 2.9545, 1.2529, 0.736, 0.7703, 0.2727, 0.7296, 1.1756, 0.2375, 
	1.4499, 0.0924, 1.4348, 0.2389]

digits_cosine_sample_ranking = [424, 615, 1545, 1385, 1482, 112, 1539, 1075, 
	331, 493, 885, 345, 1282, 823, 1432, 1051, 537, 1788, 1549, 1622, 834, 
	1634, 1718, 1474, 1185, 655, 1292, 396, 1711, 1676, 2, 183, 1536, 983, 
	1276, 305, 384, 403, 1353, 620, 1026, 162, 1012, 798, 213, 227, 1291, 
	533, 1655, 1485, 410, 1206, 157, 1387, 556, 29, 1588, 1682, 1295, 164, 
	514, 948, 579, 972, 1325, 1461, 1312, 1222, 82, 959, 943, 520, 762, 
	1066, 898, 1227, 1568, 1086, 881, 1678, 1470, 1364, 732, 1156, 241, 
	573, 1570, 908, 1504, 1584, 126, 1639, 1158, 937, 815, 411, 151, 149, 
	846, 696]

digits_cosine_sample_gains = [1126.631, 76.1809, 43.868, 35.4606, 33.088, 
	32.6405, 27.5345, 23.2181, 20.0442, 15.384, 10.8113, 9.5884, 8.3655, 
	7.7554, 7.6675, 7.3457, 5.2902, 5.0039, 4.6488, 4.2388, 4.2165, 4.1855, 
	3.7814, 3.4738, 3.1826, 3.1456, 3.1258, 3.0707, 2.9035, 2.8744, 2.788, 
	2.4376, 2.3306, 1.9438, 1.8376, 1.7047, 1.6997, 1.6733, 1.6453, 1.5747, 
	1.5371, 1.4879, 1.4603, 1.4256, 1.4233, 1.4098, 1.3889, 1.3861, 1.3246, 
	1.2986, 1.278, 1.2779, 1.2775, 1.2059, 1.1914, 1.1437, 1.1221, 1.0978, 
	1.0829, 1.0708, 1.0427, 1.0099, 0.9943, 0.981, 0.98, 0.9759, 0.9456, 
	0.945, 0.927, 0.9064, 0.8899, 0.8746, 0.8651, 0.8649, 0.859, 0.8526, 
	0.84, 0.8284, 0.8269, 0.8067, 0.8066, 0.7645, 0.7517, 0.7295, 0.7237, 
	0.7209, 0.7021, 0.6945, 0.6898, 0.6806, 0.6585, 0.6576, 0.6469, 0.6395, 
	0.6198, 0.6111, 0.6078, 0.5964, 0.5963, 0.5945]

digits_cosine_modular_ranking = [424, 148, 615, 1747, 1030, 1766, 818, 1363, 
	768, 509, 1774, 138, 1295, 402, 255, 1069, 1709, 852, 248, 1327, 168, 459, 
	890, 513, 899, 945, 452, 183, 423, 923, 1658, 1325, 491, 898, 742, 478, 
	352, 514, 823, 1040, 978, 903, 426, 1071, 1796, 269, 1794, 332, 1781, 657, 
	913, 814, 1340, 417, 254, 505, 1668, 1320, 1423, 684, 1433, 309, 114, 649, 
	420, 1455, 1453, 1647, 508, 224, 1632, 1705, 1763, 296, 301, 1737, 515, 
	836, 370, 1026, 997, 500, 1199, 736, 686, 1596, 76, 1757, 816, 693, 448, 
	339, 547, 721, 1726, 1678, 854, 405, 249, 1323]

digits_cosine_modular_gains = [1126.631, 44.738, 40.5236, 21.8344, 14.9058, 
	2.0543, 0.301, 1.4012, 6.271, 5.6542, 0.6256, 2.3966, 2.8463, 16.5448, 
	0.4742, 1.412, 0.4318, 3.4858, 0.7938, 0.5358, 0.3774, 14.1052, 7.2849, 
	1.9459, 3.8649, 1.2624, 9.6255, 1.0289, 4.0384, 2.9969, 5.4019, 1.0495, 
	1.2719, 1.6337, 0.3053, 3.4975, 0.7808, 5.4622, 2.994, 0.5453, 0.2372, 
	0.3399, 0.1372, 0.7602, 0.4801, 6.4833, 0.1203, 0.2693, 0.5488, 1.5296, 
	0.1185, 0.3899, 0.6843, 0.2927, 0.5842, 2.3053, 0.3581, 6.3265, 0.3343, 
	0.3976, 0.1859, 0.1083, 0.4559, 1.2732, 0.4012, 0.2193, 0.0957, 2.9362, 
	0.1251, 0.7071, 1.123, 0.1063, 0.1448, 0.1763, 2.1166, 0.467, 0.1193, 
	1.0707, 0.0863, 0.133, 0.0905, 0.1449, 2.0452, 2.1527, 0.061, 0.2085, 
	0.29, 0.7726, 0.1614, 1.1461, 0.649, 0.8119, 0.3708, 3.3097, 0.8278, 
	0.5271, 15.1531, 2.0343, 0.0976, 0.6053]

digits_cosine_sieve_ranking = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 
	14, 15, 16, 17, 18, 20, 22, 25, 26, 27, 29, 32, 34, 39, 40, 41, 44, 52, 
	56, 58, 62, 65, 69, 74, 76, 79, 84, 94, 99, 117, 126, 128, 131, 136, 
	138, 144, 146, 148, 149, 152, 154, 157, 164, 165, 174, 180, 220, 251, 
	257, 259, 260, 262, 264, 275, 281, 287, 297, 301, 324, 331, 345, 349, 
	358, 366, 375, 387, 393, 424, 425, 436, 442, 450, 460, 470, 517, 518, 
	519, 520, 522, 528, 531, 533, 534, 540, 543, 545, 546]

digits_cosine_sieve_gains = [0.4861, 0.1082, 0.0127, 0.0322, 0.0112, 0.011, 
	0.0236, 0.0145, 0.0084, 0.0029, 0.0059, 0.0078, 0.0082, 0.0094, 0.0139, 
	0.0017, 0.0091, 0.0026, 0.0015, 0.0069, 0.0015, 0.0014, 0.0014, 0.0018, 
	0.0052, 0.0015, 0.0051, 0.0026, 0.0026, 0.0031, 0.002, 0.0014, 0.0024, 
	0.0012, 0.0028, 0.001, 0.0011, 0.0022, 0.0021, 0.0031, 0.0019, 0.0019, 
	0.0013, 0.001, 0.0017, 0.0012, 0.0017, 0.0009, 0.0009, 0.0011, 0.0016, 
	0.001, 0.0011, 0.0009, 0.0016, 0.0008, 0.0009, 0.0009, 0.0012, 0.001, 
	0.001, 0.0009, 0.0008, 0.001, 0.0008, 0.0009, 0.0009, 0.0009, 0.0008, 
	0.0009, 0.0011, 0.0008, 0.001, 0.0009, 0.0008, 0.0008, 0.0009, 0.0014, 
	0.0007, 0.0011, 0.0007, 0.0008, 0.001, 0.0018, 0.0006, 0.0006, 0.0007, 
	0.0008, 0.0008, 0.0006, 0.001, 0.0006, 0.0004, 0.0005, 0.0005, 0.0005, 
	0.0004, 0.0005, 0.0004, 0.0004]

# Test some similarity functions

def test_digits_euclidean_naive():
	model = FacilityLocationSelection(100, 'euclidean', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_euclidean_ranking[:30])
	assert_array_equal(model.ranking[-30:], digits_euclidean_ranking[-30:])
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_lazy():
	model = FacilityLocationSelection(100, 'euclidean', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_two_stage():
	model = FacilityLocationSelection(100, 'euclidean', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_naive():
	model = FacilityLocationSelection(100, 'corr', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_lazy():
	model = FacilityLocationSelection(100, 'corr', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_two_stage():
	model = FacilityLocationSelection(100, 'corr', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_naive():
	model = FacilityLocationSelection(100, 'cosine', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_lazy():
	model = FacilityLocationSelection(100, 'cosine', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_two_stage():
	model = FacilityLocationSelection(100, 'cosine', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_precomputed_naive():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='naive')
	model.fit(X_digits_corr_cupy)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_precomputed_lazy():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='lazy')
	model.fit(X_digits_corr_cupy)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_precomputed_two_stage():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='two-stage')
	model.fit(X_digits_corr_cupy)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

# Test with initialization

def test_digits_euclidean_naive_init():
	model = FacilityLocationSelection(100, 'euclidean', optimizer='naive', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:20], digits_euclidean_ranking[5:25])
	assert_array_almost_equal(model.gains[:20], digits_euclidean_gains[5:25], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_lazy_init():
	model = FacilityLocationSelection(100, 'euclidean', optimizer='lazy', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_euclidean_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_euclidean_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_two_stage_init():
	model = FacilityLocationSelection(100, 'euclidean', optimizer='two-stage', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_euclidean_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_euclidean_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_naive_init():
	model = FacilityLocationSelection(100, 'corr', optimizer='naive', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_lazy_init():
	model = FacilityLocationSelection(100, 'corr', optimizer='lazy', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_two_stage_init():
	model = FacilityLocationSelection(100, 'corr', optimizer='two-stage', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_naive_init():
	model = FacilityLocationSelection(100, 'cosine', optimizer='naive', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_lazy_init():
	model = FacilityLocationSelection(100, 'cosine', optimizer='lazy', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_two_stage_init():
	model = FacilityLocationSelection(100, 'cosine', optimizer='two-stage', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_precomputed_naive_init():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='naive', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_precomputed_lazy_init():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='lazy', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_precomputed_two_stage_init():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='two-stage', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

# Test all optimizers

def test_digits_cosine_greedi_nn():
	model = FacilityLocationSelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:50], digits_cosine_greedi_ranking[:50])
	assert_array_almost_equal(model.gains[:50], digits_cosine_greedi_gains[:50], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ll():
	model = FacilityLocationSelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ln():
	model = FacilityLocationSelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_nl():
	model = FacilityLocationSelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_approximate():
	model = FacilityLocationSelection(100, 'cosine', optimizer='approximate-lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_stochastic():
	model = FacilityLocationSelection(100, 'cosine', optimizer='stochastic',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_sample():
	model = FacilityLocationSelection(100, 'cosine', optimizer='sample',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_modular():
	model = FacilityLocationSelection(100, 'cosine', optimizer='modular',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

# Using the partial_fit method

def test_digits_cosine_sieve_batch():
	model = FacilityLocationSelection(100, 'cosine', random_state=0, 
		reservoir=X_digits)
	model.partial_fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_sieve_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sieve_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_sieve_minibatch():
	model = FacilityLocationSelection(100, 'cosine', random_state=0,
		reservoir=X_digits)
	model.partial_fit(X_digits[:300])
	model.partial_fit(X_digits[300:500])
	model.partial_fit(X_digits[500:])
	assert_array_equal(model.ranking, digits_cosine_sieve_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sieve_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

# Using Optimizer Objects

def test_digits_cosine_naive_object():
	model = FacilityLocationSelection(100, 'cosine', optimizer=NaiveGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_lazy_object():
	model = FacilityLocationSelection(100, 'cosine', optimizer=LazyGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_two_stage_object():
	model = FacilityLocationSelection(100, 'cosine', optimizer=TwoStageGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_nn_object():
	model = FacilityLocationSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='naive', optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ll_object():
	model = FacilityLocationSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='lazy', optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ln_object():
	model = FacilityLocationSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='lazy', optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_nl_object():
	model = FacilityLocationSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='naive', optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_approximate_object():
	model = FacilityLocationSelection(100, 'cosine', 
		optimizer=ApproximateLazyGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_stochastic_object():
	model = FacilityLocationSelection(100, 'cosine', 
		optimizer=StochasticGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_sample_object():
	model = FacilityLocationSelection(100, 'cosine', 
		optimizer=SampleGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_modular_object():
	model = FacilityLocationSelection(100, 'cosine', 
		optimizer=ModularGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

# Test all optimizers on sparse data

def test_digits_cosine_naive_sparse():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='naive')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_lazy_sparse():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='lazy')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_two_stage_sparse():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='two-stage')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_greedi_nn_sparse():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_ll_sparse():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)

def test_digits_cosine_greedi_ln_sparse():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_nl_sparse():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)

def test_digits_cosine_approximate_sparse():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='approximate-lazy')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)

def test_digits_cosine_stochastic_sparse():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='stochastic',
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)

def test_digits_cosine_sample_sparse():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='sample',
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)

def test_digits_sqrt_modular_sparse():
	model = FacilityLocationSelection(100, 'precomputed', optimizer='modular',
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)
	