import scipy
import numpy

try:
	import cupy
except:
	import numpy as cupy

from apricot import MixtureSelection
from apricot import FacilityLocationSelection
from apricot import GraphCutSelection
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

digits_corr_ranking = [424, 615, 452, 514, 1030, 269, 1747, 1545, 1295, 148, 
	1363, 1327, 1766, 509, 852, 818, 890, 1774, 138, 1320, 945, 248, 255, 1709, 
	768, 402, 823, 899, 1658, 1069, 1647, 923, 183, 1325, 459, 168, 657, 301, 
	761, 513, 1040, 426, 478, 742, 1632, 423, 1071, 814, 1199, 649, 1617, 1423, 
	903, 1794, 114, 102, 339, 448, 1381, 1352, 1346, 254, 1340, 1781, 978, 898, 
	721, 41, 491, 816, 997, 1737, 522, 1688, 1075, 913, 1323, 332, 249, 1757, 
	244, 405, 1668, 836, 1276, 352, 74, 684, 1455, 1433, 417, 1453, 1796, 1409, 
	1639, 736, 508, 1752, 420, 1760]

digits_corr_gains = [957.5322, 330.2478, 254.6831, 251.206, 245.0905, 225.7705, 
	223.1784, 219.9972, 219.2199, 216.0972, 213.3351, 210.5622, 209.4755, 
	208.1544, 206.7405, 204.93, 204.522, 203.7405, 202.8789, 201.3152, 200.8991, 
	200.1773, 199.5643, 197.7437, 197.3154, 196.6495, 196.1545, 194.9538, 
	194.3206, 192.8184, 192.3224, 191.6689, 190.3903, 189.0685, 188.1243, 
	187.8275, 186.6411, 185.0932, 184.2903, 183.842, 183.3303, 182.6601, 
	181.9112, 180.7916, 180.3595, 179.6811, 179.4377, 178.8864, 178.0831, 
	177.3396, 176.8052, 176.3837, 175.9921, 175.4704, 174.8651, 174.4919, 
	173.9766, 173.1983, 172.7893, 172.5758, 171.5445, 171.1853, 170.9032, 
	170.4194, 169.8876, 169.1956, 168.6746, 168.3132, 167.8759, 167.4243, 
	166.7872, 165.9742, 165.5812, 165.0026, 164.6565, 164.167, 163.7766, 
	162.8939, 162.5745, 162.1263, 161.7664, 161.4983, 161.1224, 160.7753, 
	160.1076, 159.6627, 159.2873, 158.9297, 158.6049, 158.1738, 157.7336, 
	157.1569, 156.7355, 156.4312, 156.1812, 155.7283, 155.3701, 155.0292, 
	154.418, 154.0432]

digits_euclidean_ranking = [945, 1327, 448, 923, 276, 426, 1026, 651, 1583, 
	114, 148, 1423, 1295, 255, 515, 1363, 547, 232, 1544, 955, 699, 607, 404, 
	1058, 768, 1040, 296, 1491, 814, 686, 459, 668, 293, 737, 1459, 378, 654, 
	1346, 1443, 1320, 1637, 1433, 1455, 254, 674, 478, 773, 816, 1450, 183, 
	1617, 269, 482, 170, 205, 1658, 742, 1453, 120, 284, 394, 264, 1696, 248, 
	1361, 330, 1383, 829, 1537, 274, 514, 1486, 888, 40, 419, 1529, 505, 1763, 
	684, 53, 405, 997, 1532, 872, 544, 335, 933, 281, 508, 339, 574, 521, 539, 
	879, 1187, 328, 835, 697, 1123, 836]

digits_euclidean_gains = [9681446.3, 2484661.1, 2444711.8, 2293903.6, 2275176.2, 
	2235700.7, 2221801.7, 2207625.8, 2190108.0, 2177496.9, 2171467.0, 2166543.6, 
	2161585.6, 2153715.3, 2142803.6, 2134885.8, 2131767.5, 2126399.2, 2119268.2, 
	2115821.8, 2111996.6, 2107032.0, 2103319.7, 2096543.6, 2093119.3, 2087047.1, 
	2083877.7, 2080677.5, 2076910.3, 2073829.6, 2068572.5, 2065858.2, 2061714.3, 
	2057002.9, 2052908.2, 2049766.2, 2046509.7, 2043182.8, 2038671.2, 2033131.2, 
	2028417.4, 2025607.1, 2022035.2, 2018470.1, 2015333.3, 2010681.4, 2006781.6, 
	2002971.4, 1999041.2, 1995817.7, 1991182.0, 1987682.4, 1983833.3, 1980961.4, 
	1977563.7, 1974282.9, 1971292.9, 1968097.1, 1963874.4, 1959875.5, 1956835.8, 
	1953737.3, 1950422.9, 1946943.5, 1944247.4, 1940745.7, 1937767.0, 1934156.4, 
	1930520.8, 1926043.9, 1923402.3, 1920436.0, 1917812.6, 1914970.8, 1911951.5, 
	1908892.7, 1906314.3, 1903726.4, 1900097.3, 1897212.0, 1894373.1, 1889770.8, 
	1886297.2, 1883110.5, 1880186.7, 1877545.6, 1874497.6, 1871327.3, 1866850.1, 
	1863564.4, 1859998.0, 1857097.1, 1854710.4, 1850947.3, 1848393.1, 1845132.0, 
	1842794.7, 1837741.4, 1834617.1, 1831495.0]

digits_cosine_ranking = [424, 615, 1030, 402, 514, 1747, 148, 509, 1766, 818, 
	269, 1363, 768, 1295, 452, 890, 138, 1774, 852, 185, 255, 1069, 1709, 459, 
	248, 1327, 168, 513, 899, 923, 945, 423, 183, 1320, 1658, 823, 898, 491, 
	1325, 478, 1040, 742, 352, 978, 1071, 903, 1781, 426, 1796, 332, 657, 301, 
	1794, 913, 814, 417, 1075, 1647, 505, 1340, 254, 1668, 649, 1433, 420, 1423, 
	721, 854, 684, 309, 1199, 1632, 736, 836, 114, 1455, 224, 405, 1737, 1453, 
	453, 296, 500, 339, 693, 1705, 761, 1763, 997, 508, 1276, 1757, 816, 1323, 
	1026, 1596, 448, 1545, 76, 1726]

digits_cosine_gains = [1464.3202, 409.8125, 357.7419, 352.9304, 345.8749, 
	338.5499, 334.8922, 331.7992, 330.1259, 327.8876, 327.1878, 325.2868, 
	324.7755, 323.3606, 322.8546, 322.3117, 320.727, 319.4558, 317.9748, 
	316.9812, 315.9366, 314.6272, 313.954, 313.5249, 312.6064, 311.9777, 
	311.1738, 309.8506, 309.3497, 306.8928, 305.6599, 305.1896, 304.0379, 
	303.148, 302.6694, 302.075, 301.4275, 300.8761, 300.1981, 299.0383, 
	297.1896, 296.6456, 296.1498, 295.4601, 294.9541, 294.543, 293.2464, 
	292.4009, 291.8043, 291.1761, 290.7796, 289.1387, 288.6704, 288.0586, 
	287.4932, 286.8514, 286.4441, 285.8374, 285.2712, 284.6809, 284.2737, 
	283.4397, 283.055, 282.3092, 281.7608, 281.2045, 280.7276, 280.0204, 
	279.5876, 278.7367, 278.2878, 277.7987, 277.0733, 276.2545, 275.8347, 
	275.3892, 274.9112, 274.0533, 273.5174, 273.1525, 272.5565, 272.0778, 
	271.5116, 271.071, 270.6944, 270.1725, 269.7275, 269.0934, 268.5126, 
	268.0091, 267.5602, 266.9365, 266.2236, 265.8127, 265.2859, 264.8279, 
	264.4057, 263.872, 263.5162, 263.0823]

digits_cosine_greedi_ranking = [424, 1766, 148, 138, 402, 1363, 945, 1030, 509, 
	768, 1069, 1295, 514, 615, 1747, 818, 890, 513, 248, 269, 923, 852, 1774, 183, 
	452, 459, 1327, 899, 255, 426, 168, 1423, 1320, 1323, 1709, 978, 478, 1658, 
	913, 814, 352, 1325, 903, 423, 332, 742, 898, 761, 491, 420, 1796, 1433, 823, 
	419, 649, 1794, 505, 1781, 1040, 1759, 185, 997, 657, 339, 1340, 412, 1075, 
	17, 684, 854, 405, 522, 721, 1276, 1647, 1071, 301, 1682, 1545, 1726, 309, 41, 
	244, 1690, 457, 417, 1649, 453, 1428, 1793, 693, 109, 591, 1393, 948, 1704, 
	542, 52, 331, 1641]

digits_cosine_greedi_gains = [1464.3202, 407.5959, 349.2661, 330.6768, 
	344.9928, 332.7493, 321.2969, 343.6912, 330.5744, 328.727, 320.4867, 
	324.4858, 337.2288, 334.6355, 329.0956, 324.7697, 322.1504, 314.5276, 
	315.6515, 319.887, 310.8197, 316.3086, 316.9565, 308.2517, 318.3346, 
	312.5236, 311.5595, 309.844, 311.8585, 300.6039, 309.1181, 296.4878, 
	303.7425, 303.2108, 308.5394, 298.8348, 300.5579, 301.1397, 294.6515, 
	293.7151, 296.9657, 298.8378, 296.0371, 299.6898, 293.0309, 294.7472, 
	296.7244, 289.083, 296.3041, 287.9784, 291.1465, 286.9867, 294.649, 
	284.1909, 286.9657, 287.7189, 286.4448, 288.1401, 289.7308, 279.0076, 
	286.7833, 279.8429, 286.188, 280.8258, 282.8624, 277.1444, 279.7718, 
	273.966, 279.6056, 278.5453, 276.9876, 274.6063, 276.1162, 273.9862, 
	278.0959, 282.386, 275.5896, 268.7937, 270.7513, 271.8441, 274.0963, 
	265.894, 267.1621, 265.1507, 262.735, 274.3377, 258.5314, 266.2975, 
	263.023, 262.9865, 268.3422, 261.1778, 244.0778, 262.1116, 251.3201, 
	257.1708, 250.6925, 234.3935, 254.2639, 241.2629]

digits_cosine_approx_ranking = [424, 615, 452, 1295, 509, 890, 138, 185, 1030, 
	899, 1747, 269, 1766, 514, 1069, 852, 423, 923, 768, 459, 818, 1774, 491, 
	513, 352, 148, 1075, 945, 1327, 1363, 402, 1071, 823, 332, 978, 1423, 1709, 
	814, 248, 255, 254, 657, 1433, 913, 168, 1040, 309, 426, 508, 997, 1455, 
	296, 1199, 1325, 183, 686, 1647, 1453, 1658, 1737, 515, 1668, 736, 1320, 
	693, 898, 1794, 1757, 478, 742, 1781, 903, 1340, 1796, 301, 854, 453, 420, 
	649, 505, 339, 1632, 114, 761, 721, 76, 836, 224, 417, 1323, 1763, 1545, 
	1736, 405, 1276, 448, 684, 1705, 669, 419]

digits_cosine_approx_gains = [1464.3202, 409.8125, 355.8973, 337.1536, 338.603, 
	334.6332, 327.4835, 332.0559, 336.986, 324.492, 332.9814, 329.3203, 328.0008, 
	324.2584, 318.6579, 319.5194, 312.6497, 312.8434, 321.7955, 314.3747, 
	322.7808, 317.7776, 307.2386, 311.6358, 305.2872, 323.2564, 303.5155, 
	307.3318, 311.2204, 315.855, 311.7815, 302.8176, 302.4257, 298.5372, 
	299.4686, 294.7835, 307.7774, 294.6846, 306.0374, 306.9195, 293.0625, 
	295.0703, 291.1347, 292.6244, 302.8187, 295.166, 288.5487, 292.3242, 
	285.8501, 285.0467, 285.9749, 284.8052, 285.898, 293.2744, 293.7613, 
	281.2778, 286.542, 282.3119, 291.9742, 281.4734, 279.1016, 283.671, 
	280.9888, 285.1791, 278.7845, 288.5932, 282.5049, 276.7215, 286.8471, 
	283.8127, 283.0595, 283.2037, 278.5408, 281.077, 279.7131, 276.8673, 
	274.2516, 276.2174, 276.6875, 276.6524, 272.7957, 274.0943, 272.21, 
	273.1331, 270.5371, 268.6535, 270.6709, 270.2242, 273.2398, 267.4328, 
	267.6318, 266.2455, 266.4325, 267.2235, 265.6243, 264.7812, 267.3172, 
	265.0701, 262.8586, 262.9712]

digits_cosine_stochastic_ranking = [1081, 1014, 1386, 770, 567, 137, 723, 1491, 
	1274, 1492, 1728, 1456, 186, 1448, 386, 148, 891, 1759, 1424, 587, 191, 629, 
	1507, 1084, 1473, 946, 518, 638, 1739, 502, 1537, 1227, 689, 88, 238, 1667, 
	1785, 1067, 1461, 1222, 1099, 607, 364, 1572, 1195, 217, 1034, 208, 84, 1128, 
	425, 345, 626, 843, 1070, 83, 1449, 1071, 1644, 1392, 1415, 449, 802, 1348, 
	1553, 175, 1455, 1770, 1395, 1032, 879, 1220, 1137, 129, 754, 1695, 1459, 782, 
	549, 1069, 260, 834, 517, 919, 1622, 700, 424, 1685, 245, 1339, 1152, 1212, 
	1425, 937, 1665, 291, 1535, 701, 1508, 1219]

digits_cosine_stochastic_gains = [1155.9385, 254.8155, 360.3138, 282.6132, 
	236.3255, 266.8341, 350.2697, 322.4336, 206.871, 238.7681, 262.7077, 
	295.0445, 263.0537, 307.3779, 303.8213, 365.73, 213.7843, 317.021, 
	259.0621, 262.3689, 220.9726, 250.0447, 273.275, 254.7376, 236.2742, 
	206.0356, 262.7957, 240.9132, 262.5503, 242.8142, 293.9653, 283.4526, 
	193.3127, 269.5029, 235.4814, 270.887, 262.0968, 272.0255, 262.5282, 
	234.3473, 245.3481, 270.3055, 235.1375, 186.677, 210.5494, 255.2077, 
	255.2398, 288.0392, 242.4625, 240.773, 273.5747, 283.8104, 247.1762, 
	255.0, 210.2734, 232.8511, 251.2321, 298.9408, 262.5918, 246.5624, 
	241.7203, 269.0865, 277.7033, 241.7952, 248.3954, 251.7519, 286.4213, 
	245.3896, 248.4624, 232.8217, 263.4509, 225.2678, 206.3469, 246.6863, 
	231.6677, 269.3534, 268.5371, 252.8494, 244.3021, 296.5371, 221.1392, 
	263.0767, 212.6504, 204.1959, 233.5965, 266.8103, 306.1608, 221.5655, 
	207.5582, 209.8983, 217.4947, 228.2527, 244.365, 231.299, 213.8048, 
	213.5623, 248.0197, 252.3993, 174.956, 210.9899]

digits_cosine_sample_ranking = [424, 615, 1030, 514, 1747, 509, 1766, 768, 818, 269, 
	185, 1295, 890, 138, 1774, 852, 255, 1709, 1069, 522, 248, 1327, 168, 923, 423, 
	945, 183, 1658, 823, 491, 898, 1325, 478, 1075, 742, 1040, 352, 978, 903, 1071, 
	1781, 426, 1796, 332, 1794, 913, 1647, 301, 814, 417, 505, 1340, 420, 1668, 1423, 
	721, 684, 854, 309, 1632, 1199, 736, 1455, 224, 836, 114, 1453, 405, 1737, 296, 
	500, 339, 1705, 693, 761, 997, 1763, 508, 1276, 1026, 1323, 816, 448, 1596, 453, 
	419, 1726, 1545, 547, 1736, 1678, 370, 669, 686, 208, 515, 1284, 1759, 126, 74]

digits_cosine_sample_gains = [1464.3202, 409.8125, 357.7419, 349.7638, 344.019, 
	333.6616, 331.3564, 330.7733, 328.7989, 327.4326, 326.3364, 325.0169, 323.9197, 
	322.4965, 321.0188, 319.4314, 317.9578, 316.7807, 316.228, 315.0714, 314.6453, 
	313.9382, 313.1161, 310.1544, 309.103, 308.2081, 306.9797, 305.9571, 305.218, 
	304.5775, 304.0841, 303.3675, 302.3388, 301.3635, 300.06, 299.4714, 298.9754, 
	298.4509, 297.7341, 297.245, 296.1614, 295.4268, 294.7741, 294.0317, 292.4587, 
	291.8016, 291.3888, 290.926, 290.3355, 289.686, 288.8004, 288.2831, 287.6514, 
	287.0812, 286.4034, 285.4221, 284.8761, 284.2295, 283.6803, 283.0425, 282.4516, 
	281.8893, 281.3542, 280.6605, 280.1731, 279.7562, 279.0967, 278.4143, 277.7055, 
	277.2929, 276.8125, 276.2901, 275.857, 275.2616, 274.7802, 274.2633, 273.8096, 
	273.3452, 272.7006, 271.9561, 271.3491, 270.9224, 270.4983, 269.9322, 269.5459, 
	269.0294, 268.583, 267.9683, 267.4714, 266.9448, 266.634, 266.0754, 265.5448, 
	264.9206, 264.4581, 264.0199, 263.2466, 262.8614, 262.3174, 261.748]

digits_cosine_modular_ranking = [424, 148, 615, 1747, 1030, 1766, 818, 1363, 
	768, 509, 1774, 138, 1295, 402, 255, 1069, 1709, 852, 248, 1327, 168, 459, 
	890, 513, 899, 945, 452, 183, 423, 923, 1658, 1325, 491, 898, 742, 478, 
	352, 514, 823, 1040, 978, 903, 426, 1071, 1796, 269, 1794, 332, 1781, 657, 
	913, 814, 1340, 417, 254, 505, 1668, 1320, 1423, 684, 1433, 309, 114, 649, 
	420, 1455, 1453, 1647, 508, 224, 1632, 1705, 1763, 296, 301, 1737, 515, 
	836, 370, 1026, 997, 500, 1199, 736, 686, 1596, 76, 1757, 816, 693, 448, 
	339, 547, 721, 1726, 1678, 854, 405, 249, 1323]

digits_cosine_modular_gains = [1464.3202, 379.668, 373.6321, 354.1949, 345.765, 
	332.1021, 329.4361, 327.2775, 331.9908, 330.0034, 322.7988, 324.0683, 
	323.538, 335.74, 318.7241, 318.0685, 316.6991, 319.3237, 315.4338, 
	314.4588, 313.767, 327.8905, 319.4338, 313.3453, 314.5868, 309.0388, 
	317.2022, 306.7897, 309.8489, 307.1568, 308.8679, 303.1924, 304.2899, 
	302.6674, 299.7269, 303.5833, 299.1892, 304.3989, 300.4859, 297.4696, 
	296.7541, 296.2466, 294.5349, 295.234, 293.8342, 300.5235, 291.2804, 
	291.9456, 291.9794, 291.5869, 289.3387, 288.8333, 287.6984, 287.7781, 
	286.9242, 288.5137, 285.6325, 292.0567, 284.4706, 283.2179, 283.3712, 
	282.0468, 280.8433, 283.41, 281.9369, 279.4637, 278.3677, 281.9862, 
	276.8678, 277.6657, 278.8126, 275.824, 275.296, 274.9688, 278.1863, 274.86, 
	271.9642, 274.5517, 271.13, 271.3635, 271.4965, 271.554, 273.1052, 
	272.9791, 268.1288, 268.6275, 268.0731, 268.5637, 267.7105, 268.0723, 
	267.3179, 267.6352, 265.4032, 268.3584, 264.5365, 263.7466, 281.4908, 
	265.798, 260.3703, 263.9768]

# Test some similarity functions

def test_digits_euclidean_naive():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='euclidean', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_lazy():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='euclidean', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_two_stage():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='euclidean', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_naive():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='corr', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_lazy():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='corr', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_two_stage():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='corr', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_naive():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_lazy():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_two_stage():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_precomputed_naive():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='naive')
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_precomputed_lazy():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='lazy')
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_precomputed_two_stage():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='two-stage')
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

# Test with initialization

def test_digits_euclidean_naive_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='euclidean', optimizer='naive', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:20], digits_euclidean_ranking[5:25])
	assert_array_almost_equal(model.gains[:20], digits_euclidean_gains[5:25], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_lazy_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='euclidean', optimizer='lazy', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_euclidean_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_euclidean_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_two_stage_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='euclidean', optimizer='two-stage', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_euclidean_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_euclidean_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_naive_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='corr', optimizer='naive', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_lazy_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='corr', optimizer='lazy', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_two_stage_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='corr', optimizer='two-stage', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_naive_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='naive', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_lazy_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='lazy', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_two_stage_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='two-stage', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_precomputed_naive_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='naive', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_precomputed_lazy_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='lazy', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_precomputed_two_stage_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='two-stage', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

# Test all optimizers

def test_digits_cosine_greedi_nn():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='greedi', 
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:50], digits_cosine_greedi_ranking[:50])
	assert_array_almost_equal(model.gains[:50], digits_cosine_greedi_gains[:50], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ll():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='greedi', 
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ln():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='greedi', 
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_nl():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='greedi', 
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_approximate():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='approximate-lazy', random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_stochastic():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='stochastic', random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_sample():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='sample', random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_modular():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='modular', random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

# Using Optimizer Objects

def test_digits_cosine_naive_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=NaiveGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_lazy_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=LazyGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_two_stage_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=TwoStageGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_nn_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=GreeDi(optimizer1='naive', 
			optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ll_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=GreeDi(optimizer1='lazy', 
			optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ln_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=GreeDi(optimizer1='lazy', 
			optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_nl_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=GreeDi(optimizer1='naive', 
			optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_approximate_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=ApproximateLazyGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_stochastic_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=StochasticGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_sample_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=SampleGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_modular_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=ModularGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

# Test all optimizers on sparse data

def test_digits_cosine_naive_sparse():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='naive')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_lazy_sparse():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='lazy')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_two_stage_sparse():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='two-stage')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_greedi_nn_sparse():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='greedi', optimizer_kwds={
		'optimizer1': 'naive', 'optimizer2': 'naive'}, random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_ll_sparse():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='greedi', optimizer_kwds={
		'optimizer1': 'lazy', 'optimizer2': 'lazy'}, random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)

def test_digits_cosine_greedi_ln_sparse():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='greedi', 
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_nl_sparse():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='greedi', 
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)

def test_digits_cosine_approximate_sparse():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='approximate-lazy', 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)

def test_digits_cosine_stochastic_sparse():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='stochastic', random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)

def test_digits_cosine_sample_sparse():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='sample', random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)

def test_digits_cosine_modular_sparse():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='precomputed', optimizer='modular', random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)
