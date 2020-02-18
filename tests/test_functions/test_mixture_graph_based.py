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
	1363, 1327, 1766, 509, 852, 818, 890, 1774, 138, 945, 248, 255, 1320, 
	1709, 768, 823, 402, 899, 1658, 1069, 1647, 183, 923, 1325, 168, 657, 
	459, 426, 1040, 513, 761, 301, 742, 478, 1071, 1632, 814, 1199, 1794, 
	423, 114, 903, 1423, 649, 1617, 448, 1340, 339, 102, 1781, 898, 1352, 
	978, 254, 1346, 721, 816, 997, 1737, 491, 1688, 913, 249, 1757, 332, 
	41, 352, 1381, 522, 508, 1668, 1453, 1455, 1323, 836, 244, 684, 1075, 
	1409, 1796, 1433, 405, 74, 1752, 224, 1760, 417, 1639, 1276, 1763]

digits_corr_gains = [957.5322, 330.4329, 255.0681, 251.6812, 245.8333, 226.588, 
	224.3453, 220.9238, 220.6682, 217.9278, 215.407, 212.8054, 211.9841, 
	210.8033, 209.4973, 208.1613, 207.5167, 207.339, 206.6113, 205.063, 
	204.6957, 204.2828, 203.8558, 202.3952, 201.7457, 201.4431, 201.134, 
	199.7144, 199.3097, 198.8759, 197.9336, 197.5106, 196.6069, 195.5512, 
	195.0987, 193.6363, 193.1214, 191.4883, 191.2132, 190.8403, 190.5169, 
	189.7908, 189.1798, 188.3658, 188.1614, 187.6664, 187.0964, 186.7252, 
	186.5124, 186.0944, 185.6038, 185.2186, 184.7898, 184.4067, 184.2691, 
	183.0488, 182.7173, 182.4579, 182.1341, 181.6718, 181.497, 181.2048, 
	181.0382, 180.8166, 180.6607, 180.0626, 179.0588, 178.457, 178.1998, 
	177.8704, 177.6597, 177.3293, 177.1239, 176.7983, 175.9878, 175.7916, 
	175.3894, 175.1012, 174.9731, 174.7894, 174.5991, 174.4241, 174.099, 
	173.9213, 173.7503, 173.5584, 173.3832, 173.1249, 172.5493, 172.3403, 
	172.1572, 171.9564, 171.7773, 171.6148, 171.0217, 170.6465, 170.3956, 
	170.2246, 169.9262, 169.7808]

digits_euclidean_ranking = [945, 1327, 448, 923, 276, 426, 1026, 651, 1583, 
	114, 148, 1423, 1295, 255, 515, 1363, 547, 955, 1544, 232, 699, 404, 607, 
	768, 1058, 296, 686, 814, 1040, 1491, 459, 1346, 654, 668, 293, 1443, 
	378, 1459, 737, 1433, 1455, 674, 1320, 1637, 254, 773, 478, 816, 183, 
	1453, 1450, 269, 482, 742, 1658, 1617, 264, 170, 394, 205, 248, 120, 
	284, 405, 1537, 1383, 829, 514, 1361, 419, 684, 997, 1763, 274, 1529, 
	40, 505, 1486, 330, 933, 544, 508, 1696, 53, 872, 888, 1532, 879, 281, 
	339, 835, 539, 913, 335, 836, 445, 1123, 521, 697, 574]

digits_euclidean_gains = [9681446.3, 2486225.0, 2447629.6, 2298651.1, 
	2280413.6, 2243419.7, 2230756.4, 2216576.3, 2201381.4, 2190256.5, 
	2186337.1, 2182894.2, 2179074.7, 2172870.6, 2163924.2, 2157390.9, 
	2154689.3, 2148707.0, 2145270.1, 2143525.8, 2141329.7, 2136575.0, 
	2135142.0, 2130170.5, 2125738.1, 2123103.6, 2120854.6, 2118942.5, 
	2116656.8, 2114673.1, 2110386.2, 2108532.5, 2105402.1, 2102423.0, 
	2100842.6, 2098024.7, 2095986.8, 2094796.7, 2092110.5, 2087733.2, 
	2085902.8, 2083665.5, 2082049.5, 2079770.2, 2077651.4, 2074845.9, 
	2072177.8, 2070106.9, 2067316.1, 2062900.4, 2060690.0, 2058392.1, 
	2055828.5, 2054558.2, 2052098.1, 2050610.8, 2048974.4, 2045951.9, 
	2043857.4, 2042512.5, 2038604.6, 2037016.5, 2034991.9, 2033150.4, 
	2031273.2, 2029210.1, 2027697.6, 2025758.9, 2023481.9, 2021733.0, 
	2020247.3, 2018455.8, 2016679.7, 2015153.8, 2013715.5, 2011089.0, 
	2009710.6, 2008124.5, 2006537.9, 2004069.4, 2002500.6, 1999779.5, 
	1998050.2, 1996550.1, 1993801.5, 1992380.2, 1990349.5, 1988973.0, 
	1987009.1, 1984877.0, 1981779.0, 1979569.5, 1977170.0, 1975366.3, 
	1972447.6, 1970666.5, 1969333.6, 1967282.3, 1966077.5, 1964702.2]

digits_cosine_ranking = [424, 615, 1030, 402, 514, 1747, 148, 509, 1766, 818, 
	269, 1363, 768, 1295, 452, 890, 138, 1774, 852, 255, 185, 1069, 1709, 248, 
	1327, 459, 168, 513, 899, 923, 945, 183, 423, 1658, 823, 1325, 1320, 898, 
	491, 478, 742, 1040, 352, 1071, 978, 903, 426, 1796, 1781, 332, 657, 1794, 
	913, 814, 301, 1340, 417, 1647, 254, 505, 1668, 1423, 1075, 684, 649, 1433, 
	420, 761, 309, 1632, 114, 1199, 854, 224, 1455, 736, 1453, 836, 296, 1705, 
	1737, 508, 1763, 405, 500, 693, 721, 997, 339, 1026, 1757, 453, 370, 1596, 
	515, 448, 76, 816, 1276, 1726]

digits_cosine_gains = [1464.3202, 410.0482, 358.258, 353.6497, 346.7507, 
	339.787, 336.4148, 333.5438, 332.1811, 330.2384, 329.3027, 328.0655, 
	327.6607, 326.6004, 326.1212, 325.8808, 324.7, 323.7217, 322.3019, 
	321.0362, 320.7516, 319.8187, 319.3191, 318.745, 318.4019, 318.1873, 
	317.5705, 316.1732, 315.7549, 313.6033, 312.9462, 312.0017, 311.7409, 
	310.7118, 310.4647, 309.7813, 309.5379, 309.1892, 308.8144, 307.6416, 
	306.5358, 306.2859, 305.9501, 305.2998, 305.1024, 304.808, 303.8696, 
	303.339, 302.8359, 302.3359, 302.135, 301.254, 300.4924, 299.7707, 
	299.1957, 298.9899, 298.5801, 298.3693, 298.0307, 297.5431, 297.1264, 
	296.6723, 296.2451, 295.9598, 295.7504, 295.495, 295.0495, 294.4885, 
	294.0818, 293.8481, 293.3599, 293.0433, 292.716, 292.4916, 292.2588, 
	291.8813, 291.4979, 291.2437, 290.5973, 290.3212, 290.0942, 289.7756, 
	289.524, 289.0927, 288.6177, 288.4124, 288.0311, 287.7333, 287.4952, 
	287.0381, 286.8194, 286.5042, 286.2153, 285.8454, 285.6212, 285.2833, 
	285.0758, 284.8404, 284.4311, 284.2047]

digits_cosine_greedi_ranking = [424, 1766, 148, 138, 1363, 402, 945, 509, 
	1030, 768, 1069, 1295, 818, 615, 248, 1747, 890, 923, 513, 1327, 459, 
	183, 1774, 852, 426, 168, 255, 269, 1709, 899, 978, 913, 452, 514, 352, 
	742, 1658, 1325, 814, 332, 1794, 1423, 515, 903, 478, 1323, 1453, 823, 
	1796, 508, 684, 1320, 761, 1433, 1455, 898, 1040, 423, 114, 224, 1340, 
	491, 1781, 1026, 370, 657, 448, 254, 309, 1071, 420, 649, 249, 505, 1763, 
	1668, 1443, 1632, 997, 419, 686, 1647, 836, 1596, 417, 721, 296, 301, 76, 
	412, 1678, 405, 816, 40, 126, 773, 445, 1726, 1759, 1757]

digits_cosine_greedi_gains = [1464.3202, 407.837, 349.8017, 331.4684, 
	334.2798, 345.7276, 322.8184, 338.3267, 339.7626, 330.9712, 323.0624, 
	327.2921, 329.7227, 337.5731, 321.4469, 332.304, 326.8193, 321.8555, 
	320.7178, 319.6813, 327.4403, 314.4639, 322.3356, 321.02, 308.9231, 
	317.7023, 319.0262, 320.7956, 317.8779, 316.0276, 308.3987, 305.2149, 
	321.8358, 318.5405, 308.0955, 307.6035, 309.9295, 309.2946, 303.0758, 
	304.4482, 303.757, 300.9853, 297.091, 305.2705, 306.7176, 308.0639, 
	297.8782, 307.5428, 303.0562, 296.7192, 298.7042, 305.7558, 297.8347, 
	298.0027, 296.3832, 305.0132, 303.1339, 306.0614, 295.8067, 295.4247, 
	297.9822, 304.151, 299.7277, 292.615, 292.1036, 298.9919, 291.788, 
	295.9899, 293.8422, 299.854, 294.2048, 295.0042, 287.9333, 295.0895, 
	291.2427, 294.1098, 286.2728, 292.4022, 289.3947, 289.7323, 287.9358, 
	293.3957, 290.2942, 288.0004, 292.5548, 288.1055, 288.5616, 290.4996, 
	286.9228, 285.6963, 285.7955, 287.2154, 285.7055, 282.076, 290.6174, 
	282.5524, 281.173, 284.4995, 283.3196, 285.7966]

digits_cosine_approx_ranking = [424, 615, 452, 1295, 509, 890, 138, 185, 
	1030, 899, 1747, 269, 1766, 514, 1069, 183, 852, 423, 923, 768, 459, 
	818, 478, 898, 742, 1774, 1320, 1796, 491, 903, 513, 148, 352, 1075, 
	761, 836, 945, 1327, 1363, 402, 417, 1071, 448, 823, 332, 419, 978, 
	1423, 248, 1709, 1726, 814, 1705, 684, 255, 500, 657, 254, 168, 1433, 
	913, 1040, 547, 309, 426, 1026, 816, 1537, 508, 773, 1325, 775, 699, 
	997, 1199, 1455, 1658, 296, 370, 1443, 686, 1647, 251, 1737, 1795, 
	1453, 414, 1668, 1794, 736, 515, 693, 439, 1757, 1781, 943, 1340, 
	1678, 654, 8]

digits_cosine_approx_gains = [1464.3202, 410.0482, 356.3771, 337.8929, 
	339.6068, 335.8092, 328.9658, 333.5075, 338.9455, 326.5309, 335.4832, 
	331.6768, 331.0086, 327.0624, 322.0501, 316.8597, 322.8309, 315.8678, 
	315.9841, 325.7954, 318.4831, 327.3971, 311.2843, 312.1877, 310.5524, 
	321.9529, 311.8357, 308.041, 310.8214, 308.4469, 315.4218, 327.986, 
	308.4601, 302.5634, 302.4439, 301.2839, 311.5941, 315.8034, 320.6281, 
	316.7673, 301.9243, 307.6464, 297.1947, 307.3102, 303.5301, 295.0697, 
	304.4982, 299.8527, 312.7778, 313.3382, 294.8015, 299.881, 296.3183, 
	297.9857, 312.8376, 294.6131, 301.0481, 298.2246, 310.1146, 296.7523, 
	298.6021, 302.1563, 291.8602, 295.0658, 299.6214, 292.1222, 291.1276, 
	287.7381, 292.6338, 288.229, 301.9518, 287.0778, 285.4309, 290.5538, 
	292.77, 291.8672, 300.7701, 290.5972, 288.8637, 285.6077, 287.7527, 
	293.7179, 281.7985, 289.8831, 281.1396, 289.0007, 281.064, 291.7233, 
	293.3529, 289.168, 286.1936, 287.3981, 280.3543, 286.6023, 293.2677, 
	278.4735, 289.8189, 284.263, 279.9617, 278.5135]

digits_cosine_stochastic_ranking = [1081, 1014, 1386, 770, 567, 137, 723, 
	1491, 1274, 1492, 1728, 1456, 186, 1448, 386, 148, 891, 1759, 1424, 587, 
	191, 629, 1507, 1084, 1473, 946, 518, 638, 1739, 502, 1537, 1227, 689, 
	88, 238, 1667, 1785, 1067, 1461, 1222, 1099, 607, 364, 1572, 1195, 217, 
	1034, 208, 84, 1128, 425, 345, 626, 843, 1070, 83, 1449, 1071, 1644, 1392, 
	1415, 449, 802, 1348, 1553, 175, 1455, 1770, 1395, 1032, 879, 1220, 1137, 
	129, 754, 1695, 1459, 782, 549, 1069, 260, 834, 517, 919, 1622, 700, 424, 
	1685, 245, 1339, 1152, 1212, 1425, 937, 1665, 291, 1535, 701, 1508, 1219]

digits_cosine_stochastic_gains = [1155.9385, 255.0445, 360.5837, 282.9572, 
	237.0038, 267.5937, 351.2828, 323.5048, 207.8157, 240.1792, 264.2394, 
	296.4451, 264.4971, 309.0642, 305.6469, 368.4019, 215.817, 319.7344, 
	261.6537, 265.046, 223.4193, 253.3707, 276.6852, 258.4959, 238.9576, 
	208.545, 267.0264, 244.3287, 266.5229, 246.8596, 298.9806, 288.3174, 
	196.6856, 274.0576, 239.9255, 276.1321, 267.4875, 277.7008, 268.0178, 
	239.2704, 250.924, 276.7295, 240.874, 191.1981, 215.5657, 261.757, 
	261.5517, 295.652, 249.1987, 247.4519, 281.5032, 291.8243, 254.7162, 
	262.8061, 216.7785, 240.5877, 259.0937, 308.7279, 271.6278, 255.2602, 
	250.5613, 278.7828, 287.646, 250.8282, 257.7071, 261.6185, 297.7562, 
	255.3366, 258.699, 242.7107, 274.8644, 235.535, 214.8505, 257.2322, 
	241.8713, 281.5764, 280.7555, 264.4589, 256.119, 310.8894, 231.4834, 
	275.5976, 223.0074, 214.4801, 245.687, 280.8101, 322.4263, 233.4239, 
	218.2476, 221.6902, 229.3926, 241.0131, 258.5089, 244.5395, 226.0217, 
	226.1359, 262.8657, 267.5999, 185.6665, 224.6065]

digits_cosine_sample_ranking = [424, 615, 1030, 514, 1747, 509, 1766, 768, 
	818, 269, 185, 1295, 890, 138, 1774, 852, 255, 1069, 1709, 248, 1327, 
	168, 522, 923, 945, 423, 183, 1658, 823, 1325, 898, 491, 478, 352, 742, 
	1040, 978, 903, 1071, 1075, 426, 1796, 1781, 332, 1794, 913, 1647, 814, 
	301, 1340, 417, 505, 1668, 1423, 420, 684, 761, 309, 1632, 114, 1455, 
	1199, 224, 736, 854, 1453, 836, 1705, 296, 508, 1763, 1737, 405, 500, 
	693, 997, 339, 721, 1026, 370, 515, 1596, 448, 816, 1276, 1726, 686, 
	547, 1323, 1678, 419, 1736, 453, 669, 1284, 126, 1759, 846, 773, 208]

digits_cosine_sample_gains = [1464.3202, 410.0482, 358.258, 350.4226, 
	345.0145, 334.9053, 332.9049, 332.4572, 330.863, 329.3521, 328.4152, 
	327.6873, 326.7648, 325.6795, 324.5005, 323.0235, 321.8598, 320.9681, 
	320.6278, 319.787, 319.3705, 318.7466, 318.4006, 315.4277, 314.4529, 
	314.0257, 313.2866, 312.1686, 311.8558, 311.1541, 310.7365, 310.4579, 
	309.3299, 308.5392, 308.0552, 307.6216, 307.052, 306.6577, 306.431, 
	306.2149, 305.3828, 304.9012, 304.2872, 303.7758, 302.9699, 302.1412, 
	301.5048, 301.2133, 300.5002, 300.1913, 300.001, 299.3754, 298.9636, 
	298.5087, 298.2232, 297.8213, 296.8566, 296.5639, 296.3168, 295.7521, 
	295.4665, 295.1727, 294.9125, 294.6199, 294.4213, 294.0703, 293.6341, 
	293.0358, 292.7682, 292.5043, 292.1686, 291.8855, 291.5039, 291.0415, 
	290.6937, 290.4058, 290.1642, 289.9272, 289.6012, 289.03, 288.7521, 
	288.3756, 288.1675, 287.8331, 287.3571, 287.1479, 286.8341, 286.4877, 
	286.2703, 286.0339, 285.6377, 285.013, 284.6484, 284.3441, 283.9737, 
	283.6829, 283.1198, 282.6349, 282.2337, 281.8957]

digits_cosine_modular_ranking = [424, 148, 615, 1747, 1030, 1766, 818, 1363, 
	768, 509, 1774, 138, 1295, 402, 255, 1069, 1709, 852, 248, 1327, 168, 459, 
	890, 513, 899, 945, 452, 183, 423, 923, 1658, 1325, 491, 898, 742, 478, 
	352, 514, 823, 1040, 978, 903, 426, 1071, 1796, 269, 1794, 332, 1781, 657, 
	913, 814, 1340, 417, 254, 505, 1668, 1320, 1423, 684, 1433, 309, 114, 649, 
	420, 1455, 1453, 1647, 508, 224, 1632, 1705, 1763, 296, 301, 1737, 515, 
	836, 370, 1026, 997, 500, 1199, 736, 686, 1596, 76, 1757, 816, 693, 448, 
	339, 547, 721, 1726, 1678, 854, 405, 249, 1323]

digits_cosine_modular_gains = [1464.3202, 379.9404, 374.1293, 354.9668, 
	346.8089, 333.4581, 331.0816, 329.1381, 333.9057, 332.3215, 325.4346, 
	326.9149, 326.6611, 338.8581, 322.3027, 321.9334, 320.7568, 323.4786, 
	320.0627, 319.4411, 318.8936, 332.449, 324.6462, 318.7445, 320.0426, 
	315.1604, 323.2019, 313.4247, 315.8704, 313.994, 315.7646, 310.715, 
	311.3097, 310.0176, 307.879, 311.2801, 307.7226, 312.6065, 309.4064, 
	306.6542, 305.9657, 305.6768, 304.7139, 305.2449, 303.9692, 310.2705, 
	302.3735, 302.7206, 302.813, 302.7997, 300.9098, 300.1932, 299.6492, 
	299.2308, 298.9255, 300.4969, 298.0381, 304.1338, 297.3426, 296.7329, 
	296.576, 295.5439, 295.1109, 296.7876, 295.5775, 294.13, 293.4925, 
	296.553, 292.5862, 293.2399, 293.9549, 291.9159, 291.6691, 291.4597, 
	293.8281, 291.2279, 289.5196, 291.2755, 289.0039, 289.053, 288.9906, 
	289.0376, 290.6934, 290.6374, 287.0914, 287.4041, 287.0989, 287.5763, 
	286.6306, 287.2988, 286.669, 286.7735, 285.4246, 288.2117, 285.0541, 
	284.4756, 300.5956, 286.0819, 282.3362, 284.302]

# Test some similarity functions

def test_digits_euclidean_naive():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='euclidean', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)


def test_digits_euclidean_lazy():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='euclidean', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)

def test_digits_euclidean_two_stage():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='euclidean', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)

def test_digits_corr_naive():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='corr', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_corr_lazy():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='corr', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_corr_two_stage():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='corr', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_cosine_naive():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_lazy():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_two_stage():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

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

def test_digits_euclidean_lazy_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='euclidean', optimizer='lazy', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_euclidean_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_euclidean_gains[5:], 4)

def test_digits_euclidean_two_stage_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='euclidean', optimizer='two-stage', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_euclidean_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_euclidean_gains[5:], 4)

def test_digits_corr_naive_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='corr', optimizer='naive', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)

def test_digits_corr_lazy_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='corr', optimizer='lazy', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)

def test_digits_corr_two_stage_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='corr', optimizer='two-stage', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)

def test_digits_cosine_naive_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='naive', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_cosine_lazy_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='lazy', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_cosine_two_stage_init():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='two-stage', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

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

def test_digits_cosine_approximate():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='approximate-lazy', random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)

def test_digits_cosine_stochastic():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='stochastic', random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)

def test_digits_cosine_sample():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='sample', random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)

def test_digits_cosine_modular():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer='modular', random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)

# Using Optimizer Objects

def test_digits_cosine_naive_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=NaiveGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_lazy_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=LazyGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_two_stage_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=TwoStageGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_greedi_nn_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=GreeDi(optimizer1='naive', 
			optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_ll_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=GreeDi(optimizer1='lazy', 
			optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)

def test_digits_cosine_greedi_ln_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=GreeDi(optimizer1='lazy', 
			optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_nl_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=GreeDi(optimizer1='naive', 
			optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)

def test_digits_cosine_approximate_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=ApproximateLazyGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)

def test_digits_cosine_stochastic_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=StochasticGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)

def test_digits_cosine_sample_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=SampleGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)

def test_digits_sqrt_modular_object():
	model1 = FacilityLocationSelection(100)
	model2 = GraphCutSelection(100)
	model = MixtureSelection(100, [model1, model2], [1.0, 0.3],
		metric='cosine', optimizer=ModularGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)

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
