import scipy
import numpy

try:
	import cupy
except:
	import numpy as cupy

from apricot import SaturatedCoverageSelection
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

digits_corr_ranking = [424, 615, 148, 1363, 1747, 1030, 1766, 1327, 818, 1295, 
	1774, 509, 138, 255, 852, 945, 248, 1709, 402, 768, 1069, 899, 1658, 183, 
	168, 823, 890, 1325, 923, 452, 657, 426, 269, 1040, 459, 513, 1647, 742, 
	1794, 514, 1071, 1320, 114, 814, 478, 903, 1632, 1423, 423, 649, 1199, 
	1340, 978, 1781, 448, 254, 1617, 301, 249, 339, 816, 913, 997, 898, 1737, 
	508, 1688, 352, 491, 1346, 721, 1453, 332, 1757, 1455, 684, 1409, 1668, 
	1752, 836, 1796, 761, 1763, 1433, 515, 1760, 224, 74, 1323, 846, 1705, 76, 
	1352, 417, 123, 1678, 102, 1026, 1596, 1726]

digits_corr_gains = [736.794, 728.7945, 726.0041, 721.1145, 718.5609, 717.6458, 
	709.1748, 708.2061, 705.2339, 702.4296, 701.5875, 700.3459, 696.0746, 
	695.4157, 693.8984, 693.8288, 693.3028, 689.4966, 685.9117, 685.0159, 
	682.5688, 679.713, 676.2974, 676.0061, 673.5937, 670.5503, 668.8547, 
	668.4239, 666.6083, 664.9959, 663.5193, 662.774, 660.7986, 660.3805, 
	659.2781, 659.2689, 659.1084, 657.6543, 653.8686, 653.2742, 651.7059, 
	650.3496, 649.7362, 648.6875, 647.2088, 646.7462, 645.4554, 643.6405, 
	643.2375, 641.2061, 640.5451, 640.536, 638.406, 637.9054, 637.4893, 
	636.7768, 636.2113, 635.0522, 634.1, 633.9916, 632.4109, 632.0167, 
	631.4498, 630.8923, 630.6696, 629.6054, 629.5616, 628.5034, 628.2588, 
	628.0632, 627.9028, 627.7535, 627.5283, 627.3801, 626.2458, 626.0258, 
	625.5498, 625.5167, 624.071, 623.6636, 623.408, 623.1698, 623.0551, 
	621.8752, 621.3288, 621.1112, 619.3935, 619.3573, 618.5579, 617.6756, 
	617.3202, 617.227, 616.728, 615.7226, 614.7519, 614.7003, 614.2642, 
	613.366, 613.0137, 612.0439]

digits_euclidean_ranking = [945, 426, 923, 1026, 448, 1327, 1423, 114, 255, 
	1295, 148, 515, 1363, 955, 699, 1583, 1544, 547, 768, 404, 686, 296, 814, 
	654, 607, 1443, 378, 459, 1491, 1433, 1455, 674, 773, 1637, 737, 254, 
	1453, 1058, 293, 183, 816, 478, 264, 742, 482, 1658, 1346, 394, 1320, 
	248, 1537, 997, 269, 684, 170, 1040, 514, 1763, 1450, 933, 274, 829, 
	284, 508, 120, 1529, 505, 1617, 651, 419, 40, 544, 1383, 1486, 405, 1361, 
	879, 913, 205, 424, 53, 872, 835, 1532, 281, 332, 445, 539, 836, 339, 330, 
	1123, 509, 383, 1720, 697, 1696, 138, 668, 1401]

digits_euclidean_gains = [7448636.0, 7444064.0, 7419997.0, 7360582.0, 
	7331093.0, 7319478.0, 7300111.0, 7299052.0, 7292268.0, 7291891.0, 
	7272959.0, 7268681.0, 7253865.0, 7229771.0, 7228501.0, 7225051.0, 
	7223169.0, 7216252.0, 7213040.0, 7207400.0, 7203153.0, 7198716.0, 
	7197844.0, 7178426.0, 7171971.0, 7168551.0, 7156279.0, 7155638.0, 
	7154794.0, 7153284.0, 7150967.0, 7150588.0, 7138543.0, 7132505.0, 
	7130026.0, 7129684.0, 7123532.0, 7120791.0, 7120338.0, 7117866.0, 
	7117503.0, 7109115.0, 7108884.0, 7103022.0, 7097284.0, 7092916.0, 
	7088701.0, 7087043.0, 7081955.0, 7079572.0, 7071873.0, 7070167.0, 
	7070015.0, 7068784.0, 7068111.0, 7068048.0, 7061470.0, 7058259.0, 
	7058225.0, 7055013.0, 7053514.0, 7052815.0, 7052430.0, 7052118.0, 
	7051703.0, 7050078.0, 7050046.0, 7049387.0, 7047321.0, 7045948.0, 
	7045937.0, 7044698.0, 7043791.0, 7041027.0, 7038518.0, 7037093.0, 
	7035658.0, 7032707.0, 7028424.0, 7025738.0, 7023211.0, 7023044.0, 
	7021550.0, 7017815.0, 7017340.0, 7017282.0, 7015014.0, 7010927.0, 
	7008295.0, 7007353.0, 7006586.0, 6999532.0, 6998834.0, 6998447.0, 
	6995552.0, 6992601.0, 6992582.0, 6992160.0, 6984849.0, 6984833.0]

digits_cosine_ranking = [424, 148, 615, 1747, 1030, 1766, 818, 1363, 768, 509, 
	1774, 138, 1295, 402, 255, 1069, 1709, 852, 248, 1327, 168, 459, 890, 513, 
	899, 945, 452, 183, 423, 923, 1658, 1325, 491, 898, 742, 478, 352, 514, 823, 
	1040, 978, 903, 426, 1071, 1796, 269, 1794, 332, 1781, 657, 913, 814, 1340, 
	417, 254, 505, 1668, 1320, 1423, 684, 1433, 309, 114, 649, 420, 1455, 1453, 
	1647, 508, 224, 1632, 1705, 1763, 296, 301, 1737, 515, 836, 370, 1026, 997, 
	500, 1199, 736, 686, 1596, 76, 1757, 816, 693, 448, 339, 547, 721, 1726, 
	1678, 854, 405, 249, 1323]

digits_cosine_gains = [1126.631, 1119.2491, 1114.6764, 1114.0146, 1110.8235, 
	1110.1998, 1109.0872, 1099.6581, 1099.4984, 1097.6181, 1092.4825, 1092.2165, 
	1090.7929, 1085.771, 1085.6908, 1082.2874, 1082.2755, 1081.4922, 1080.6593, 
	1080.6257, 1079.8097, 1077.3409, 1076.2455, 1074.9927, 1073.112, 1067.7319, 
	1066.2536, 1064.4357, 1060.5119, 1060.4477, 1058.5315, 1058.294, 1057.8585, 
	1053.4475, 1053.4192, 1052.5984, 1052.5837, 1052.1729, 1052.11, 1051.9781, 
	1050.8002, 1050.2236, 1050.1859, 1049.3187, 1046.4134, 1046.1143, 1045.4878, 
	1045.0879, 1044.6597, 1042.6095, 1042.2074, 1038.2103, 1037.3861, 1035.6361, 
	1035.4754, 1034.9157, 1034.6184, 1033.9479, 1033.9346, 1033.8343, 1032.9832, 
	1030.7758, 1030.742, 1030.6396, 1030.3893, 1029.5897, 1029.4052, 1028.2788, 
	1028.2653, 1028.0231, 1027.5809, 1027.3385, 1027.3248, 1026.9142, 1025.5112, 
	1024.7626, 1024.1854, 1024.0951, 1023.6384, 1023.0316, 1022.314, 1022.2542, 
	1021.7879, 1021.4765, 1020.9766, 1020.9071, 1020.4493, 1020.3874, 1018.9646, 
	1018.9303, 1018.9035, 1017.9999, 1017.917, 1016.8509, 1016.8133, 1016.5919, 
	1016.1574, 1015.4388, 1015.015, 1014.406]

digits_cosine_greedi_ranking = [424, 148, 138, 818, 509, 1363, 1766, 1069, 
	1030, 1747, 768, 615, 945, 1295, 248, 168, 513, 478, 1276, 899, 514, 452, 
	898, 1781, 1040, 332, 649, 301, 224, 1455, 749, 547, 392, 1793, 162, 25, 
	16, 28, 32, 34, 255, 1325, 459, 1071, 1340, 657, 846, 1726, 296, 684, 669, 
	761, 74, 1026, 1647, 854, 1772, 1323, 548, 0, 3, 20, 35, 37, 1327, 742, 
	1668, 423, 1760, 249, 775, 699, 339, 1410, 126, 1723, 1736, 667, 485, 455, 
	457, 468, 1558, 9, 11, 17, 19, 852, 814, 309, 1537, 453, 448, 836, 1759, 
	208, 451, 748, 445, 1792]

digits_cosine_greedi_gains = [1126.631, 1119.2491, 1092.2165, 1109.0872, 
	1097.6181, 1099.6581, 1110.1998, 1082.2874, 1110.8235, 1114.0146, 
	1099.4984, 1114.6764, 1067.7319, 1090.7929, 1080.6593, 1079.8097, 
	1074.9927, 1052.5984, 1011.7786, 1073.112, 1052.1729, 1066.2536, 
	1053.4475, 1044.6597, 1051.9781, 1045.0879, 1030.6396, 1025.5112, 
	1028.0231, 1029.5897, 1002.4851, 1017.917, 994.7059, 988.0163, 916.1074, 
	828.9046, 791.3804, 949.2322, 929.9305, 848.3833, 1085.6908, 1058.294, 
	1077.3409, 1049.3187, 1037.3861, 1042.6095, 1011.0501, 1016.8133, 
	1026.9142, 1033.8343, 1012.7852, 1011.4212, 1005.6484, 1023.0316, 
	1028.2788, 1016.1574, 984.3362, 1014.406, 907.879, 873.4687, 871.881, 
	945.3447, 910.6865, 843.0415, 1080.6257, 1053.4192, 1034.6184, 1060.5119, 
	1001.9779, 1015.015, 1009.8758, 1005.6549, 1017.9999, 988.039, 1005.4937, 
	987.8275, 1010.1136, 976.1268, 990.0549, 980.8194, 981.1644, 976.0369, 
	905.9908, 918.82, 817.0471, 974.5432, 743.412, 1081.4922, 1038.2103, 
	1030.7758, 1007.5848, 999.7418, 1018.9035, 1024.0951, 1009.6634, 
	1000.3178, 987.1787, 986.6804, 1007.4064, 989.1554]

digits_cosine_approx_ranking = [424, 148, 615, 1747, 1030, 1766, 818, 1363, 
	768, 509, 1774, 138, 1295, 402, 255, 1069, 1709, 852, 248, 1327, 168, 
	459, 890, 513, 899, 945, 452, 183, 423, 923, 1658, 1325, 491, 898, 742, 
	478, 352, 514, 823, 1040, 978, 903, 426, 1071, 1796, 269, 1794, 332, 
	1781, 657, 913, 814, 1340, 417, 254, 505, 1668, 1320, 1423, 684, 1433, 
	309, 114, 649, 420, 1455, 1453, 1647, 508, 224, 1632, 1705, 1763, 296, 
	301, 1737, 515, 836, 370, 1026, 997, 500, 1199, 736, 686, 1596, 76, 
	1757, 816, 693, 448, 339, 547, 721, 1726, 1678, 854, 405, 249, 1323]

digits_cosine_approx_gains = [1126.631, 1119.2491, 1114.6764, 1114.0146, 
	1110.8235, 1110.1998, 1109.0872, 1099.6581, 1099.4984, 1097.6181, 
	1092.4825, 1092.2165, 1090.7929, 1085.771, 1085.6908, 1082.2874, 
	1082.2755, 1081.4922, 1080.6593, 1080.6257, 1079.8097, 1077.3409, 
	1076.2455, 1074.9927, 1073.112, 1067.7319, 1066.2536, 1064.4357, 
	1060.5119, 1060.4477, 1058.5315, 1058.294, 1057.8585, 1053.4475, 
	1053.4192, 1052.5984, 1052.5837, 1052.1729, 1052.11, 1051.9781, 
	1050.8002, 1050.2236, 1050.1859, 1049.3187, 1046.4134, 1046.1143, 
	1045.4878, 1045.0879, 1044.6597, 1042.6095, 1042.2074, 1038.2103, 
	1037.3861, 1035.6361, 1035.4754, 1034.9157, 1034.6184, 1033.9479, 
	1033.9346, 1033.8343, 1032.9832, 1030.7758, 1030.742, 1030.6396, 
	1030.3893, 1029.5897, 1029.4052, 1028.2788, 1028.2653, 1028.0231, 
	1027.5809, 1027.3385, 1027.3248, 1026.9142, 1025.5112, 1024.7626, 
	1024.1854, 1024.0951, 1023.6384, 1023.0316, 1022.314, 1022.2542, 
	1021.7879, 1021.4765, 1020.9766, 1020.9071, 1020.4493, 1020.3874, 
	1018.9646, 1018.9303, 1018.9035, 1017.9999, 1017.917, 1016.8509, 
	1016.8133, 1016.5919, 1016.1574, 1015.4388, 1015.015, 1014.406]

digits_cosine_stochastic_ranking = [1081, 1014, 1386, 770, 567, 137, 723, 
	1491, 1274, 1492, 1728, 1456, 186, 1448, 386, 148, 891, 1759, 1424, 587, 
	191, 629, 1507, 1084, 1473, 946, 518, 638, 1739, 502, 1537, 1227, 689, 
	88, 238, 1667, 1785, 1067, 1461, 1222, 1099, 607, 364, 1572, 1195, 217, 
	1034, 208, 84, 1128, 425, 345, 626, 843, 1070, 83, 1449, 1071, 1644, 
	1392, 1415, 449, 802, 1348, 1553, 175, 1455, 1770, 1395, 1032, 879, 
	1220, 1137, 129, 754, 1695, 1459, 782, 549, 1069, 260, 834, 517, 919, 
	1622, 700, 424, 1685, 245, 1339, 1152, 1212, 1425, 937, 1665, 291, 1535, 
	701, 1508, 1219]

digits_cosine_stochastic_gains = [889.4143, 783.6639, 859.115, 791.2481, 
	778.8383, 833.7737, 994.1403, 964.5607, 675.0311, 787.2753, 867.9847, 
	868.3858, 828.9468, 922.5752, 884.1671, 1119.2491, 723.1304, 1009.6634, 
	869.1274, 881.9403, 744.8806, 852.139, 929.622, 871.6638, 756.9338, 
	696.1716, 903.8396, 821.9581, 887.2578, 833.9313, 1007.5848, 965.7079, 
	660.9638, 886.8445, 807.7042, 927.6503, 897.7933, 941.881, 907.081, 
	800.3562, 851.6937, 938.5545, 821.2862, 649.5475, 732.8538, 874.5872, 
	859.9888, 1000.3178, 840.2265, 844.489, 957.844, 963.6975, 869.599, 
	900.5326, 742.8335, 827.2343, 883.1609, 1049.3187, 933.9916, 879.6359, 
	862.5694, 952.1149, 988.7095, 858.4486, 889.6854, 903.7716, 1029.5897, 
	883.6901, 894.9335, 839.2439, 954.7655, 818.5447, 743.4008, 891.9863, 
	830.7171, 978.0894, 968.5476, 920.4653, 893.0029, 1082.2874, 800.2828, 
	959.0716, 763.6543, 743.3563, 857.9948, 981.3566, 1126.631, 817.5022, 
	754.2749, 775.0972, 802.8483, 844.344, 908.9335, 857.8347, 791.2102, 
	795.907, 921.8723, 943.3603, 650.8075, 794.2805]

digits_cosine_sample_ranking =  [424, 615, 1747, 1030, 1766, 818, 768, 509, 
	1774, 138, 1295, 255, 1069, 1709, 852, 248, 1327, 168, 890, 945, 183, 423, 
	923, 1658, 1325, 491, 898, 742, 478, 352, 514, 823, 1040, 978, 903, 426, 
	1071, 1796, 269, 1794, 332, 1781, 913, 814, 1340, 417, 505, 1668, 1423, 
	684, 309, 114, 420, 1455, 1453, 1647, 508, 224, 1632, 1705, 1763, 296, 
	301, 1737, 515, 836, 370, 1026, 997, 500, 1199, 736, 686, 1596, 816, 693, 
	448, 339, 547, 721, 1726, 1678, 854, 405, 1323, 419, 773, 669, 1276, 1284, 
	761, 846, 185, 1443, 1736, 40, 775, 1759, 1305, 654]

digits_cosine_sample_gains = [1126.631, 1114.6764, 1114.0146, 1110.8235, 
	1110.1998, 1109.0872, 1099.4984, 1097.6181, 1092.4825, 1092.2165, 1090.7929, 
	1085.6908, 1082.2874, 1082.2755, 1081.4922, 1080.6593, 1080.6257, 1079.8097, 
	1076.2455, 1067.7319, 1064.4357, 1060.5119, 1060.4477, 1058.5315, 1058.294, 
	1057.8585, 1053.4475, 1053.4192, 1052.5984, 1052.5837, 1052.1729, 1052.11, 
	1051.9781, 1050.8002, 1050.2236, 1050.1859, 1049.3187, 1046.4134, 1046.1143, 
	1045.4878, 1045.0879, 1044.6597, 1042.2074, 1038.2103, 1037.3861, 1035.6361, 
	1034.9157, 1034.6184, 1033.9346, 1033.8343, 1030.7758, 1030.742, 1030.3893, 
	1029.5897, 1029.4052, 1028.2788, 1028.2653, 1028.0231, 1027.5809, 1027.3385, 
	1027.3248, 1026.9142, 1025.5112, 1024.7626, 1024.1854, 1024.0951, 1023.6384, 
	1023.0316, 1022.314, 1022.2542, 1021.7879, 1021.4765, 1020.9766, 1020.9071, 
	1018.9646, 1018.9303, 1018.9035, 1017.9999, 1017.917, 1016.8509, 1016.8133, 
	1016.5919, 1016.1574, 1015.4388, 1014.406, 1014.1314, 1012.8243, 1012.7852, 
	1011.7786, 1011.651, 1011.4212, 1011.0501, 1010.9774, 1010.9712, 1010.1136, 
	1010.0059, 1009.8758, 1009.6634, 1009.2388, 1008.3862]

digits_cosine_modular_ranking = [424, 148, 615, 1747, 1030, 1766, 818, 1363, 
	768, 509, 1774, 138, 1295, 402, 255, 1069, 1709, 852, 248, 1327, 168, 459, 
	890, 513, 899, 945, 452, 183, 423, 923, 1658, 1325, 491, 898, 742, 478, 
	352, 514, 823, 1040, 978, 903, 426, 1071, 1796, 269, 1794, 332, 1781, 657, 
	913, 814, 1340, 417, 254, 505, 1668, 1320, 1423, 684, 1433, 309, 114, 649, 
	420, 1455, 1453, 1647, 508, 224, 1632, 1705, 1763, 296, 301, 1737, 515, 
	836, 370, 1026, 997, 500, 1199, 736, 686, 1596, 76, 1757, 816, 693, 448, 
	339, 547, 721, 1726, 1678, 854, 405, 249, 1323]

digits_cosine_modular_gains = [1126.631, 1119.2491, 1114.6764, 1114.0146, 
	1110.8235, 1110.1998, 1109.0872, 1099.6581, 1099.4984, 1097.6181, 1092.4825, 
	1092.2165, 1090.7929, 1085.771, 1085.6908, 1082.2874, 1082.2755, 1081.4922, 
	1080.6593, 1080.6257, 1079.8097, 1077.3409, 1076.2455, 1074.9927, 1073.112, 
	1067.7319, 1066.2536, 1064.4357, 1060.5119, 1060.4477, 1058.5315, 1058.294, 
	1057.8585, 1053.4475, 1053.4192, 1052.5984, 1052.5837, 1052.1729, 1052.11, 
	1051.9781, 1050.8002, 1050.2236, 1050.1859, 1049.3187, 1046.4134, 1046.1143, 
	1045.4878, 1045.0879, 1044.6597, 1042.6095, 1042.2074, 1038.2103, 1037.3861, 
	1035.6361, 1035.4754, 1034.9157, 1034.6184, 1033.9479, 1033.9346, 1033.8343, 
	1032.9832, 1030.7758, 1030.742, 1030.6396, 1030.3893, 1029.5897, 1029.4052, 
	1028.2788, 1028.2653, 1028.0231, 1027.5809, 1027.3385, 1027.3248, 1026.9142, 
	1025.5112, 1024.7626, 1024.1854, 1024.0951, 1023.6384, 1023.0316, 1022.314, 
	1022.2542, 1021.7879, 1021.4765, 1020.9766, 1020.9071, 1020.4493, 1020.3874, 
	1018.9646, 1018.9303, 1018.9035, 1017.9999, 1017.917, 1016.8509, 1016.8133, 
	1016.5919, 1016.1574, 1015.4388, 1015.015, 1014.406]

# Test some similarity functions

def test_digits_euclidean_naive():
	model = SaturatedCoverageSelection(100, 'euclidean', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)

def test_digits_euclidean_lazy():
	model = SaturatedCoverageSelection(100, 'euclidean', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)

def test_digits_euclidean_two_stage():
	model = SaturatedCoverageSelection(100, 'euclidean', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)

def test_digits_corr_naive():
	model = SaturatedCoverageSelection(100, 'corr', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_corr_lazy():
	model = SaturatedCoverageSelection(100, 'corr', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_corr_two_stage():
	model = SaturatedCoverageSelection(100, 'corr', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_cosine_naive():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_lazy():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_two_stage():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_precomputed_naive():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='naive')
	model.fit(X_digits_corr_cupy)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_precomputed_lazy():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='lazy')
	model.fit(X_digits_corr_cupy)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_precomputed_two_stage():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='two-stage')
	model.fit(X_digits_corr_cupy)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

# Test with initialization

def test_digits_euclidean_naive_init():
	model = SaturatedCoverageSelection(100, 'euclidean', optimizer='naive', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:20], digits_euclidean_ranking[5:25])
	assert_array_almost_equal(model.gains[:20], digits_euclidean_gains[5:25], 4)

def test_digits_euclidean_lazy_init():
	model = SaturatedCoverageSelection(100, 'euclidean', optimizer='lazy', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_euclidean_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_euclidean_gains[5:], 4)

def test_digits_euclidean_two_stage_init():
	model = SaturatedCoverageSelection(100, 'euclidean', optimizer='two-stage', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_euclidean_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_euclidean_gains[5:], 4)

def test_digits_corr_naive_init():
	model = SaturatedCoverageSelection(100, 'corr', optimizer='naive', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)

def test_digits_corr_lazy_init():
	model = SaturatedCoverageSelection(100, 'corr', optimizer='lazy', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)

def test_digits_corr_two_stage_init():
	model = SaturatedCoverageSelection(100, 'corr', optimizer='two-stage', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)

def test_digits_cosine_naive_init():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer='naive', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_cosine_lazy_init():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer='lazy', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_cosine_two_stage_init():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer='two-stage', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_precomputed_naive_init():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='naive', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_precomputed_lazy_init():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='lazy', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_precomputed_two_stage_init():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='two-stage', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

# Test all optimizers

def test_digits_cosine_greedi_nn():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_ll():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:2], digits_cosine_greedi_ranking[:2])
	assert_array_almost_equal(model.gains[:2], digits_cosine_greedi_gains[:2], 4)

def test_digits_cosine_greedi_ln():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:2], digits_cosine_greedi_ranking[:2])
	assert_array_almost_equal(model.gains[:2], digits_cosine_greedi_gains[:2], 4)

def test_digits_cosine_greedi_nl():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:2], digits_cosine_greedi_ranking[:2])
	assert_array_almost_equal(model.gains[:2], digits_cosine_greedi_gains[:2], 4)

def test_digits_cosine_approximate():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer='approximate-lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)

def test_digits_cosine_stochastic():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer='stochastic',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)

def test_digits_cosine_sample():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer='sample',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)

def test_digits_sqrt_modular():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer='modular',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)

# Using Optimizer Objects

def test_digits_cosine_naive_object():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer=NaiveGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_lazy_object():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer=LazyGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_two_stage_object():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer=TwoStageGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_greedi_nn_object():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='naive', optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_ll_object():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='lazy', optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking[:2], digits_cosine_greedi_ranking[:2])
	assert_array_almost_equal(model.gains[:2], digits_cosine_greedi_gains[:2], 4)

def test_digits_cosine_greedi_ln_object():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='lazy', optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking[:2], digits_cosine_greedi_ranking[:2])
	assert_array_almost_equal(model.gains[:2], digits_cosine_greedi_gains[:2], 4)

def test_digits_cosine_greedi_nl_object():
	model = SaturatedCoverageSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='naive', optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking[:2], digits_cosine_greedi_ranking[:2])
	assert_array_almost_equal(model.gains[:2], digits_cosine_greedi_gains[:2], 4)

def test_digits_cosine_approximate_object():
	model = SaturatedCoverageSelection(100, 'cosine', 
		optimizer=ApproximateLazyGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)

def test_digits_cosine_stochastic_object():
	model = SaturatedCoverageSelection(100, 'cosine', 
		optimizer=StochasticGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)

def test_digits_cosine_sample_object():
	model = SaturatedCoverageSelection(100, 'cosine', 
		optimizer=SampleGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)

def test_digits_sqrt_modular_object():
	model = SaturatedCoverageSelection(100, 'cosine', 
		optimizer=ModularGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)

# Test all optimizers on sparse data

def test_digits_cosine_naive_sparse():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='naive')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_lazy_sparse():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='lazy')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_two_stage_sparse():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='two-stage')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_greedi_nn_sparse():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_ll_sparse():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking[:2], digits_cosine_greedi_ranking[:2])
	assert_array_almost_equal(model.gains[:2], digits_cosine_greedi_gains[:2], 4)

def test_digits_cosine_greedi_ln_sparse():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking[:2], digits_cosine_greedi_ranking[:2])
	assert_array_almost_equal(model.gains[:2], digits_cosine_greedi_gains[:2], 4)

def test_digits_cosine_greedi_nl_sparse():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking[:2], digits_cosine_greedi_ranking[:2])
	assert_array_almost_equal(model.gains[:2], digits_cosine_greedi_gains[:2], 4)

def test_digits_cosine_approximate_sparse():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='approximate-lazy')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)

def test_digits_cosine_stochastic_sparse():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='stochastic',
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)

def test_digits_cosine_sample_sparse():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='sample',
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)

def test_digits_sqrt_modular_sparse():
	model = SaturatedCoverageSelection(100, 'precomputed', optimizer='modular',
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)
