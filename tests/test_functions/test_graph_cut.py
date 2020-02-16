import scipy
import numpy

try:
	import cupy
except:
	import numpy as cupy

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

digits_corr_ranking = [424, 615, 148, 1363, 1747, 1030, 1766, 1327, 818, 1295, 
	1774, 509, 138, 255, 945, 852, 248, 1709, 402, 768, 899, 1069, 1658, 183, 
	168, 890, 823, 923, 1325, 452, 269, 459, 657, 426, 513, 1647, 1040, 514, 
	742, 1320, 478, 1794, 1071, 814, 423, 114, 1632, 903, 1423, 649, 1199, 448, 
	1340, 978, 301, 1617, 1781, 254, 339, 816, 997, 898, 491, 249, 913, 1737, 
	721, 1688, 1346, 332, 508, 1668, 1757, 1453, 352, 1455, 684, 836, 1323, 
	1409, 1433, 761, 1796, 1752, 74, 1352, 1760, 417, 1763, 102, 224, 515, 
	76, 846, 1705, 505, 420, 1639, 669, 1678]

digits_corr_gains = [735.794, 727.1776, 723.4118, 717.6456, 714.5894, 712.8575, 
	703.2873, 701.6207, 697.585, 694.3199, 692.4773, 691.0427, 685.9522, 
	684.4411, 683.1595, 682.3551, 679.9656, 676.0284, 673.748, 672.065, 666.96, 
	665.9173, 661.8589, 658.7166, 655.1153, 652.4249, 651.2557, 648.7175, 
	647.6779, 647.0935, 643.7318, 642.5224, 641.4525, 639.4796, 637.5326, 
	636.5814, 635.2541, 633.2874, 631.7058, 628.6688, 625.2603, 624.7565, 
	623.801, 623.0885, 620.0489, 619.5546, 618.6169, 617.7876, 615.7869, 
	614.6474, 609.5907, 608.5982, 607.8857, 607.0946, 606.1929, 605.5979, 
	604.7832, 604.307, 602.2126, 599.4264, 597.6939, 596.6442, 595.6675, 
	594.4871, 593.8232, 592.8965, 591.3181, 590.2314, 589.5026, 587.6838, 
	586.4536, 585.7414, 585.0933, 584.6202, 583.9192, 582.2645, 581.2585, 
	580.4233, 579.6279, 578.4151, 577.6573, 577.064, 576.5773, 575.7223, 
	575.1513, 573.6477, 571.6638, 571.2355, 570.2064, 569.4456, 567.8682, 
	566.3628, 565.5941, 564.7298, 563.5829, 562.1953, 561.4163, 560.8734, 
	559.9623, 559.2609]

digits_euclidean_ranking = [945, 426, 923, 1026, 448, 1327, 114, 1423, 255, 
	1295, 148, 515, 1363, 955, 699, 1583, 1544, 547, 768, 404, 686, 814, 296, 
	607, 654, 1443, 1491, 378, 459, 1433, 1455, 674, 737, 1637, 773, 1058, 254, 
	293, 816, 183, 1453, 478, 742, 482, 264, 1658, 1320, 1346, 394, 1040, 170, 
	248, 269, 651, 1450, 1537, 120, 284, 1617, 684, 997, 514, 1763, 274, 1361, 
	829, 505, 1486, 1529, 1383, 40, 933, 419, 205, 405, 508, 53, 544, 1532, 879, 
	872, 330, 281, 835, 913, 539, 445, 836, 668, 1696, 424, 383, 339, 1123, 521, 
	328, 697, 332, 335, 276]

digits_euclidean_gains = [7442701.0, 7432566.0, 7402964.0, 7338579.0, 7304265.0, 
	7289196.0, 7263385.0, 7257836.0, 7246594.0, 7240207.0, 7214976.0, 7205975.0, 
	7185754.0, 7157767.0, 7152824.0, 7147149.0, 7137390.0, 7128201.0, 7115580.0, 
	7107643.0, 7094633.0, 7088280.0, 7082879.0, 7056584.0, 7047936.0, 7038207.0, 
	7029007.0, 7023672.0, 7018052.0, 7002331.0, 6995339.0, 6988636.0, 6983739.0, 
	6970910.0, 6963087.0, 6957408.0, 6952412.0, 6937841.0, 6927533.0, 6921196.0, 
	6915329.0, 6909098.0, 6893578.0, 6888483.0, 6881666.0, 6876835.0, 6867795.0, 
	6859146.0, 6853123.0, 6845634.0, 6837558.0, 6830885.0, 6823015.0, 6817472.0, 
	6811941.0, 6806804.0, 6798061.0, 6790728.0, 6785838.0, 6781763.0, 6776709.0, 
	6767266.0, 6762120.0, 6756909.0, 6749867.0, 6745302.0, 6740836.0, 6735035.0, 
	6728625.0, 6724360.0, 6718437.0, 6713148.0, 6706748.0, 6702091.0, 6695844.0, 
	6689913.0, 6680741.0, 6675238.0, 6665088.0, 6660115.0, 6654240.0, 6650127.0, 
	6641348.0, 6631487.0, 6623535.0, 6618906.0, 6608348.0, 6602679.0, 6596166.0, 
	6590525.0, 6585700.0, 6578807.0, 6574738.0, 6568867.0, 6564040.0, 6557223.0, 
	6553122.0, 6548639.0, 6542633.0, 6532770.0]

digits_cosine_ranking = [424, 148, 615, 1747, 1030, 1766, 818, 768, 1363, 509, 
	1774, 138, 1295, 402, 255, 1709, 1069, 852, 248, 1327, 459, 168, 890, 513, 
	899, 945, 452, 183, 423, 923, 491, 1658, 1325, 898, 478, 514, 742, 352, 
	823, 1040, 978, 903, 1071, 426, 269, 1796, 1781, 332, 1794, 657, 913, 814, 
	417, 1340, 505, 1320, 254, 1668, 1423, 1433, 684, 649, 420, 309, 114, 1647, 
	1455, 1632, 1453, 301, 224, 1705, 508, 296, 1763, 1737, 836, 500, 736, 997, 
	1199, 1026, 854, 370, 515, 1757, 1596, 339, 76, 816, 693, 448, 686, 721, 
	1323, 547, 405, 185, 1726, 1678]

digits_cosine_gains = [1125.631, 1117.3413, 1112.0192, 1110.4414, 1106.3437, 
	1104.6795, 1102.6022, 1092.9112, 1091.6603, 1088.8909, 1082.6965, 1081.7277, 
	1079.3826, 1074.3775, 1072.7619, 1068.5721, 1067.5823, 1066.6426, 1064.2297, 
	1063.0179, 1061.8602, 1061.0066, 1057.8709, 1055.9954, 1053.9257, 1046.3265, 
	1045.2547, 1041.3193, 1039.4401, 1036.6569, 1034.7613, 1033.8895, 1031.5694, 
	1027.9466, 1026.6763, 1025.2278, 1023.7876, 1022.4497, 1021.375, 1020.3629, 
	1019.0949, 1017.7897, 1015.6614, 1014.5433, 1013.346, 1010.9085, 1009.0249, 
	1008.2313, 1005.9733, 1004.2337, 1002.6374, 999.3441, 997.0774, 995.9327, 
	994.7977, 994.1059, 992.9368, 991.5599, 990.0278, 988.6885, 987.0628, 
	986.4082, 985.278, 983.3302, 980.9209, 980.0652, 979.0382, 978.1864, 976.631, 
	975.6811, 974.4616, 972.7893, 972.0264, 971.1067, 970.2771, 969.203, 968.1287, 
	966.0978, 964.9124, 964.0975, 963.2177, 961.7674, 960.3681, 959.7021, 958.7057, 
	957.5836, 956.6275, 956.0556, 954.8145, 954.1901, 953.2699, 952.6243, 951.1958, 
	949.7792, 948.9328, 948.1647, 947.2608, 946.3338, 945.114, 944.0014]

digits_cosine_greedi_ranking = [424, 148, 1363, 138, 1069, 945, 509, 768, 818, 
	1766, 1295, 248, 1030, 513, 426, 615, 183, 1747, 1327, 923, 168, 852, 1774, 
	978, 255, 913, 890, 899, 402, 459, 742, 332, 1709, 515, 352, 1658, 1453, 
	514, 1423, 1325, 478, 1794, 814, 508, 1455, 903, 1796, 684, 1433, 423, 452, 
	370, 269, 114, 1026, 505, 823, 224, 1340, 491, 254, 309, 1040, 1781, 898, 
	249, 686, 1763, 1443, 448, 1071, 997, 773, 420, 417, 657, 649, 40, 296, 
	761, 1596, 955, 1320, 1632, 1705, 445, 76, 1759, 1668, 301, 816, 1647, 
	405, 699, 849, 1678, 836, 654, 1323, 547]

digits_cosine_greedi_gains = [1125.631, 1117.3413, 1096.8793, 1088.609, 
	1077.7218, 1062.3311, 1091.3468, 1092.5465, 1101.3262, 1101.519, 1081.1376, 
	1070.2316, 1099.6477, 1063.474, 1037.3137, 1101.1383, 1049.8947, 1098.6086, 
	1064.0809, 1044.0613, 1061.7054, 1063.3774, 1072.7622, 1031.672, 1064.3914, 
	1021.2589, 1055.0001, 1051.5282, 1063.0297, 1054.9165, 1028.0381, 
	1019.7981, 1055.6169, 996.7373, 1024.1256, 1030.703, 1000.2088, 1023.7117, 
	1003.9377, 1026.1193, 1022.1973, 1011.4618, 1005.4627, 992.83, 994.3828, 
	1014.8269, 1010.0701, 996.1909, 995.9212, 1023.5891, 1027.9596, 982.5373, 
	1007.3429, 988.8329, 980.708, 993.6095, 1008.8604, 983.5006, 992.4691, 
	1013.5195, 989.5312, 983.7542, 1004.5674, 997.4167, 1004.9426, 964.5325, 
	969.3357, 975.1724, 959.353, 967.3349, 996.553, 969.0909, 957.3273, 
	977.5167, 980.6415, 987.138, 975.6737, 951.1598, 966.8935, 952.7165, 
	960.8443, 945.5266, 975.7075, 966.9917, 962.5707, 944.4037, 956.3369, 
	948.0565, 970.6745, 961.3205, 952.9817, 963.0826, 949.1971, 935.9883, 
	932.8182, 947.6442, 953.7408, 933.2642, 946.5488, 945.1596]

digits_cosine_approx_ranking = [424, 148, 615, 1747, 1030, 1766, 818, 
	1363, 768, 509, 1774, 138, 1295, 402, 255, 1069, 1709, 852, 248, 
	1327, 168, 459, 890, 513, 899, 945, 452, 183, 423, 923, 1658, 1325, 
	491, 898, 742, 478, 352, 514, 823, 1040, 978, 903, 426, 1071, 1796, 
	269, 1794, 332, 1781, 657, 913, 814, 1340, 417, 254, 505, 1668, 
	1320, 1423, 684, 1433, 309, 114, 649, 420, 1455, 1453, 1647, 508, 
	224, 1632, 1705, 1763, 296, 301, 1737, 515, 836, 370, 1026, 997, 
	500, 1199, 736, 686, 1596, 76, 1757, 816, 693, 448, 339, 547, 721, 
	1726, 1678, 854, 405, 249, 1323]

digits_cosine_approx_gains = [1125.631, 1117.3413, 1112.0192, 1110.4414, 
	1106.3437, 1104.6795, 1102.6022, 1092.4561, 1092.1154, 1088.8909, 
	1082.6965, 1081.7277, 1079.3826, 1074.3775, 1072.7619, 1068.4045, 
	1067.7499, 1066.6426, 1064.2297, 1063.0179, 1061.7208, 1061.146, 
	1057.8709, 1055.9954, 1053.9257, 1046.3265, 1045.2547, 1041.3193, 
	1039.4401, 1036.6569, 1034.5424, 1032.2185, 1033.4593, 1027.9466, 
	1025.2455, 1025.9423, 1023.1393, 1023.8144, 1021.375, 1020.3629, 
	1019.0949, 1017.7897, 1015.2558, 1014.949, 1011.6303, 1012.6242, 
	1007.5107, 1008.1711, 1007.5476, 1004.2337, 1002.6374, 999.3441, 
	996.5498, 996.4603, 994.471, 993.9718, 992.2667, 992.6908, 990.0278, 
	987.7842, 987.9671, 984.7854, 982.1833, 985.0478, 983.9208, 979.7021, 
	977.9891, 978.7227, 974.8705, 975.1091, 976.1064, 972.6988, 971.7477, 
	970.9446, 972.3719, 969.203, 964.6675, 967.3492, 963.0586, 963.0667, 
	963.0003, 962.9757, 962.1606, 961.6157, 956.768, 957.3185, 956.0297, 
	956.0122, 954.8974, 953.8422, 953.4, 953.2054, 950.1792, 949.6732, 
	947.4213, 946.495, 951.4748, 946.8256, 940.7953, 945.6555]

digits_cosine_stochastic_ranking = [1081, 1014, 1386, 770, 567, 137, 723, 
	1491, 1274, 1492, 1728, 1456, 186, 1448, 386, 148, 891, 1759, 1424, 587, 
	191, 629, 1507, 1084, 1473, 946, 518, 638, 1739, 502, 1537, 1227, 689, 88, 
	238, 1667, 1785, 1067, 1461, 1222, 1099, 607, 364, 1572, 1195, 217, 1034,
	208, 84, 1128, 425, 345, 626, 843, 1070, 83, 1449, 1071, 1644, 1392, 1415, 
	449, 802, 1348, 1553, 175, 1455, 1770, 1395, 1032, 879, 1220, 1137, 129, 
	754, 1695, 1459, 782, 549, 1069, 260, 834, 517, 919, 1622, 700, 424, 1685, 
	245, 1339, 1152, 1212, 1425, 937, 1665, 291, 1535, 701, 1508, 1219]

digits_cosine_stochastic_gains = [888.4143, 781.9005, 857.2153, 789.1016, 
	775.5775, 830.2419, 989.7633, 959.9903, 670.8823, 781.5716, 861.8791, 
	862.7171, 823.1354, 915.954, 877.0817, 1109.3425, 715.3545, 999.6187, 
	859.4885, 872.0167, 735.7249, 840.0524, 917.2544, 858.1361, 746.9892, 
	686.807, 888.7371, 809.573, 873.0158, 819.4468, 989.8669, 948.4917, 
	648.7208, 870.6623, 791.8907, 909.1666, 878.8242, 921.9633, 887.7823, 
	782.9458, 832.1075, 916.1411, 801.1645, 633.4771, 715.133, 851.7562, 
	837.9491, 973.9417, 816.7727, 821.2259, 930.4159, 935.9843, 843.4655, 
	873.512, 720.1499, 800.4457, 855.9556, 1015.695, 902.8715, 849.6435, 
	832.0995, 918.794, 954.5672, 827.3385, 857.6463, 869.8829, 990.8066, 
	849.5334, 859.8116, 805.2807, 915.7205, 783.3207, 714.0554, 855.8335, 
	795.7052, 936.3462, 926.8196, 880.7669, 852.6134, 1033.4466, 764.8023, 
	916.335, 728.1311, 708.0759, 816.693, 933.6905, 1071.4125, 776.9742, 
	717.6437, 734.7908, 762.1886, 800.8094, 860.7873, 812.6997, 749.487, 
	752.995, 871.3857, 891.6916, 614.1057, 747.8919]

digits_cosine_sample_ranking = [424, 615, 1747, 1030, 1766, 818, 768, 509, 
	1774, 138, 1295, 255, 1709, 1069, 852, 248, 1327, 168, 890, 945, 183, 
	423, 923, 491, 1658, 1325, 898, 478, 514, 742, 352, 823, 1040, 978, 903, 
	426, 1071, 269, 1796, 1781, 332, 1794, 913, 814, 417, 1340, 505, 1668, 
	1423, 684, 420, 309, 1647, 1455, 114, 1632, 1453, 301, 224, 1705, 508, 
	296, 1763, 1737, 836, 500, 997, 736, 1026, 1199, 515, 370, 854, 339, 
	1596, 816, 448, 693, 686, 721, 547, 1323, 405, 185, 1726, 1678, 419, 
	669, 1276, 1736, 1284, 1759, 773, 846, 761, 1075, 1443, 126, 1688, 1537]

digits_cosine_sample_gains = [1125.631, 1112.8909, 1111.3033, 1107.1809, 
	1105.5572, 1103.4957, 1093.8138, 1090.6241, 1084.3876, 1083.4827, 
	1081.1928, 1075.2986, 1071.1206, 1070.108, 1069.0501, 1066.7516, 
	1065.5364, 1064.1852, 1061.0074, 1051.2558, 1046.8869, 1044.7768, 
	1042.1952, 1040.1766, 1039.0607, 1036.7214, 1033.0665, 1032.0248, 
	1030.6189, 1029.1516, 1027.8075, 1026.4913, 1025.5415, 1024.6649, 
	1023.0235, 1020.9489, 1020.0088, 1018.4326, 1016.4982, 1014.3648, 
	1013.6914, 1011.4747, 1008.823, 1005.4414, 1002.9037, 1001.7435, 
	1000.6435, 998.8884, 997.402, 995.2971, 994.3926, 992.2291, 989.741, 
	988.8511, 988.0681, 986.6505, 985.9041, 984.3618, 983.3554, 981.7265, 
	980.9536, 979.8119, 978.903, 977.3713, 976.7121, 974.9219, 973.584, 
	972.5756, 971.4105, 970.7025, 969.3354, 968.2987, 967.2911, 965.9186, 
	965.2119, 964.1663, 963.2723, 962.5841, 961.4689, 959.5403, 958.7148, 
	958.0178, 957.0356, 955.8384, 954.991, 954.0756, 953.1607, 951.5552, 
	950.3357, 947.629, 946.2986, 945.6759, 944.2653, 943.4489, 942.5374, 
	941.7992, 940.9055, 940.2754, 938.5729, 937.4811]

digits_cosine_modular_ranking = [424, 148, 615, 1747, 1030, 1766, 818, 1363, 
	768, 509, 1774, 138, 1295, 402, 255, 1069, 1709, 852, 248, 1327, 168, 
	459, 890, 513, 899, 945, 452, 183, 423, 923, 1658, 1325, 491, 898, 742, 
	478, 352, 514, 823, 1040, 978, 903, 426, 1071, 1796, 269, 1794, 332, 
	1781, 657, 913, 814, 1340, 417, 254, 505, 1668, 1320, 1423, 684, 1433, 
	309, 114, 649, 420, 1455, 1453, 1647, 508, 224, 1632, 1705, 1763, 296, 
	301, 1737, 515, 836, 370, 1026, 997, 500, 1199, 736, 686, 1596, 76, 
	1757, 816, 693, 448, 339, 547, 721, 1726, 1678, 854, 405, 249, 1323]

digits_cosine_modular_gains = [1125.631, 1117.3413, 1112.0192, 1110.4414, 
	1106.3437, 1104.6795, 1102.6022, 1092.4561, 1092.1154, 1088.8909, 1082.6965, 
	1081.7277, 1079.3826, 1074.3775, 1072.7619, 1068.4045, 1067.7499, 1066.6426, 
	1064.2297, 1063.0179, 1061.7208, 1061.146, 1057.8709, 1055.9954, 1053.9257, 
	1046.3265, 1045.2547, 1041.3193, 1039.4401, 1036.6569, 1034.5424, 1032.2185, 
	1033.4593, 1027.9466, 1025.2455, 1025.9423, 1023.1393, 1023.8144, 1021.375, 
	1020.3629, 1019.0949, 1017.7897, 1015.2558, 1014.949, 1011.6303, 1012.6242, 
	1007.5107, 1008.1711, 1007.5476, 1004.2337, 1002.6374, 999.3441, 996.5498, 
	996.4603, 994.471, 993.9718, 992.2667, 992.6908, 990.0278, 987.7842, 987.9671, 
	984.7854, 982.1833, 985.0478, 983.9208, 979.7021, 977.9891, 978.7227, 974.8705, 
	975.1091, 976.1064, 972.6988, 971.7477, 970.9446, 972.3719, 969.203, 964.6675, 
	967.3492, 963.0586, 963.0667, 963.0003, 962.9757, 962.1606, 961.6157, 956.768, 
	957.3185, 956.0297, 956.0122, 954.8974, 953.8422, 953.4, 953.2054, 950.1792, 
	949.6732, 947.4213, 946.495, 951.4748, 946.8256, 940.7953, 945.6555]

# Test some similarity functions

def test_digits_euclidean_naive():
	model = GraphCutSelection(100, 'euclidean', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)

def test_digits_euclidean_lazy():
	model = GraphCutSelection(100, 'euclidean', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)

def test_digits_euclidean_two_stage():
	model = GraphCutSelection(100, 'euclidean', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)

def test_digits_corr_naive():
	model = GraphCutSelection(100, 'corr', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_corr_lazy():
	model = GraphCutSelection(100, 'corr', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_corr_two_stage():
	model = GraphCutSelection(100, 'corr', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_cosine_naive():
	model = GraphCutSelection(100, 'cosine', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_lazy():
	model = GraphCutSelection(100, 'cosine', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_two_stage():
	model = GraphCutSelection(100, 'cosine', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_precomputed_naive():
	model = GraphCutSelection(100, 'precomputed', optimizer='naive')
	model.fit(X_digits_corr_cupy)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_precomputed_lazy():
	model = GraphCutSelection(100, 'precomputed', optimizer='lazy')
	model.fit(X_digits_corr_cupy)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

def test_digits_precomputed_two_stage():
	model = GraphCutSelection(100, 'precomputed', optimizer='two-stage')
	model.fit(X_digits_corr_cupy)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)

# Test with initialization

def test_digits_euclidean_naive_init():
	model = GraphCutSelection(100, 'euclidean', optimizer='naive', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:20], digits_euclidean_ranking[5:25])
	assert_array_almost_equal(model.gains[:20], digits_euclidean_gains[5:25], 4)

def test_digits_euclidean_lazy_init():
	model = GraphCutSelection(100, 'euclidean', optimizer='lazy', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_euclidean_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_euclidean_gains[5:], 4)

def test_digits_euclidean_two_stage_init():
	model = GraphCutSelection(100, 'euclidean', optimizer='two-stage', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_euclidean_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_euclidean_gains[5:], 4)

def test_digits_corr_naive_init():
	model = GraphCutSelection(100, 'corr', optimizer='naive', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)

def test_digits_corr_lazy_init():
	model = GraphCutSelection(100, 'corr', optimizer='lazy', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)

def test_digits_corr_two_stage_init():
	model = GraphCutSelection(100, 'corr', optimizer='two-stage', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)

def test_digits_cosine_naive_init():
	model = GraphCutSelection(100, 'cosine', optimizer='naive', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_cosine_lazy_init():
	model = GraphCutSelection(100, 'cosine', optimizer='lazy', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_cosine_two_stage_init():
	model = GraphCutSelection(100, 'cosine', optimizer='two-stage', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_precomputed_naive_init():
	model = GraphCutSelection(100, 'precomputed', optimizer='naive', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_precomputed_lazy_init():
	model = GraphCutSelection(100, 'precomputed', optimizer='lazy', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

def test_digits_precomputed_two_stage_init():
	model = GraphCutSelection(100, 'precomputed', optimizer='two-stage', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits_cosine_cupy)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)

# Test all optimizers

def test_digits_cosine_greedi_nn():
	model = GraphCutSelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:50], digits_cosine_greedi_ranking[:50])
	assert_array_almost_equal(model.gains[:50], digits_cosine_greedi_gains[:50], 4)

def test_digits_cosine_greedi_ll():
	model = GraphCutSelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)

def test_digits_cosine_greedi_ln():
	model = GraphCutSelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_nl():
	model = GraphCutSelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)

def test_digits_cosine_approximate():
	model = GraphCutSelection(100, 'cosine', optimizer='approximate-lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)

def test_digits_cosine_stochastic():
	model = GraphCutSelection(100, 'cosine', optimizer='stochastic',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)

def test_digits_cosine_sample():
	model = GraphCutSelection(100, 'cosine', optimizer='sample',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)

def test_digits_sqrt_modular():
	model = GraphCutSelection(100, 'cosine', optimizer='modular',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)

# Using Optimizer Objects

def test_digits_cosine_naive_object():
	model = GraphCutSelection(100, 'cosine', optimizer=NaiveGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_lazy_object():
	model = GraphCutSelection(100, 'cosine', optimizer=LazyGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_two_stage_object():
	model = GraphCutSelection(100, 'cosine', optimizer=TwoStageGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_greedi_nn_object():
	model = GraphCutSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='naive', optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_ll_object():
	model = GraphCutSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='lazy', optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)

def test_digits_cosine_greedi_ln_object():
	model = GraphCutSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='lazy', optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_nl_object():
	model = GraphCutSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='naive', optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)

def test_digits_cosine_approximate_object():
	model = GraphCutSelection(100, 'cosine', 
		optimizer=ApproximateLazyGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)

def test_digits_cosine_stochastic_object():
	model = GraphCutSelection(100, 'cosine', 
		optimizer=StochasticGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)

def test_digits_cosine_sample_object():
	model = GraphCutSelection(100, 'cosine', 
		optimizer=SampleGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)

def test_digits_sqrt_modular_object():
	model = GraphCutSelection(100, 'cosine', 
		optimizer=ModularGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)

# Test all optimizers on sparse data

def test_digits_cosine_naive_sparse():
	model = GraphCutSelection(100, 'precomputed', optimizer='naive')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_lazy_sparse():
	model = GraphCutSelection(100, 'precomputed', optimizer='lazy')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_two_stage_sparse():
	model = GraphCutSelection(100, 'precomputed', optimizer='two-stage')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)

def test_digits_cosine_greedi_nn_sparse():
	model = GraphCutSelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_ll_sparse():
	model = GraphCutSelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)

def test_digits_cosine_greedi_ln_sparse():
	model = GraphCutSelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)

def test_digits_cosine_greedi_nl_sparse():
	model = GraphCutSelection(100, 'precomputed', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)

def test_digits_cosine_approximate_sparse():
	model = GraphCutSelection(100, 'precomputed', optimizer='approximate-lazy')
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)

def test_digits_cosine_stochastic_sparse():
	model = GraphCutSelection(100, 'precomputed', optimizer='stochastic',
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)

def test_digits_cosine_sample_sparse():
	model = GraphCutSelection(100, 'precomputed', optimizer='sample',
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)

def test_digits_sqrt_modular_sparse():
	model = GraphCutSelection(100, 'precomputed', optimizer='modular',
		random_state=0)
	model.fit(X_digits_cosine_sparse)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)
