import scipy
import numpy

try:
	import cupy
except:
	import numpy as cupy

from apricot import GraphCutSelection
from apricot.optimizers import NaiveGreedy, LazyGreedy, TwoStageGreedy, GreeDi, ApproximateLazyGreedy, StochasticGreedy, SampleGreedy, ModularGreedy

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
	1774, 509, 138, 945, 852, 255, 248, 402, 1709, 768, 899, 1069, 1658, 183, 
	890, 168, 452, 823, 923, 459, 269, 1325, 657, 514, 513, 1647, 426, 1040, 
	1320, 742, 478, 423, 814, 1071, 1632, 649, 1794, 903, 1423, 114, 301, 448, 
	1617, 1199, 978, 254, 339, 1340, 1781, 816, 491, 997, 898, 721, 1737, 913, 
	1346, 1688, 1323, 1668, 332, 249, 1757, 836, 1352, 1453, 74, 352, 1433, 
	761, 417, 102, 684, 1455, 508, 1796, 1752, 1409, 420, 1760, 505, 1276, 
	1563, 1639, 392, 76, 669, 846, 224, 1763]

digits_corr_gains = [735.794, 726.5607, 721.8195, 715.1766, 711.618, 709.0691, 
	698.3998, 696.0352, 690.936, 687.2101, 684.367, 682.7395, 676.8297, 
	674.7931, 673.3689, 671.6063, 667.6284, 663.8161, 662.3284, 660.1142, 
	655.2069, 650.2657, 648.4204, 642.427, 637.9989, 636.6332, 633.1892, 
	631.9442, 630.7131, 628.8502, 627.4997, 625.146, 620.3857, 618.3749, 
	616.8959, 615.4579, 613.2892, 610.446, 608.9605, 605.785, 604.3117, 
	600.4383, 599.0397, 596.8669, 594.8858, 593.2829, 591.6467, 589.8474, 
	588.5694, 586.3158, 582.4899, 580.0048, 578.9603, 576.9437, 575.7387, 
	574.489, 573.2631, 572.0652, 570.6573, 567.4419, 566.0, 564.0186, 
	562.3914, 558.7558, 557.2309, 555.9309, 553.9387, 552.4717, 550.8894, 
	549.2331, 548.0805, 546.6924, 543.974, 542.8549, 541.7866, 540.7781, 
	539.306, 538.1135, 536.767, 535.111, 534.0093, 532.9543, 531.8669, 
	530.6082, 528.9325, 527.4136, 526.2152, 524.8406, 522.2271, 520.8602, 
	520.0685, 518.3558, 516.5727, 515.0325, 514.1975, 513.274, 512.0967, 
	511.0568, 509.8714, 508.1492]

digits_euclidean_ranking = [945, 426, 923, 1026, 448, 1327, 114, 1423, 255, 
	1295, 148, 515, 1363, 1583, 699, 955, 1544, 547, 768, 404, 814, 296, 686, 
	607, 1491, 378, 459, 654, 1443, 737, 1433, 1455, 1058, 1637, 674, 254, 773, 
	293, 478, 816, 183, 1320, 482, 1658, 1453, 1040, 742, 651, 170, 1346, 394, 
	264, 1450, 120, 248, 1617, 269, 284, 1537, 1361, 1486, 274, 205, 505, 1763, 
	684, 829, 514, 1383, 1529, 53, 997, 40, 330, 405, 419, 1532, 933, 872, 508, 
	281, 544, 668, 335, 879, 521, 539, 328, 835, 1696, 1459, 1187, 836, 276, 
	697, 574, 1123, 899, 383, 339]

digits_euclidean_gains = [7442701.0, 7427003.0, 7391866.0, 7322511.0, 7283372.0, 
	7264849.0, 7233653.0, 7221496.0, 7206855.0, 7194458.0, 7162928.0, 7149204.0, 
	7123578.0, 7094334.0, 7082878.0, 7072750.0, 7057546.0, 7046085.0, 7024055.0, 
	7013821.0, 6994383.0, 6983357.0, 6971936.0, 6947132.0, 6927637.0, 6916662.0, 
	6905791.0, 6895329.0, 6884316.0, 6869019.0, 6848647.0, 6836834.0, 6825512.0, 
	6816680.0, 6806097.0, 6790051.0, 6775976.0, 6761279.0, 6743658.0, 6733564.0, 
	6721609.0, 6705102.0, 6694665.0, 6684379.0, 6675061.0, 6664743.0, 6654607.0, 
	6642962.0, 6631890.0, 6619494.0, 6607142.0, 6597715.0, 6588550.0, 6576890.0, 
	6564869.0, 6555612.0, 6544766.0, 6535529.0, 6520758.0, 6511304.0, 6496512.0, 
	6486627.0, 6476195.0, 6466599.0, 6456078.0, 6445523.0, 6436816.0, 6425703.0, 
	6417178.0, 6407329.0, 6397208.0, 6386930.0, 6378182.0, 6369183.0, 6359381.0, 
	6348449.0, 6334670.0, 6324898.0, 6309631.0, 6298731.0, 6288283.0, 6279927.0, 
	6269136.0, 6255130.0, 6246085.0, 6234993.0, 6226342.0, 6212742.0, 6204657.0, 
	6193651.0, 6184202.0, 6174295.0, 6165640.0, 6155770.0, 6142794.0, 6133382.0, 
	6124361.0, 6114045.0, 6106200.0, 6095592.0]

digits_cosine_ranking = [424, 148, 615, 1747, 1030, 1766, 818, 768, 1363, 509, 
	138, 1774, 1295, 402, 255, 852, 1709, 1069, 459, 248, 1327, 168, 890, 513, 
	899, 452, 945, 423, 183, 491, 923, 1658, 1325, 898, 478, 514, 742, 352, 
	823, 1040, 978, 903, 269, 1071, 426, 1796, 1781, 332, 657, 1794, 913, 814, 
	417, 1320, 505, 1340, 254, 1668, 649, 1433, 420, 1423, 684, 309, 1647, 
	1632, 301, 114, 1455, 1453, 1737, 224, 836, 854, 296, 500, 736, 1705, 1763, 
	1199, 997, 508, 339, 185, 1026, 693, 816, 1757, 1323, 448, 1596, 721, 76, 
	405, 419, 1276, 370, 669, 547, 686]

digits_cosine_gains = [1125.631, 1116.4334, 1110.3619, 1107.8682, 1102.8639, 
	1100.1593, 1097.1171, 1087.324, 1084.6625, 1081.1638, 1073.9924, 1072.1571, 
	1068.9723, 1063.9841, 1060.8331, 1055.983, 1054.3242, 1052.2316, 1050.1474, 
	1047.3745, 1045.0676, 1043.2035, 1040.4963, 1037.9981, 1035.7395, 1026.7499, 
	1024.427, 1020.7876, 1017.7835, 1014.1761, 1012.3542, 1010.2475, 1005.8448, 
	1003.4457, 1001.7543, 999.2828, 995.1559, 993.3157, 991.6399, 989.7477, 
	988.3895, 986.3558, 984.3293, 981.7637, 978.3895, 976.4035, 974.3902, 
	972.3747, 968.5371, 965.7794, 964.0673, 961.4778, 959.5186, 957.8536, 
	955.7932, 952.7764, 951.3981, 949.5014, 947.7064, 945.6255, 943.8797, 942.2532, 
	938.6848, 936.8846, 934.2809, 932.5912, 930.9676, 928.0929, 926.5528, 923.4524, 
	921.9201, 920.488, 918.9356, 916.8895, 915.4592, 913.7898, 912.329, 910.635, 
	908.8096, 906.9795, 905.6998, 903.8216, 901.8546, 900.5586, 897.6146, 896.1891, 
	894.563, 893.1249, 891.6622, 890.0585, 888.5845, 887.2155, 885.1279, 883.9388, 
	882.1048, 880.3597, 878.6431, 876.9752, 875.6371, 873.2418]

digits_cosine_greedi_ranking = [424, 148, 138, 1363, 768, 509, 945, 818, 1069, 
	1766, 1030, 513, 1747, 615, 1295, 183, 426, 248, 402, 923, 890, 168, 459, 899, 
	852, 978, 1327, 913, 1774, 514, 255, 478, 1709, 332, 1658, 814, 352, 1423, 452, 
	742, 423, 269, 1453, 1325, 903, 505, 420, 491, 1794, 898, 1320, 1455, 1781, 
	1433, 1340, 1796, 508, 649, 1323, 684, 823, 309, 301, 126, 657, 1759, 1668, 
	997, 1413, 419, 1040, 417, 669, 185, 1071, 1276, 412, 761, 1075, 1555, 1596, 
	1443, 55, 836, 1647, 457, 500, 1690, 693, 1532, 1385, 1199, 666, 700, 1393, 
	109, 17, 178, 1474, 407]

digits_cosine_greedi_gains = [1125.631, 1116.4334, 1087.7212, 1093.3808, 
	1091.6622, 1087.8283, 1056.2655, 1096.2936, 1067.2725, 1093.8382, 1092.9129, 
	1056.0357, 1092.6489, 1091.7967, 1065.8791, 1038.2608, 1022.3374, 1050.4396, 
	1056.4113, 1028.7627, 1043.8886, 1043.4331, 1043.713, 1036.7286, 1041.9576, 
	1010.2869, 1036.2449, 998.2041, 1045.0749, 1008.026, 1035.4611, 1005.7826, 
	1030.4246, 992.4873, 1005.1995, 984.2387, 994.3666, 976.4607, 1007.983, 
	990.2323, 1000.6849, 985.0151, 962.7264, 989.9898, 982.1059, 967.6117, 
	962.8401, 987.1467, 968.8581, 979.2295, 961.9455, 950.3317, 966.8441, 
	952.4936, 955.5363, 961.4793, 939.8794, 947.9843, 931.9156, 943.1238, 
	961.481, 938.7155, 936.0861, 918.2042, 948.451, 916.9374, 938.5427, 923.1422, 
	901.0512, 915.0605, 947.6232, 931.8231, 910.7304, 911.7605, 940.6555, 
	905.9804, 900.3013, 899.7678, 899.6988, 884.6817, 904.1478, 892.203, 
	891.2392, 904.6856, 908.0614, 869.8637, 899.1515, 877.5662, 894.6796, 
	873.0077, 866.0964, 894.8458, 854.875, 854.2783, 869.7832, 847.8489, 
	848.853, 847.3636, 852.911, 853.4502]

digits_cosine_approx_ranking = [424, 148, 615, 1747, 1030, 1766, 818, 1363, 
	768, 509, 1774, 138, 1295, 402, 255, 1069, 1709, 852, 248, 1327, 168, 
	459, 890, 513, 899, 945, 452, 183, 423, 923, 1658, 1325, 491, 898, 742, 
	478, 352, 514, 823, 1040, 978, 903, 426, 1071, 1796, 269, 1794, 332, 1781, 
	657, 913, 814, 1340, 417, 254, 505, 1668, 1320, 1423, 684, 1433, 309, 114, 
	649, 420, 1455, 1453, 1647, 224, 1632, 301, 736, 854, 1276, 185, 208, 
	1793, 1292, 508, 296, 1705, 1737, 836, 1763, 500, 1199, 997, 1026, 339, 
	370, 1757, 693, 816, 515, 1596, 76, 448, 721, 547, 686]

digits_cosine_approx_gains = [1125.631, 1116.4334, 1110.3619, 1107.8682, 
	1102.8639, 1100.1593, 1097.1171, 1086.2541, 1085.7324, 1081.1638, 1073.9105, 
	1072.239, 1068.9723, 1063.9841, 1060.8331, 1055.5216, 1054.2243, 1052.793, 
	1048.8001, 1046.41, 1044.6319, 1045.951, 1040.4963, 1037.9981, 1035.7395, 
	1025.9212, 1025.2558, 1019.2028, 1019.3683, 1013.8661, 1011.5533, 1007.143, 
	1010.0601, 1003.4457, 998.0718, 1000.2861, 994.6949, 996.4559, 991.6399, 
	989.7477, 988.3895, 986.3558, 981.3256, 981.5793, 977.8471, 980.1341, 
	970.5337, 972.2543, 971.4356, 966.8578, 964.0673, 961.4778, 956.7136, 
	958.2845, 954.4666, 954.0279, 950.9149, 952.4338, 947.121, 942.7341, 
	943.9511, 939.795, 934.6246, 940.4561, 938.4522, 930.8145, 927.5731, 
	930.1666, 924.6783, 927.0749, 925.4307, 919.227, 918.0896, 909.8963, 
	911.2968, 899.4732, 888.9093, 848.1238, 908.8201, 909.2536, 907.2134, 
	907.3827, 906.0856, 902.7765, 901.9748, 900.5461, 897.8239, 894.334, 
	894.1257, 889.1623, 890.5617, 888.7265, 886.5046, 881.969, 883.4505, 
	881.6255, 880.4515, 879.048, 875.9963, 873.1088]

digits_cosine_stochastic_ranking = [1081, 1014, 1386, 770, 567, 137, 723, 
	1491, 1274, 1492, 1728, 1456, 186, 1448, 386, 148, 891, 1759, 1424, 
	587, 191, 629, 1507, 1084, 1473, 946, 518, 638, 1739, 502, 1537, 1227, 
	689, 88, 238, 1667, 1785, 1067, 1461, 1222, 1099, 607, 364, 1572, 1195, 
	217, 1034, 208, 84, 1128, 425, 345, 626, 843, 1070, 83, 1449, 1071, 
	1644, 1392, 1415, 449, 802, 1348, 1553, 175, 1455, 1770, 1395, 1032, 
	879, 1220, 1137, 129, 754, 1695, 1459, 782, 549, 1069, 260, 834, 517, 
	919, 1622, 700, 424, 1685, 245, 1339, 1152, 1212, 1425, 937, 1665, 291, 
	1535, 701, 1508, 1219]

digits_cosine_stochastic_gains = [888.4143, 781.137, 856.3156, 787.955, 
	773.3167, 827.71, 986.3862, 956.4199, 667.7336, 776.868, 856.7735, 
	858.0485, 818.324, 910.3328, 870.9962, 1100.4359, 708.5786, 990.574, 
	850.8497, 863.0931, 727.5691, 828.9658, 905.8869, 845.6085, 738.0446, 
	678.4423, 874.6345, 798.1879, 859.7738, 805.9624, 973.1491, 932.2754, 
	637.4778, 855.48, 777.0772, 891.683, 860.8551, 903.0456, 869.4836, 
	766.5354, 813.5212, 894.7277, 782.0428, 618.4067, 698.4122, 829.9252, 
	816.9094, 948.5656, 794.3189, 798.9629, 903.9878, 909.2711, 818.3321, 
	847.4914, 698.4663, 774.6571, 829.7503, 983.0712, 872.7514, 820.6512, 
	802.6295, 886.4731, 921.4248, 797.2285, 826.6073, 836.9943, 953.0235, 
	816.3766, 825.6897, 772.3176, 877.6755, 749.0967, 685.71, 820.6807, 
	761.6934, 895.603, 886.0917, 842.0685, 813.2238, 985.6058, 730.3218, 
	874.5984, 693.6079, 673.7955, 776.3912, 887.0244, 1017.1941, 737.4462, 
	682.0124, 695.4844, 722.5289, 758.2748, 813.6412, 768.5648, 708.7639, 
	711.0829, 821.8991, 841.0228, 578.404, 702.5032]

digits_cosine_sample_ranking = [424, 615, 1747, 1030, 1766, 818, 768, 509, 
	138, 1774, 1295, 255, 1709, 852, 1069, 248, 1327, 168, 890, 945, 423, 
	183, 491, 923, 1658, 1325, 478, 898, 514, 742, 352, 978, 1040, 823, 
	903, 269, 1071, 426, 1796, 1781, 332, 1794, 913, 814, 417, 505, 1340, 
	1668, 420, 1423, 684, 309, 1647, 301, 1632, 1455, 114, 1453, 224, 836, 
	1705, 1737, 500, 296, 854, 736, 997, 1763, 508, 1199, 339, 185, 1026, 
	816, 448, 693, 1323, 1596, 721, 405, 370, 419, 547, 669, 1276, 686, 515, 
	1678, 1726, 1075, 126, 1736, 1759, 55, 1284, 453, 74, 208, 846, 1688]

digits_cosine_sample_gains = [1125.631, 1112.1054, 1109.5919, 1104.5383, 
	1101.9147, 1098.9042, 1089.1292, 1084.6301, 1077.5023, 1075.5393, 
	1072.5927, 1065.9065, 1060.9657, 1059.2536, 1057.2829, 1053.8439, 
	1051.4471, 1049.5608, 1046.7694, 1035.7797, 1031.4609, 1028.9188, 
	1025.0066, 1023.4309, 1020.5899, 1016.1487, 1013.9896, 1012.147, 
	1010.0649, 1005.8841, 1004.0312, 1002.0633, 1000.7069, 998.737, 
	996.8234, 994.5026, 991.8834, 989.7757, 987.5829, 985.0698, 983.2949, 
	978.4616, 976.4386, 973.6725, 971.1713, 968.7036, 965.7687, 964.1585, 
	962.109, 960.4553, 956.461, 954.6824, 952.2032, 949.9055, 947.9428, 
	946.1786, 943.8165, 941.9986, 939.6877, 937.4197, 935.6647, 934.1413, 
	932.6361, 930.9798, 929.4286, 927.5936, 926.1213, 924.6194, 923.0118, 
	920.6783, 919.2436, 917.1787, 915.6507, 913.3613, 911.4307, 910.2685, 
	908.513, 906.8863, 905.2225, 903.4884, 902.1355, 900.4484, 898.2079, 
	896.8174, 895.2279, 893.788, 891.955, 890.5398, 888.3937, 886.4727, 
	884.9978, 883.2743, 880.9702, 879.3279, 877.1812, 875.8943, 874.1455, 
	872.718, 870.9218, 868.8899]

digits_cosine_modular_ranking = [424, 148, 615, 1747, 1030, 1766, 818, 1363, 
	768, 509, 1774, 138, 1295, 402, 255, 1069, 1709, 852, 248, 1327, 168, 459, 
	890, 513, 899, 945, 452, 183, 423, 923, 1658, 1325, 491, 898, 742, 478, 
	352, 514, 823, 1040, 978, 903, 426, 1071, 1796, 269, 1794, 332, 1781, 657, 
	913, 814, 1340, 417, 254, 505, 1668, 1320, 1423, 684, 1433, 309, 114, 649, 
	420, 1455, 1453, 1647, 508, 224, 1632, 1705, 1763, 296, 301, 1737, 515, 
	836, 370, 1026, 997, 500, 1199, 736, 686, 1596, 76, 1757, 816, 693, 448, 
	339, 547, 721, 1726, 1678, 854, 405, 249, 1323]

digits_cosine_modular_gains = [1125.631, 1116.4334, 1110.3619, 1107.8682, 
	1102.8639, 1100.1593, 1097.1171, 1086.2541, 1085.7324, 1081.1638, 1073.9105,
	1072.239, 1068.9723, 1063.9841, 1060.8331, 1055.5216, 1054.2243, 1052.793, 
	1048.8001, 1046.41, 1044.6319, 1045.951, 1040.4963, 1037.9981, 1035.7395, 
	1025.9212, 1025.2558, 1019.2028, 1019.3683, 1013.8661, 1011.5533, 1007.143, 
	1010.0601, 1003.4457, 998.0718, 1000.2861, 994.6949, 996.4559, 991.6399, 
	989.7477, 988.3895, 986.3558, 981.3256, 981.5793, 977.8471, 980.1341, 
	970.5337, 972.2543, 971.4356, 966.8578, 964.0673, 961.4778, 956.7136, 
	958.2845, 954.4666, 954.0279, 950.9149, 952.4338, 947.121, 942.7341, 
	943.9511, 939.795, 934.6246, 940.4561, 938.4522, 930.8145, 927.5731, 
	930.1666, 922.4757, 923.1952, 925.632, 919.0591, 917.1706, 915.9749, 
	920.2326, 914.6434, 906.1495, 911.6033, 903.4789, 904.1017, 904.6866, 
	904.6973, 903.5333, 902.7549, 893.5594, 894.73, 892.61, 892.637, 891.8303, 
	889.754, 888.8964, 889.4109, 883.4413, 883.4955, 879.0293, 877.3982, 
	887.7923, 879.2123, 867.5755, 877.9051]


# Test some similarity functions

def test_digits_euclidean_naive():
	model = GraphCutSelection(100, 'euclidean', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_lazy():
	model = GraphCutSelection(100, 'euclidean', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_two_stage():
	model = GraphCutSelection(100, 'euclidean', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_euclidean_ranking)
	assert_array_almost_equal(model.gains, digits_euclidean_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_naive():
	model = GraphCutSelection(100, 'corr', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_lazy():
	model = GraphCutSelection(100, 'corr', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_two_stage():
	model = GraphCutSelection(100, 'corr', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_corr_ranking)
	assert_array_almost_equal(model.gains, digits_corr_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_naive():
	model = GraphCutSelection(100, 'cosine', optimizer='naive')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_lazy():
	model = GraphCutSelection(100, 'cosine', optimizer='lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_two_stage():
	model = GraphCutSelection(100, 'cosine', optimizer='two-stage')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

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
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_lazy_init():
	model = GraphCutSelection(100, 'euclidean', optimizer='lazy', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_euclidean_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_euclidean_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_euclidean_two_stage_init():
	model = GraphCutSelection(100, 'euclidean', optimizer='two-stage', 
		initial_subset=digits_euclidean_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_euclidean_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_euclidean_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_naive_init():
	model = GraphCutSelection(100, 'corr', optimizer='naive', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_lazy_init():
	model = GraphCutSelection(100, 'corr', optimizer='lazy', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_corr_two_stage_init():
	model = GraphCutSelection(100, 'corr', optimizer='two-stage', 
		initial_subset=digits_corr_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_corr_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_corr_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_naive_init():
	model = GraphCutSelection(100, 'cosine', optimizer='naive', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_lazy_init():
	model = GraphCutSelection(100, 'cosine', optimizer='lazy', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_two_stage_init():
	model = GraphCutSelection(100, 'cosine', optimizer='two-stage', 
		initial_subset=digits_cosine_ranking[:5])
	model.fit(X_digits)
	assert_array_equal(model.ranking[:-5], digits_cosine_ranking[5:])
	assert_array_almost_equal(model.gains[:-5], digits_cosine_gains[5:], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

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
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ll():
	model = GraphCutSelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ln():
	model = GraphCutSelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'lazy', 'optimizer2': 'naive'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_nl():
	model = GraphCutSelection(100, 'cosine', optimizer='greedi',
		optimizer_kwds={'optimizer1': 'naive', 'optimizer2': 'lazy'}, 
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_approximate():
	model = GraphCutSelection(100, 'cosine', optimizer='approximate-lazy')
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_stochastic():
	model = GraphCutSelection(100, 'cosine', optimizer='stochastic',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_sample():
	model = GraphCutSelection(100, 'cosine', optimizer='sample',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_modular():
	model = GraphCutSelection(100, 'cosine', optimizer='modular',
		random_state=0)
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

# Using Optimizer Objects

def test_digits_cosine_naive_object():
	model = GraphCutSelection(100, 'cosine', optimizer=NaiveGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_lazy_object():
	model = GraphCutSelection(100, 'cosine', optimizer=LazyGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_two_stage_object():
	model = GraphCutSelection(100, 'cosine', optimizer=TwoStageGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_nn_object():
	model = GraphCutSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='naive', optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ll_object():
	model = GraphCutSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='lazy', optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_ln_object():
	model = GraphCutSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='lazy', optimizer2='naive', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_greedi_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_greedi_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_greedi_nl_object():
	model = GraphCutSelection(100, 'cosine', optimizer=GreeDi(
		optimizer1='naive', optimizer2='lazy', random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking[:30], digits_cosine_greedi_ranking[:30])
	assert_array_almost_equal(model.gains[:30], digits_cosine_greedi_gains[:30], 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_approximate_object():
	model = GraphCutSelection(100, 'cosine', 
		optimizer=ApproximateLazyGreedy())
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_approx_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_approx_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_stochastic_object():
	model = GraphCutSelection(100, 'cosine', 
		optimizer=StochasticGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_stochastic_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_stochastic_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_cosine_sample_object():
	model = GraphCutSelection(100, 'cosine', 
		optimizer=SampleGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_sample_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_sample_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

def test_digits_sqrt_modular_object():
	model = GraphCutSelection(100, 'cosine', 
		optimizer=ModularGreedy(random_state=0))
	model.fit(X_digits)
	assert_array_equal(model.ranking, digits_cosine_modular_ranking)
	assert_array_almost_equal(model.gains, digits_cosine_modular_gains, 4)
	assert_array_almost_equal(model.subset, X_digits[model.ranking])

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
