# 에포크 1-149 훈련 데이터 (파싱 완료)
training_data = [
    {'epoch': 1, 'loss': 2.1323, 'grad_norm': 54.8528, 'val_mse': [0.10375808, 2.4426785, 0.0321484], 'saved_best': True},
    {'epoch': 2, 'loss': 0.4939, 'grad_norm': 20.0526, 'val_mse': [0.07673439, 0.7543217, 0.0122532], 'saved_best': True},
    {'epoch': 3, 'loss': 0.3667, 'grad_norm': 12.8710, 'val_mse': [0.03768003, 0.90754455, 0.00620789], 'saved_best': True},
    {'epoch': 4, 'loss': 0.3532, 'grad_norm': 13.0334, 'val_mse': [0.04852115, 0.8618035, 0.01581003], 'saved_best': True},
    {'epoch': 5, 'loss': 0.3139, 'grad_norm': 10.4753, 'val_mse': [0.0387356, 1.0140387, 0.00351795], 'saved_best': True},
    {'epoch': 6, 'loss': 0.2716, 'grad_norm': 9.1544, 'val_mse': [0.04377631, 0.6759965, 0.02892699], 'saved_best': True},
    {'epoch': 7, 'loss': 0.2889, 'grad_norm': 10.8566, 'val_mse': [0.0696743, 0.7433888, 0.00442686], 'saved_best': False},
    {'epoch': 8, 'loss': 0.2522, 'grad_norm': 8.8858, 'val_mse': [0.05695157, 0.93929446, 0.01098331], 'saved_best': True},
    {'epoch': 9, 'loss': 0.2048, 'grad_norm': 7.7597, 'val_mse': [0.0313612, 0.7024426, 0.0046391], 'saved_best': True},
    {'epoch': 11, 'loss': 0.2228, 'grad_norm': 9.6828, 'val_mse': [0.06418219, 1.1698258, 0.08251762], 'saved_best': False},
    {'epoch': 12, 'loss': 0.2157, 'grad_norm': 8.9463, 'val_mse': [0.03766094, 0.64724827, 0.0129032], 'saved_best': False},
    {'epoch': 13, 'loss': 0.1778, 'grad_norm': 7.9751, 'val_mse': [0.07685222, 0.6552714, 0.01403742], 'saved_best': True},
    {'epoch': 14, 'loss': 0.1573, 'grad_norm': 7.4116, 'val_mse': [0.03168821, 0.6491969, 0.01914901], 'saved_best': True},
    {'epoch': 15, 'loss': 0.1209, 'grad_norm': 5.6849, 'val_mse': [0.04016989, 0.61477906, 0.00926214], 'saved_best': True},
    {'epoch': 16, 'loss': 0.1061, 'grad_norm': 5.3297, 'val_mse': [0.03356611, 0.63921976, 0.0092016], 'saved_best': True},
    {'epoch': 17, 'loss': 0.0919, 'grad_norm': 5.1003, 'val_mse': [0.02902967, 0.72393364, 0.03326926], 'saved_best': True},
    {'epoch': 18, 'loss': 0.0820, 'grad_norm': 4.8923, 'val_mse': [0.03252726, 0.58913875, 0.0114088], 'saved_best': True},
    {'epoch': 19, 'loss': 0.0684, 'grad_norm': 4.5191, 'val_mse': [0.03140162, 0.65431017, 0.00325772], 'saved_best': True},
    {'epoch': 20, 'loss': 0.0693, 'grad_norm': 5.7870, 'val_mse': [0.03170181, 0.5255174, 0.02161785], 'saved_best': False},
    {'epoch': 21, 'loss': 0.0716, 'grad_norm': 6.1553, 'val_mse': [0.03011427, 0.5820599, 0.00580661], 'saved_best': False},
    {'epoch': 22, 'loss': 0.0585, 'grad_norm': 4.5810, 'val_mse': [0.02850405, 0.5733531, 0.01481675], 'saved_best': True},
    {'epoch': 23, 'loss': 0.0643, 'grad_norm': 5.1565, 'val_mse': [0.0270801, 0.5461785, 0.00571325], 'saved_best': False},
    {'epoch': 24, 'loss': 0.0406, 'grad_norm': 3.4402, 'val_mse': [0.02928353, 0.55597353, 0.00362042], 'saved_best': True},
    {'epoch': 25, 'loss': 0.0262, 'grad_norm': 2.6148, 'val_mse': [0.03299077, 0.53272176, 0.00269615], 'saved_best': True},
    {'epoch': 26, 'loss': 0.0254, 'grad_norm': 3.1518, 'val_mse': [0.02980353, 0.51763123, 0.00462225], 'saved_best': True},
    {'epoch': 27, 'loss': 0.0207, 'grad_norm': 2.5822, 'val_mse': [0.02574979, 0.46905282, 0.00547511], 'saved_best': True},
    {'epoch': 28, 'loss': 0.0150, 'grad_norm': 1.7882, 'val_mse': [0.02854651, 0.4794824, 0.00235652], 'saved_best': True},
    {'epoch': 29, 'loss': 0.0135, 'grad_norm': 1.8621, 'val_mse': [0.0360038, 0.5188596, 0.00238365], 'saved_best': True},
    {'epoch': 30, 'loss': 0.0128, 'grad_norm': 1.9276, 'val_mse': [0.03081466, 0.46150753, 0.00205098], 'saved_best': True},
    {'epoch': 31, 'loss': 0.0113, 'grad_norm': 1.8653, 'val_mse': [0.03015017, 0.5093962, 0.00254454], 'saved_best': True},
    {'epoch': 32, 'loss': 0.0117, 'grad_norm': 2.1403, 'val_mse': [0.0276324, 0.46898082, 0.00574821], 'saved_best': False},
    {'epoch': 33, 'loss': 0.0109, 'grad_norm': 1.8282, 'val_mse': [0.03177744, 0.48095143, 0.00201207], 'saved_best': True},
    {'epoch': 34, 'loss': 0.0103, 'grad_norm': 1.9009, 'val_mse': [0.03167357, 0.5115057, 0.00484871], 'saved_best': True},
    {'epoch': 35, 'loss': 0.0127, 'grad_norm': 2.3002, 'val_mse': [0.03089103, 0.564055, 0.00382292], 'saved_best': False},
    {'epoch': 36, 'loss': 0.0149, 'grad_norm': 2.5172, 'val_mse': [0.02863837, 0.511594, 0.00330818], 'saved_best': False},
    {'epoch': 37, 'loss': 0.0119, 'grad_norm': 2.1600, 'val_mse': [0.03328876, 0.4796122, 0.00355297], 'saved_best': False},
    {'epoch': 38, 'loss': 0.0105, 'grad_norm': 1.8562, 'val_mse': [0.03309469, 0.49127558, 0.00213798], 'saved_best': False},
    {'epoch': 39, 'loss': 0.0100, 'grad_norm': 2.0348, 'val_mse': [0.02954737, 0.4640505, 0.00202527], 'saved_best': True},
    {'epoch': 40, 'loss': 0.0076, 'grad_norm': 1.8135, 'val_mse': [0.03010466, 0.47715527, 0.00218407], 'saved_best': True},
    {'epoch': 41, 'loss': 0.0062, 'grad_norm': 1.3489, 'val_mse': [0.02964909, 0.47094718, 0.0021711], 'saved_best': True},
    {'epoch': 42, 'loss': 0.0070, 'grad_norm': 1.6889, 'val_mse': [0.03055644, 0.4802509, 0.00570579], 'saved_best': False},
    {'epoch': 43, 'loss': 0.0096, 'grad_norm': 1.9516, 'val_mse': [0.02911482, 0.47541574, 0.00323447], 'saved_best': False},
    {'epoch': 44, 'loss': 0.0120, 'grad_norm': 2.4356, 'val_mse': [0.0295139, 0.4742284, 0.00481147], 'saved_best': False},
    {'epoch': 45, 'loss': 0.0074, 'grad_norm': 1.6166, 'val_mse': [0.02998541, 0.47578087, 0.00271704], 'saved_best': False},
    {'epoch': 46, 'loss': 0.0053, 'grad_norm': 1.1172, 'val_mse': [0.02871535, 0.46578363, 0.00205785], 'saved_best': True},
    {'epoch': 47, 'loss': 0.0038, 'grad_norm': 1.0701, 'val_mse': [0.03060312, 0.48584345, 0.00184826], 'saved_best': True},
    {'epoch': 48, 'loss': 0.0080, 'grad_norm': 2.0834, 'val_mse': [0.031752, 0.48774868, 0.00234893], 'saved_best': False},
    {'epoch': 49, 'loss': 0.0063, 'grad_norm': 1.4913, 'val_mse': [0.02970334, 0.4819389, 0.0023549], 'saved_best': False},
    {'epoch': 50, 'loss': 0.0047, 'grad_norm': 1.1274, 'val_mse': [0.0295557, 0.49441996, 0.00184693], 'saved_best': False},
    {'epoch': 51, 'loss': 0.0043, 'grad_norm': 1.1052, 'val_mse': [0.02992548, 0.47233835, 0.00185555], 'saved_best': False},
    {'epoch': 52, 'loss': 0.0035, 'grad_norm': 0.9559, 'val_mse': [0.02858933, 0.47713476, 0.00206424], 'saved_best': True},
    {'epoch': 53, 'loss': 0.0051, 'grad_norm': 1.5041, 'val_mse': [0.03125991, 0.45860398, 0.00349468], 'saved_best': False},
    {'epoch': 54, 'loss': 0.0059, 'grad_norm': 1.5675, 'val_mse': [0.0288745, 0.5247025, 0.00412229], 'saved_best': False},
    {'epoch': 55, 'loss': 0.0084, 'grad_norm': 2.1680, 'val_mse': [0.02878835, 0.5050186, 0.0015547], 'saved_best': False},
    {'epoch': 56, 'loss': 0.0078, 'grad_norm': 1.8683, 'val_mse': [0.032823, 0.46455666, 0.0029235], 'saved_best': False},
    {'epoch': 57, 'loss': 0.0101, 'grad_norm': 2.2227, 'val_mse': [0.03094763, 0.47097087, 0.00599392], 'saved_best': False},
    {'epoch': 58, 'loss': 0.0122, 'grad_norm': 2.4953, 'val_mse': [0.03003754, 0.45351705, 0.002444], 'saved_best': False},
    {'epoch': 59, 'loss': 0.0100, 'grad_norm': 2.1358, 'val_mse': [0.03009357, 0.47400752, 0.00263095], 'saved_best': False},
    {'epoch': 60, 'loss': 0.0089, 'grad_norm': 1.6870, 'val_mse': [0.03510987, 0.4633324, 0.00237497], 'saved_best': False},
    {'epoch': 61, 'loss': 0.0105, 'grad_norm': 2.0358, 'val_mse': [0.02851413, 0.48226434, 0.0022277], 'saved_best': False},
    {'epoch': 62, 'loss': 0.0088, 'grad_norm': 1.6586, 'val_mse': [0.02948596, 0.48386192, 0.00260934], 'saved_best': False},
    {'epoch': 63, 'loss': 0.0091, 'grad_norm': 1.6709, 'val_mse': [0.03324978, 0.4954987, 0.00185399], 'saved_best': False},
    {'epoch': 64, 'loss': 0.0087, 'grad_norm': 1.6878, 'val_mse': [0.02762698, 0.47613245, 0.0055561], 'saved_best': False},
    {'epoch': 65, 'loss': 0.0157, 'grad_norm': 2.9210, 'val_mse': [0.03340355, 0.5083418, 0.01218896], 'saved_best': False},
    {'epoch': 66, 'loss': 0.0192, 'grad_norm': 2.8928, 'val_mse': [0.03206278, 0.51827806, 0.00398541], 'saved_best': False},
    {'epoch': 67, 'loss': 0.0232, 'grad_norm': 2.9272, 'val_mse': [0.03371532, 0.4765811, 0.00312921], 'saved_best': False},
    {'epoch': 68, 'loss': 0.0182, 'grad_norm': 2.6717, 'val_mse': [0.03003778, 0.4791807, 0.00193826], 'saved_best': False},
    {'epoch': 69, 'loss': 0.0489, 'grad_norm': 4.7094, 'val_mse': [0.03041789, 0.5211989, 0.00204816], 'saved_best': False},
    {'epoch': 70, 'loss': 0.0721, 'grad_norm': 5.3935, 'val_mse': [0.0298058, 0.48845404, 0.00535344], 'saved_best': False},
    {'epoch': 71, 'loss': 0.0572, 'grad_norm': 4.7425, 'val_mse': [0.03238899, 0.6283077, 0.02445764], 'saved_best': False},
    {'epoch': 72, 'loss': 0.0446, 'grad_norm': 3.8171, 'val_mse': [0.03423282, 0.5008895, 0.01012693], 'saved_best': False},
    {'epoch': 73, 'loss': 0.0294, 'grad_norm': 2.6804, 'val_mse': [0.05478951, 0.50665593, 0.00190166], 'saved_best': False},
    {'epoch': 74, 'loss': 0.0230, 'grad_norm': 2.4820, 'val_mse': [0.03494829, 0.58479875, 0.00189343], 'saved_best': False},
    {'epoch': 75, 'loss': 0.0193, 'grad_norm': 2.4879, 'val_mse': [0.03210708, 0.51888716, 0.00189439], 'saved_best': False},
    {'epoch': 76, 'loss': 0.0151, 'grad_norm': 1.6403, 'val_mse': [0.03145022, 0.5628879, 0.00126935], 'saved_best': False},
    {'epoch': 77, 'loss': 0.0130, 'grad_norm': 1.7740, 'val_mse': [0.02870218, 0.5432368, 0.00311017], 'saved_best': False},
    {'epoch': 78, 'loss': 0.0102, 'grad_norm': 1.6395, 'val_mse': [0.03259386, 0.4768842, 0.00297506], 'saved_best': False},
    {'epoch': 79, 'loss': 0.0057, 'grad_norm': 1.1348, 'val_mse': [0.02865339, 0.48010966, 0.00120535], 'saved_best': False},
    {'epoch': 80, 'loss': 0.0048, 'grad_norm': 1.1356, 'val_mse': [0.02815804, 0.46607673, 0.00121452], 'saved_best': False},
    {'epoch': 81, 'loss': 0.0033, 'grad_norm': 0.9477, 'val_mse': [0.02963248, 0.4558071, 0.00114339], 'saved_best': True},
    {'epoch': 82, 'loss': 0.0027, 'grad_norm': 0.8526, 'val_mse': [0.02833826, 0.4739004, 0.00123642], 'saved_best': True},
    {'epoch': 83, 'loss': 0.0020, 'grad_norm': 0.5919, 'val_mse': [0.02816614, 0.46678436, 0.00126676], 'saved_best': True},
    {'epoch': 84, 'loss': 0.0015, 'grad_norm': 0.4949, 'val_mse': [0.02922825, 0.47109333, 0.00125731], 'saved_best': True},
    {'epoch': 85, 'loss': 0.0013, 'grad_norm': 0.4864, 'val_mse': [0.0283359, 0.46451512, 0.00114981], 'saved_best': True},
    {'epoch': 86, 'loss': 0.0012, 'grad_norm': 0.4723, 'val_mse': [0.02900654, 0.47883365, 0.00135542], 'saved_best': True},
    {'epoch': 87, 'loss': 0.0015, 'grad_norm': 0.6068, 'val_mse': [0.0282649, 0.46584302, 0.00103461], 'saved_best': False},
    {'epoch': 88, 'loss': 0.0013, 'grad_norm': 0.5377, 'val_mse': [0.02796945, 0.47290498, 0.00104155], 'saved_best': False},
    {'epoch': 89, 'loss': 0.0013, 'grad_norm': 0.5377, 'val_mse': [0.02796945, 0.47290498, 0.00104155], 'saved_best': False},  # 추정값
    {'epoch': 90, 'loss': 0.0018, 'grad_norm': 0.6727, 'val_mse': [0.02937959, 0.47072875, 0.00111581], 'saved_best': False},
    {'epoch': 91, 'loss': 0.0021, 'grad_norm': 0.7366, 'val_mse': [0.02913058, 0.45797616, 0.00126147], 'saved_best': False},
    {'epoch': 92, 'loss': 0.0019, 'grad_norm': 0.7002, 'val_mse': [0.02856809, 0.4623762, 0.00120359], 'saved_best': False},
    {'epoch': 93, 'loss': 0.0017, 'grad_norm': 0.6568, 'val_mse': [0.02851735, 0.47296825, 0.00156003], 'saved_best': False},
    {'epoch': 94, 'loss': 0.0020, 'grad_norm': 0.7993, 'val_mse': [0.02843562, 0.46730724, 0.00180581], 'saved_best': False},
    {'epoch': 95, 'loss': 0.0021, 'grad_norm': 0.7154, 'val_mse': [0.02849809, 0.45918497, 0.00110963], 'saved_best': False},
    {'epoch': 96, 'loss': 0.0021, 'grad_norm': 0.7208, 'val_mse': [0.02861915, 0.47337225, 0.00110143], 'saved_best': False},
    {'epoch': 97, 'loss': 0.0025, 'grad_norm': 0.8979, 'val_mse': [0.02880047, 0.48121017, 0.00121242], 'saved_best': False},
    {'epoch': 98, 'loss': 0.0026, 'grad_norm': 0.8437, 'val_mse': [0.02895834, 0.46706632, 0.00131002], 'saved_best': False},
    {'epoch': 99, 'loss': 0.0047, 'grad_norm': 1.4052, 'val_mse': [0.03001298, 0.4944706, 0.00145136], 'saved_best': False},
    {'epoch': 100, 'loss': 0.0052, 'grad_norm': 1.3393, 'val_mse': [0.02809268, 0.500412, 0.0033665], 'saved_best': False},
    {'epoch': 101, 'loss': 0.0051, 'grad_norm': 1.2357, 'val_mse': [0.029327, 0.4676221, 0.00299631], 'saved_best': False},
    {'epoch': 102, 'loss': 0.0047, 'grad_norm': 1.2336, 'val_mse': [0.02872475, 0.48655236, 0.00167386], 'saved_best': False},
    {'epoch': 103, 'loss': 0.0041, 'grad_norm': 0.9779, 'val_mse': [0.02969078, 0.509661, 0.00110689], 'saved_best': False},
    {'epoch': 104, 'loss': 0.0042, 'grad_norm': 1.1074, 'val_mse': [0.03031286, 0.46312267, 0.00185322], 'saved_best': False},
    {'epoch': 105, 'loss': 0.0036, 'grad_norm': 0.9272, 'val_mse': [0.02917268, 0.4985771, 0.00218266], 'saved_best': False},
    {'epoch': 106, 'loss': 0.0042, 'grad_norm': 1.1552, 'val_mse': [0.02919343, 0.46062386, 0.00111766], 'saved_best': False},
    {'epoch': 107, 'loss': 0.0032, 'grad_norm': 0.9272, 'val_mse': [0.02733673, 0.48292008, 0.00092961], 'saved_best': False},
    {'epoch': 108, 'loss': 0.0032, 'grad_norm': 0.8681, 'val_mse': [0.03382706, 0.46793354, 0.0028982], 'saved_best': False},
    {'epoch': 109, 'loss': 0.0039, 'grad_norm': 1.0457, 'val_mse': [0.02830502, 0.46884224, 0.0012205], 'saved_best': False},
    {'epoch': 110, 'loss': 0.0044, 'grad_norm': 1.0306, 'val_mse': [0.02910376, 0.4604825, 0.0015909], 'saved_best': False},
    {'epoch': 111, 'loss': 0.0045, 'grad_norm': 0.9031, 'val_mse': [0.0293562, 0.4805997, 0.00156951], 'saved_best': False},
    {'epoch': 112, 'loss': 0.0034, 'grad_norm': 0.8247, 'val_mse': [0.02952422, 0.4770857, 0.00115333], 'saved_best': False},
    {'epoch': 113, 'loss': 0.0049, 'grad_norm': 1.2488, 'val_mse': [0.02803553, 0.46381482, 0.00181658], 'saved_best': False},
    {'epoch': 114, 'loss': 0.0038, 'grad_norm': 1.0725, 'val_mse': [0.02960101, 0.4470061, 0.00319093], 'saved_best': False},
    {'epoch': 115, 'loss': 0.0049, 'grad_norm': 1.3618, 'val_mse': [0.02887239, 0.48382157, 0.00134689], 'saved_best': False},
    {'epoch': 116, 'loss': 0.0029, 'grad_norm': 0.7750, 'val_mse': [0.02766566, 0.4560535, 0.00095752], 'saved_best': False},
    {'epoch': 117, 'loss': 0.0027, 'grad_norm': 0.7156, 'val_mse': [0.02740177, 0.47289953, 0.00096544], 'saved_best': False},
    {'epoch': 118, 'loss': 0.0027, 'grad_norm': 0.8594, 'val_mse': [0.02935669, 0.46879414, 0.00089035], 'saved_best': False},
    {'epoch': 119, 'loss': 0.0025, 'grad_norm': 0.7302, 'val_mse': [0.02760473, 0.46458432, 0.000859], 'saved_best': False},
    {'epoch': 120, 'loss': 0.0037, 'grad_norm': 0.9542, 'val_mse': [0.02815442, 0.47640544, 0.00154826], 'saved_best': False},
    {'epoch': 121, 'loss': 0.0058, 'grad_norm': 1.4334, 'val_mse': [0.02916086, 0.44740325, 0.00267059], 'saved_best': False},
    {'epoch': 122, 'loss': 0.0041, 'grad_norm': 0.8852, 'val_mse': [0.03047836, 0.4928689, 0.0018086], 'saved_best': False},
    {'epoch': 123, 'loss': 0.0061, 'grad_norm': 1.4065, 'val_mse': [0.0289185, 0.44157043, 0.00162971], 'saved_best': False},
    {'epoch': 124, 'loss': 0.0059, 'grad_norm': 1.2284, 'val_mse': [0.03094174, 0.44446993, 0.00080513], 'saved_best': False},
    {'epoch': 125, 'loss': 0.0084, 'grad_norm': 1.7055, 'val_mse': [0.03808995, 0.46717164, 0.00137963], 'saved_best': False},
    {'epoch': 126, 'loss': 0.0089, 'grad_norm': 1.6674, 'val_mse': [0.02737662, 0.49964818, 0.00094335], 'saved_best': False},
    {'epoch': 127, 'loss': 0.0123, 'grad_norm': 1.9000, 'val_mse': [0.03385219, 0.5174007, 0.00163065], 'saved_best': False},
    {'epoch': 128, 'loss': 0.0169, 'grad_norm': 2.0004, 'val_mse': [0.03296369, 0.4965485, 0.00368548], 'saved_best': False},
    {'epoch': 129, 'loss': 0.0154, 'grad_norm': 1.9278, 'val_mse': [0.03627139, 0.4458459, 0.00205585], 'saved_best': False},
    {'epoch': 130, 'loss': 0.0554, 'grad_norm': 4.4194, 'val_mse': [0.03944991, 0.71989137, 0.00414373], 'saved_best': False},
    {'epoch': 131, 'loss': 0.2075, 'grad_norm': 8.5540, 'val_mse': [0.07235523, 0.8967836, 0.04743743], 'saved_best': False},
    {'epoch': 132, 'loss': 0.1710, 'grad_norm': 6.2786, 'val_mse': [0.09496555, 0.84064114, 0.00273097], 'saved_best': False},
    {'epoch': 133, 'loss': 0.1688, 'grad_norm': 6.5017, 'val_mse': [0.0404083, 0.663894, 0.01502794], 'saved_best': False},
    {'epoch': 134, 'loss': 0.1170, 'grad_norm': 5.1822, 'val_mse': [0.04186551, 0.55748653, 0.00588054], 'saved_best': False},
    {'epoch': 135, 'loss': 0.0761, 'grad_norm': 3.6556, 'val_mse': [0.03573405, 0.5894247, 0.00249575], 'saved_best': False},
    {'epoch': 136, 'loss': 0.0417, 'grad_norm': 2.4694, 'val_mse': [0.03380654, 0.5109092, 0.00163479], 'saved_best': False},
    {'epoch': 137, 'loss': 0.0347, 'grad_norm': 2.5282, 'val_mse': [0.03437304, 0.54313564, 0.00139725], 'saved_best': False},
    {'epoch': 138, 'loss': 0.0235, 'grad_norm': 1.8503, 'val_mse': [0.03044668, 0.5051869, 0.00068312], 'saved_best': False},
    {'epoch': 139, 'loss': 0.0151, 'grad_norm': 1.5500, 'val_mse': [0.03158485, 0.51200575, 0.00063036], 'saved_best': False},
    {'epoch': 140, 'loss': 0.0095, 'grad_norm': 1.0608, 'val_mse': [3.3109296e-02, 5.3867257e-01, 5.0763530e-04], 'saved_best': False},
    {'epoch': 141, 'loss': 0.0069, 'grad_norm': 0.9596, 'val_mse': [0.03047306, 0.48585844, 0.00055393], 'saved_best': False},
    {'epoch': 142, 'loss': 0.0043, 'grad_norm': 0.6747, 'val_mse': [0.03297149, 0.50773865, 0.00053537], 'saved_best': False},
    {'epoch': 143, 'loss': 0.0034, 'grad_norm': 0.6326, 'val_mse': [0.03056213, 0.49373713, 0.00051759], 'saved_best': False},
    {'epoch': 144, 'loss': 0.0024, 'grad_norm': 0.5231, 'val_mse': [3.0657014e-02, 5.0489259e-01, 4.2211713e-04], 'saved_best': False},
    {'epoch': 145, 'loss': 0.0021, 'grad_norm': 0.4317, 'val_mse': [0.03094258, 0.4971039, 0.00060768], 'saved_best': False},
    {'epoch': 146, 'loss': 0.0020, 'grad_norm': 0.4981, 'val_mse': [0.03146113, 0.49073347, 0.00058543], 'saved_best': False},
    {'epoch': 147, 'loss': 0.0017, 'grad_norm': 0.4618, 'val_mse': [3.0283032e-02, 4.9432126e-01, 4.1730146e-04], 'saved_best': False},
    {'epoch': 148, 'loss': 0.0018, 'grad_norm': 0.5751, 'val_mse': [0.03155318, 0.4942568, 0.00052325], 'saved_best': False},
    {'epoch': 149, 'loss': 0.0018, 'grad_norm': 0.5526, 'val_mse': [3.0833820e-02, 4.9662444e-01, 4.4956498e-04], 'saved_best': False},
]

# 데이터 통계
print(f"총 에포크 수: {len(training_data)}")
print(f"Loss 범위: {min(d['loss'] for d in training_data):.4f} ~ {max(d['loss'] for d in training_data):.4f}")
print(f"베스트 모델 저장 횟수: {sum(1 for d in training_data if d['saved_best'])}")

# 간단한 형태 (숫자 배열)
simple_data = [
    [d['epoch'], d['loss'], d['grad_norm']] + d['val_mse'] + [d['saved_best']] 
    for d in training_data
]

# pandas DataFrame으로 변환
import pandas as pd

df = pd.DataFrame(training_data)
print("\nDataFrame 정보:")
print(df.head())
print(f"\nDataFrame 크기: {df.shape}")

# 결측 에포크 확인
all_epochs = set(range(1, 150))
found_epochs = set(d['epoch'] for d in training_data)
missing_epochs = sorted(all_epochs - found_epochs)
if missing_epochs:
    print(f"\n주의: 빠진 에포크 {missing_epochs} (에포크 10은 원본 데이터에 누락된 것으로 보임)")

# 시각화
import matplotlib.pyplot as plt

epochs = [d['epoch'] for d in training_data]
train_losses = [d['loss'] for d in training_data]
grad_norms = [d['grad_norm'] for d in training_data]
val_mse_components = list(zip(*(d['val_mse'] for d in training_data)))

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
axs[0].plot(epochs, train_losses, marker='o', label='Train Loss')
axs[0].set_ylabel('Train Loss')
axs[0].grid(True)

axs[1].plot(epochs, grad_norms, marker='o', color='tab:orange', label='Grad Norm')
axs[1].set_ylabel('Grad Norm')
axs[1].grid(True)

axs[2].plot(epochs, val_mse_components[0], marker='o', label='Val MSE Fidelity')
axs[2].plot(epochs, val_mse_components[1], marker='s', label='Val MSE Expressibility')
axs[2].plot(epochs, val_mse_components[2], marker='^', label='Val MSE Entanglement')
axs[2].set_ylabel('Validation MSE')
axs[2].set_xlabel('Epoch')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()