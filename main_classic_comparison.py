import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from utils_classic import *
import csv

K_dic = {
    '2':1,
    '3':0.48725,
    '4':0.32328,
    '5':0.24209,
    '6':0.19357,
    '7':0.16127,
    '8':0.13822,
    '10':0.10750,
    '16':0.06451,
    '32':0.03122,
    '64':0.01536,
    '100':0.00978,
}

IMPROVING = True
N = 64
K = K_dic[str(N)] * np.ones(N)
csv_file = open('./Autoregressive_DIM64_RUNS1000_r0_7.txt','w')
csv_writer = csv.writer(csv_file)

rand_matrix = np.random.randn(N, N)
B_mask = np.tril(rand_matrix)
B_mask[B_mask != 0] = 1
C_mask = np.tril(rand_matrix, k=-1)
C_mask[C_mask != 0] = 1

###########################
def generating_constraints(N):
    c0_dic = {
        'fun': constraint0,
        'type': 'ineq',
    }
    cL1_dic = {
        'fun': constraintL1,
        'type': 'ineq',
    }
    cU1_dic = {
        'fun': constraintU1,
        'type': 'ineq',
    }
    cL2_dic = {
        'fun': constraintL2,
        'type': 'ineq',
    }
    cU2_dic = {
        'fun': constraintU2,
        'type': 'ineq',
    }
    cL3_dic = {
        'fun': constraintL3,
        'type': 'ineq',
    }
    cU3_dic = {
        'fun': constraintU3,
        'type': 'ineq',
    }
    cL4_dic = {
        'fun': constraintL4,
        'type': 'ineq',
    }
    cU4_dic = {
        'fun': constraintU4,
        'type': 'ineq',
    }
    cL5_dic = {
        'fun': constraintL5,
        'type': 'ineq',
    }
    cU5_dic = {
        'fun': constraintU5,
        'type': 'ineq',
    }
    cL6_dic = {
        'fun': constraintL6,
        'type': 'ineq',
    }
    cU6_dic = {
        'fun': constraintU6,
        'type': 'ineq',
    }
    cL7_dic = {
        'fun': constraintL7,
        'type': 'ineq',
    }
    cU7_dic = {
        'fun': constraintU7,
        'type': 'ineq',
    }
    cL8_dic = {
        'fun': constraintL8,
        'type': 'ineq',
    }
    cU8_dic = {
        'fun': constraintU8,
        'type': 'ineq',
    }
    cL9_dic = {
        'fun': constraintL9,
        'type': 'ineq',
    }
    cU9_dic = {
        'fun': constraintU9,
        'type': 'ineq',
    }
    cL10_dic = {
        'fun': constraintL10,
        'type': 'ineq',
    }
    cU10_dic = {
        'fun': constraintU10,
        'type': 'ineq',
    }
    cL11_dic = {
        'fun': constraintL11,
        'type': 'ineq',
    }
    cU11_dic = {
        'fun': constraintU11,
        'type': 'ineq',
    }
    cL12_dic = {
        'fun': constraintL12,
        'type': 'ineq',
    }
    cU12_dic = {
        'fun': constraintU12,
        'type': 'ineq',
    }
    cL13_dic = {
        'fun': constraintL13,
        'type': 'ineq',
    }
    cU13_dic = {
        'fun': constraintU13,
        'type': 'ineq',
    }
    cL14_dic = {
        'fun': constraintL14,
        'type': 'ineq',
    }
    cU14_dic = {
        'fun': constraintU14,
        'type': 'ineq',
    }
    cL15_dic = {
        'fun': constraintL15,
        'type': 'ineq',
    }
    cU15_dic = {
        'fun': constraintU15,
        'type': 'ineq',
    }
    cL16_dic = {
        'fun': constraintL16,
        'type': 'ineq',
    }
    cU16_dic = {
        'fun': constraintU16,
        'type': 'ineq',
    }
    cL17_dic = {
        'fun': constraintL17,
        'type': 'ineq',
    }
    cU17_dic = {
        'fun': constraintU17,
        'type': 'ineq',
    }
    cL18_dic = {
        'fun': constraintL18,
        'type': 'ineq',
    }
    cU18_dic = {
        'fun': constraintU18,
        'type': 'ineq',
    }
    cL19_dic = {
        'fun': constraintL19,
        'type': 'ineq',
    }
    cU19_dic = {
        'fun': constraintU19,
        'type': 'ineq',
    }
    cL20_dic = {
        'fun': constraintL20,
        'type': 'ineq',
    }
    cU20_dic = {
        'fun': constraintU20,
        'type': 'ineq',
    }
    cL21_dic = {
        'fun': constraintL21,
        'type': 'ineq',
    }
    cU21_dic = {
        'fun': constraintU21,
        'type': 'ineq',
    }
    cL22_dic = {
        'fun': constraintL22,
        'type': 'ineq',
    }
    cU22_dic = {
        'fun': constraintU22,
        'type': 'ineq',
    }
    cL23_dic = {
        'fun': constraintL23,
        'type': 'ineq',
    }
    cU23_dic = {
        'fun': constraintU23,
        'type': 'ineq',
    }
    cL24_dic = {
        'fun': constraintL24,
        'type': 'ineq',
    }
    cU24_dic = {
        'fun': constraintU24,
        'type': 'ineq',
    }
    cL25_dic = {
        'fun': constraintL25,
        'type': 'ineq',
    }
    cU25_dic = {
        'fun': constraintU25,
        'type': 'ineq',
    }
    cL26_dic = {
        'fun': constraintL26,
        'type': 'ineq',
    }
    cU26_dic = {
        'fun': constraintU26,
        'type': 'ineq',
    }
    cL27_dic = {
        'fun': constraintL27,
        'type': 'ineq',
    }
    cU27_dic = {
        'fun': constraintU27,
        'type': 'ineq',
    }
    cL28_dic = {
        'fun': constraintL28,
        'type': 'ineq',
    }
    cU28_dic = {
        'fun': constraintU28,
        'type': 'ineq',
    }
    cL29_dic = {
        'fun': constraintL29,
        'type': 'ineq',
    }
    cU29_dic = {
        'fun': constraintU29,
        'type': 'ineq',
    }
    cL30_dic = {
        'fun': constraintL30,
        'type': 'ineq',
    }
    cU30_dic = {
        'fun': constraintU30,
        'type': 'ineq',
    }
    cL31_dic = {
        'fun': constraintL31,
        'type': 'ineq',
    }
    cU31_dic = {
        'fun': constraintU31,
        'type': 'ineq',
    }
    cL32_dic = {
        'fun': constraintL32,
        'type': 'ineq',
    }
    cU32_dic = {
        'fun': constraintU32,
        'type': 'ineq',
    }

    cL33_dic = {
        'fun': constraintL33,
        'type': 'ineq',
    }
    cU33_dic = {
        'fun': constraintU33,
        'type': 'ineq',
    }
    cL34_dic = {
        'fun': constraintL34,
        'type': 'ineq',
    }
    cU34_dic = {
        'fun': constraintU34,
        'type': 'ineq',
    }
    cL35_dic = {
        'fun': constraintL35,
        'type': 'ineq',
    }
    cU35_dic = {
        'fun': constraintU35,
        'type': 'ineq',
    }
    cL36_dic = {
        'fun': constraintL36,
        'type': 'ineq',
    }
    cU36_dic = {
        'fun': constraintU36,
        'type': 'ineq',
    }
    cL37_dic = {
        'fun': constraintL37,
        'type': 'ineq',
    }
    cU37_dic = {
        'fun': constraintU37,
        'type': 'ineq',
    }
    cL38_dic = {
        'fun': constraintL38,
        'type': 'ineq',
    }
    cU38_dic = {
        'fun': constraintU38,
        'type': 'ineq',
    }
    cL39_dic = {
        'fun': constraintL39,
        'type': 'ineq',
    }
    cU39_dic = {
        'fun': constraintU39,
        'type': 'ineq',
    }
    cL40_dic = {
        'fun': constraintL40,
        'type': 'ineq',
    }
    cU40_dic = {
        'fun': constraintU40,
        'type': 'ineq',
    }
    cL41_dic = {
        'fun': constraintL41,
        'type': 'ineq',
    }
    cU41_dic = {
        'fun': constraintU41,
        'type': 'ineq',
    }
    cL42_dic = {
        'fun': constraintL42,
        'type': 'ineq',
    }
    cU42_dic = {
        'fun': constraintU42,
        'type': 'ineq',
    }
    cL43_dic = {
        'fun': constraintL43,
        'type': 'ineq',
    }
    cU43_dic = {
        'fun': constraintU43,
        'type': 'ineq',
    }
    cL44_dic = {
        'fun': constraintL44,
        'type': 'ineq',
    }
    cU44_dic = {
        'fun': constraintU44,
        'type': 'ineq',
    }
    cL45_dic = {
        'fun': constraintL45,
        'type': 'ineq',
    }
    cU45_dic = {
        'fun': constraintU45,
        'type': 'ineq',
    }
    cL46_dic = {
        'fun': constraintL46,
        'type': 'ineq',
    }
    cU46_dic = {
        'fun': constraintU46,
        'type': 'ineq',
    }
    cL47_dic = {
        'fun': constraintL47,
        'type': 'ineq',
    }
    cU47_dic = {
        'fun': constraintU47,
        'type': 'ineq',
    }
    cL48_dic = {
        'fun': constraintL48,
        'type': 'ineq',
    }
    cU48_dic = {
        'fun': constraintU48,
        'type': 'ineq',
    }
    cL49_dic = {
        'fun': constraintL49,
        'type': 'ineq',
    }
    cU49_dic = {
        'fun': constraintU49,
        'type': 'ineq',
    }
    cL50_dic = {
        'fun': constraintL50,
        'type': 'ineq',
    }
    cU50_dic = {
        'fun': constraintU50,
        'type': 'ineq',
    }
    cL51_dic = {
        'fun': constraintL51,
        'type': 'ineq',
    }
    cU51_dic = {
        'fun': constraintU51,
        'type': 'ineq',
    }
    cL52_dic = {
        'fun': constraintL52,
        'type': 'ineq',
    }
    cU52_dic = {
        'fun': constraintU52,
        'type': 'ineq',
    }
    cL53_dic = {
        'fun': constraintL53,
        'type': 'ineq',
    }
    cU53_dic = {
        'fun': constraintU53,
        'type': 'ineq',
    }
    cL54_dic = {
        'fun': constraintL54,
        'type': 'ineq',
    }
    cU54_dic = {
        'fun': constraintU54,
        'type': 'ineq',
    }
    cL55_dic = {
        'fun': constraintL55,
        'type': 'ineq',
    }
    cU55_dic = {
        'fun': constraintU55,
        'type': 'ineq',
    }
    cL56_dic = {
        'fun': constraintL56,
        'type': 'ineq',
    }
    cU56_dic = {
        'fun': constraintU56,
        'type': 'ineq',
    }
    cL57_dic = {
        'fun': constraintL57,
        'type': 'ineq',
    }
    cU57_dic = {
        'fun': constraintU57,
        'type': 'ineq',
    }
    cL58_dic = {
        'fun': constraintL58,
        'type': 'ineq',
    }
    cU58_dic = {
        'fun': constraintU58,
        'type': 'ineq',
    }
    cL59_dic = {
        'fun': constraintL59,
        'type': 'ineq',
    }
    cU59_dic = {
        'fun': constraintU59,
        'type': 'ineq',
    }
    cL60_dic = {
        'fun': constraintL60,
        'type': 'ineq',
    }
    cU60_dic = {
        'fun': constraintU60,
        'type': 'ineq',
    }
    cL61_dic = {
        'fun': constraintL61,
        'type': 'ineq',
    }
    cU61_dic = {
        'fun': constraintU61,
        'type': 'ineq',
    }
    cL62_dic = {
        'fun': constraintL62,
        'type': 'ineq',
    }
    cU62_dic = {
        'fun': constraintU62,
        'type': 'ineq',
    }
    cL63_dic = {
        'fun': constraintL63,
        'type': 'ineq',
    }
    cU63_dic = {
        'fun': constraintU63,
        'type': 'ineq',
    }
    cL64_dic = {
        'fun': constraintL64,
        'type': 'ineq',
    }
    cU64_dic = {
        'fun': constraintU64,
        'type': 'ineq',
    }

    cL65_dic = {
        'fun': constraintL65,
        'type': 'ineq',
    }
    cU65_dic = {
        'fun': constraintU65,
        'type': 'ineq',
    }
    cL66_dic = {
        'fun': constraintL66,
        'type': 'ineq',
    }
    cU66_dic = {
        'fun': constraintU66,
        'type': 'ineq',
    }
    cL67_dic = {
        'fun': constraintL67,
        'type': 'ineq',
    }
    cU67_dic = {
        'fun': constraintU67,
        'type': 'ineq',
    }
    cL68_dic = {
        'fun': constraintL68,
        'type': 'ineq',
    }
    cU68_dic = {
        'fun': constraintU68,
        'type': 'ineq',
    }
    cL69_dic = {
        'fun': constraintL69,
        'type': 'ineq',
    }
    cU69_dic = {
        'fun': constraintU69,
        'type': 'ineq',
    }
    cL70_dic = {
        'fun': constraintL70,
        'type': 'ineq',
    }
    cU70_dic = {
        'fun': constraintU70,
        'type': 'ineq',
    }
    cL71_dic = {
        'fun': constraintL71,
        'type': 'ineq',
    }
    cU71_dic = {
        'fun': constraintU71,
        'type': 'ineq',
    }
    cL72_dic = {
        'fun': constraintL72,
        'type': 'ineq',
    }
    cU72_dic = {
        'fun': constraintU72,
        'type': 'ineq',
    }
    cL73_dic = {
        'fun': constraintL73,
        'type': 'ineq',
    }
    cU73_dic = {
        'fun': constraintU73,
        'type': 'ineq',
    }
    cL74_dic = {
        'fun': constraintL74,
        'type': 'ineq',
    }
    cU74_dic = {
        'fun': constraintU74,
        'type': 'ineq',
    }
    cL75_dic = {
        'fun': constraintL75,
        'type': 'ineq',
    }
    cU75_dic = {
        'fun': constraintU75,
        'type': 'ineq',
    }

    cL76_dic = {
        'fun': constraintL76,
        'type': 'ineq',
    }
    cU76_dic = {
        'fun': constraintU76,
        'type': 'ineq',
    }
    cL77_dic = {
        'fun': constraintL77,
        'type': 'ineq',
    }
    cU77_dic = {
        'fun': constraintU77,
        'type': 'ineq',
    }
    cL78_dic = {
        'fun': constraintL78,
        'type': 'ineq',
    }
    cU78_dic = {
        'fun': constraintU78,
        'type': 'ineq',
    }
    cL79_dic = {
        'fun': constraintL79,
        'type': 'ineq',
    }
    cU79_dic = {
        'fun': constraintU79,
        'type': 'ineq',
    }
    cL80_dic = {
        'fun': constraintL80,
        'type': 'ineq',
    }
    cU80_dic = {
        'fun': constraintU80,
        'type': 'ineq',
    }
    cL81_dic = {
        'fun': constraintL81,
        'type': 'ineq',
    }
    cU81_dic = {
        'fun': constraintU81,
        'type': 'ineq',
    }
    cL82_dic = {
        'fun': constraintL82,
        'type': 'ineq',
    }
    cU82_dic = {
        'fun': constraintU82,
        'type': 'ineq',
    }
    cL83_dic = {
        'fun': constraintL83,
        'type': 'ineq',
    }
    cU83_dic = {
        'fun': constraintU83,
        'type': 'ineq',
    }
    cL84_dic = {
        'fun': constraintL84,
        'type': 'ineq',
    }
    cU84_dic = {
        'fun': constraintU84,
        'type': 'ineq',
    }
    cL85_dic = {
        'fun': constraintL85,
        'type': 'ineq',
    }
    cU85_dic = {
        'fun': constraintU85,
        'type': 'ineq',
    }
    cL86_dic = {
        'fun': constraintL86,
        'type': 'ineq',
    }
    cU86_dic = {
        'fun': constraintU86,
        'type': 'ineq',
    }
    cL87_dic = {
        'fun': constraintL87,
        'type': 'ineq',
    }
    cU87_dic = {
        'fun': constraintU87,
        'type': 'ineq',
    }
    cL88_dic = {
        'fun': constraintL88,
        'type': 'ineq',
    }
    cU88_dic = {
        'fun': constraintU88,
        'type': 'ineq',
    }
    cL89_dic = {
        'fun': constraintL89,
        'type': 'ineq',
    }
    cU89_dic = {
        'fun': constraintU89,
        'type': 'ineq',
    }
    cL90_dic = {
        'fun': constraintL90,
        'type': 'ineq',
    }
    cU90_dic = {
        'fun': constraintU90,
        'type': 'ineq',
    }
    cL91_dic = {
        'fun': constraintL91,
        'type': 'ineq',
    }
    cU91_dic = {
        'fun': constraintU91,
        'type': 'ineq',
    }
    cL92_dic = {
        'fun': constraintL92,
        'type': 'ineq',
    }
    cU92_dic = {
        'fun': constraintU92,
        'type': 'ineq',
    }
    cL93_dic = {
        'fun': constraintL93,
        'type': 'ineq',
    }
    cU93_dic = {
        'fun': constraintU93,
        'type': 'ineq',
    }
    cL94_dic = {
        'fun': constraintL94,
        'type': 'ineq',
    }
    cU94_dic = {
        'fun': constraintU94,
        'type': 'ineq',
    }
    cL95_dic = {
        'fun': constraintL95,
        'type': 'ineq',
    }
    cU95_dic = {
        'fun': constraintU95,
        'type': 'ineq',
    }
    cL96_dic = {
        'fun': constraintL96,
        'type': 'ineq',
    }
    cU96_dic = {
        'fun': constraintU96,
        'type': 'ineq',
    }
    cL97_dic = {
        'fun': constraintL97,
        'type': 'ineq',
    }
    cU97_dic = {
        'fun': constraintU97,
        'type': 'ineq',
    }
    cL98_dic = {
        'fun': constraintL98,
        'type': 'ineq',
    }
    cU98_dic = {
        'fun': constraintU98,
        'type': 'ineq',
    }
    cL99_dic = {
        'fun': constraintL99,
        'type': 'ineq',
    }
    cU99_dic = {
        'fun': constraintU99,
        'type': 'ineq',
    }
    cL100_dic = {
        'fun': constraintL100,
        'type': 'ineq',
    }
    cU100_dic = {
        'fun': constraintU100,
        'type': 'ineq',
    }


    constraints = [c0_dic, cL1_dic, cU1_dic, cL2_dic, cU2_dic, cL3_dic, cU3_dic, cL4_dic, cU4_dic, cL5_dic, cU5_dic,
                   cL6_dic, cU6_dic, cL7_dic, cU7_dic, cL8_dic, cU8_dic, cL9_dic, cU9_dic, cL10_dic, cU10_dic, cL11_dic,
                   cU11_dic, cL12_dic, cU12_dic, cL13_dic, cU13_dic, cL14_dic, cU14_dic, cL15_dic, cU15_dic, cL16_dic,
                   cU16_dic, cL17_dic, cU17_dic, cL18_dic, cU18_dic, cL19_dic, cU19_dic, cL20_dic, cU20_dic, cL21_dic,
                   cU21_dic, cL22_dic, cU22_dic, cL23_dic, cU23_dic, cL24_dic, cU24_dic, cL25_dic, cU25_dic, cL26_dic,
                   cU26_dic, cL27_dic, cU27_dic, cL28_dic, cU28_dic, cL29_dic, cU29_dic, cL30_dic, cU30_dic, cL31_dic,
                   cU31_dic, cL32_dic, cU32_dic]
    constraints2 = [cL33_dic, cU33_dic, cL34_dic, cU34_dic, cL35_dic, cU35_dic, cL36_dic, cU36_dic, cL37_dic, cU37_dic,
                    cL38_dic, cU38_dic, cL39_dic, cU39_dic, cL40_dic, cU40_dic, cL41_dic, cU41_dic, cL42_dic, cU42_dic,
                    cL43_dic, cU43_dic, cL44_dic, cU44_dic, cL45_dic, cU45_dic, cL46_dic, cU46_dic, cL47_dic, cU47_dic,
                    cL48_dic, cU48_dic, cL49_dic, cU49_dic, cL50_dic, cU50_dic, cL51_dic, cU51_dic, cL52_dic, cU52_dic,
                    cL53_dic, cU53_dic, cL54_dic, cU54_dic, cL55_dic, cU55_dic, cL56_dic, cU56_dic, cL57_dic, cU57_dic,
                    cL58_dic, cU58_dic, cL59_dic, cU59_dic, cL60_dic, cU60_dic, cL61_dic, cU61_dic, cL62_dic, cU62_dic,
                    cL63_dic, cU63_dic, cL64_dic, cU64_dic]
    constraints3 = [cL65_dic, cU65_dic, cL66_dic, cU66_dic, cL67_dic, cU67_dic, cL68_dic, cU68_dic, cL69_dic, cU69_dic,
                    cL70_dic, cU70_dic, cL71_dic, cU71_dic, cL72_dic, cU72_dic, cL73_dic, cU73_dic, cL74_dic, cU74_dic,
                    cL75_dic, cU75_dic, cL76_dic, cU76_dic, cL77_dic, cU77_dic, cL78_dic, cU78_dic, cL79_dic, cU79_dic,
                    cL80_dic, cU80_dic, cL81_dic, cU81_dic, cL82_dic, cU82_dic, cL83_dic, cU83_dic, cL84_dic, cU84_dic,
                    cL85_dic, cU85_dic, cL86_dic, cU86_dic, cL87_dic, cU87_dic, cL88_dic, cU88_dic, cL89_dic, cU89_dic,
                    cL90_dic, cU90_dic, cL91_dic, cU91_dic, cL92_dic, cU92_dic, cL93_dic, cU93_dic, cL94_dic, cU94_dic,
                    cL95_dic, cU95_dic, cL96_dic, cU96_dic,cL97_dic, cU97_dic,cL98_dic, cU98_dic, cL99_dic, cU99_dic,cL100_dic, cU100_dic]
    constraints.extend(constraints2)
    constraints.extend(constraints3)
    return constraints[:2 * N - 1]

def constraint0(alpha):
    return [alpha[0] - 0.0001]

def constraintU1(alpha):
    return [K[1] * alpha[0] - alpha[1]]
def constraintL1(alpha):
    return [alpha[1] + K[1] * alpha[0]]

def constraintU2(alpha):
    return [K[2] * alpha[0] - alpha[2]]
def constraintL2(alpha):
    return [alpha[2] + K[2] * alpha[0]]

def constraintU3(alpha):
    return [K[3] * alpha[0] - alpha[3]]
def constraintL3(alpha):
    return [alpha[3] + K[3] * alpha[0]]

def constraintU4(alpha):
    return [K[4] * alpha[0] - alpha[4]]
def constraintL4(alpha):
    return [alpha[4] + K[4] * alpha[0]]

def constraintU5(alpha):
    return [K[5] * alpha[0] - alpha[5]]
def constraintL5(alpha):
    return [alpha[5] + K[5] * alpha[0]]

def constraintU6(alpha):
    return [K[6] * alpha[0] - alpha[6]]
def constraintL6(alpha):
    return [alpha[6] + K[6] * alpha[0]]

def constraintU7(alpha):
    return [K[7] * alpha[0] - alpha[7]]
def constraintL7(alpha):
    return [alpha[7] + K[7] * alpha[0]]

def constraintU8(alpha):
    return [K[8] * alpha[0] - alpha[8]]
def constraintL8(alpha):
    return [alpha[8] + K[8] * alpha[0]]

def constraintU9(alpha):
    return [K[9] * alpha[0] - alpha[9]]
def constraintL9(alpha):
    return [alpha[9] + K[9] * alpha[0]]

def constraintU10(alpha):
    return [K[10] * alpha[0] - alpha[10]]
def constraintL10(alpha):
    return [alpha[10] + K[10] * alpha[0]]

def constraintU11(alpha):
    return [K[11] * alpha[0] - alpha[11]]
def constraintL11(alpha):
    return [alpha[11] + K[11] * alpha[0]]

def constraintU12(alpha):
    return [K[12] * alpha[0] - alpha[12]]
def constraintL12(alpha):
    return [alpha[12] + K[12] * alpha[0]]

def constraintU13(alpha):
    return [K[13] * alpha[0] - alpha[13]]
def constraintL13(alpha):
    return [alpha[13] + K[13] * alpha[0]]

def constraintU14(alpha):
    return [K[14] * alpha[0] - alpha[14]]
def constraintL14(alpha):
    return [alpha[14] + K[14] * alpha[0]]

def constraintU15(alpha):
    return [K[15] * alpha[0] - alpha[15]]
def constraintL15(alpha):
    return [alpha[15] + K[15] * alpha[0]]

def constraintU16(alpha):
    return [K[16] * alpha[0] - alpha[16]]
def constraintL16(alpha):
    return [alpha[16] + K[16] * alpha[0]]

def constraintU17(alpha):
    return [K[17] * alpha[0] - alpha[17]]
def constraintL17(alpha):
    return [alpha[17] + K[17] * alpha[0]]

def constraintU18(alpha):
    return [K[18] * alpha[0] - alpha[18]]
def constraintL18(alpha):
    return [alpha[18] + K[18] * alpha[0]]

def constraintU19(alpha):
    return [K[19]* alpha[0] - alpha[19]]
def constraintL19(alpha):
    return [alpha[19] + K[19] * alpha[0]]

def constraintU20(alpha):
    return [K[20] * alpha[0] - alpha[20]]
def constraintL20(alpha):
    return [alpha[20] + K[20] * alpha[0]]

def constraintU21(alpha):
    return [K[21] * alpha[0] - alpha[21]]
def constraintL21(alpha):
    return [alpha[21] + K[21] * alpha[0]]

def constraintU22(alpha):
    return [K[22] * alpha[0] - alpha[22]]
def constraintL22(alpha):
    return [alpha[22] + K[22] * alpha[0]]

def constraintU23(alpha):
    return [K[23] * alpha[0] - alpha[23]]
def constraintL23(alpha):
    return [alpha[23] + K[23] * alpha[0]]

def constraintU24(alpha):
    return [K[24] * alpha[0] - alpha[24]]
def constraintL24(alpha):
    return [alpha[24] + K[24] * alpha[0]]

def constraintU25(alpha):
    return [K[25] * alpha[0] - alpha[25]]
def constraintL25(alpha):
    return [alpha[25] + K[25] * alpha[0]]

def constraintU26(alpha):
    return [K[26] * alpha[0] - alpha[26]]
def constraintL26(alpha):
    return [alpha[26] + K[26] * alpha[0]]

def constraintU27(alpha):
    return [K[27] * alpha[0] - alpha[27]]
def constraintL27(alpha):
    return [alpha[27] + K[27] * alpha[0]]

def constraintU28(alpha):
    return [K[28] * alpha[0] - alpha[28]]
def constraintL28(alpha):
    return [alpha[28] + K[28] * alpha[0]]

def constraintU29(alpha):
    return [K[29] * alpha[0] - alpha[29]]
def constraintL29(alpha):
    return [alpha[29] + K[29] * alpha[0]]

def constraintU30(alpha):
    return [K[30] * alpha[0] - alpha[30]]
def constraintL30(alpha):
    return [alpha[30] + K[30] * alpha[0]]

def constraintU31(alpha):
    return [K[31] * alpha[0] - alpha[31]]
def constraintL31(alpha):
    return [alpha[31] + K[31] * alpha[0]]

def constraintU32(alpha):
    return [K[32] * alpha[0] - alpha[32]]
def constraintL32(alpha):
    return [alpha[32] + K[32] * alpha[0]]

def constraintU33(alpha):
    return [K[33] * alpha[0] - alpha[33]]
def constraintL33(alpha):
    return [alpha[33] + K[33] * alpha[0]]

def constraintU34(alpha):
    return [K[34] * alpha[0] - alpha[34]]
def constraintL34(alpha):
    return [alpha[34] + K[34] * alpha[0]]

def constraintU35(alpha):
    return [K[35] * alpha[0] - alpha[35]]
def constraintL35(alpha):
    return [alpha[35] + K[35] * alpha[0]]

def constraintU36(alpha):
    return [K[36] * alpha[0] - alpha[36]]
def constraintL36(alpha):
    return [alpha[36] + K[36] * alpha[0]]

def constraintU37(alpha):
    return [K[37] * alpha[0] - alpha[37]]
def constraintL37(alpha):
    return [alpha[37] + K[37] * alpha[0]]

def constraintU38(alpha):
    return [K[38] * alpha[0] - alpha[38]]
def constraintL38(alpha):
    return [alpha[38] + K[2] * alpha[0]]

def constraintU39(alpha):
    return [K[39] * alpha[0] - alpha[39]]
def constraintL39(alpha):
    return [alpha[39] + K[39] * alpha[0]]

def constraintU40(alpha):
    return [K[40] * alpha[0] - alpha[40]]
def constraintL40(alpha):
    return [alpha[40] + K[40] * alpha[0]]

def constraintU41(alpha):
    return [K[41] * alpha[0] - alpha[41]]
def constraintL41(alpha):
    return [alpha[41] + K[41] * alpha[0]]

def constraintU42(alpha):
    return [K[42] * alpha[0] - alpha[42]]
def constraintL42(alpha):
    return [alpha[42] + K[42] * alpha[0]]

def constraintU43(alpha):
    return [K[43] * alpha[0] - alpha[43]]
def constraintL43(alpha):
    return [alpha[43] + K[43] * alpha[0]]

def constraintU44(alpha):
    return [K[44] * alpha[0] - alpha[44]]
def constraintL44(alpha):
    return [alpha[44] + K[44] * alpha[0]]

def constraintU45(alpha):
    return [K[45] * alpha[0] - alpha[45]]
def constraintL45(alpha):
    return [alpha[45] + K[45] * alpha[0]]

def constraintU46(alpha):
    return [K[46] * alpha[0] - alpha[46]]
def constraintL46(alpha):
    return [alpha[46] + K[46] * alpha[0]]

def constraintU47(alpha):
    return [K[47] * alpha[0] - alpha[47]]
def constraintL47(alpha):
    return [alpha[47] + K[47] * alpha[0]]

def constraintU48(alpha):
    return [K[48] * alpha[0] - alpha[48]]
def constraintL48(alpha):
    return [alpha[48] + K[48] * alpha[0]]

def constraintU49(alpha):
    return [K[49] * alpha[0] - alpha[49]]
def constraintL49(alpha):
    return [alpha[49] + K[49] * alpha[0]]

def constraintU50(alpha):
    return [K[50] * alpha[0] - alpha[50]]
def constraintL50(alpha):
    return [alpha[50] + K[50] * alpha[0]]

def constraintU51(alpha):
    return [K[51] * alpha[0] - alpha[51]]
def constraintL51(alpha):
    return [alpha[51] + K[51] * alpha[0]]

def constraintU52(alpha):
    return [K[52] * alpha[0] - alpha[52]]
def constraintL52(alpha):
    return [alpha[52] + K[52] * alpha[0]]

def constraintU53(alpha):
    return [K[53] * alpha[0] - alpha[53]]
def constraintL53(alpha):
    return [alpha[53] + K[53] * alpha[0]]

def constraintU54(alpha):
    return [K[54] * alpha[0] - alpha[54]]
def constraintL54(alpha):
    return [alpha[54] + K[54] * alpha[0]]

def constraintU55(alpha):
    return [K[55] * alpha[0] - alpha[55]]
def constraintL55(alpha):
    return [alpha[55] + K[55] * alpha[0]]

def constraintU56(alpha):
    return [K[56] * alpha[0] - alpha[56]]
def constraintL56(alpha):
    return [alpha[56] + K[56] * alpha[0]]

def constraintU57(alpha):
    return [K[57] * alpha[0] - alpha[57]]
def constraintL57(alpha):
    return [alpha[57] + K[57] * alpha[0]]

def constraintU58(alpha):
    return [K[58] * alpha[0] - alpha[58]]
def constraintL58(alpha):
    return [alpha[58] + K[58] * alpha[0]]

def constraintU59(alpha):
    return [K[59] * alpha[0] - alpha[59]]
def constraintL59(alpha):
    return [alpha[59] + K[59] * alpha[0]]

def constraintU60(alpha):
    return [K[60] * alpha[0] - alpha[60]]
def constraintL60(alpha):
    return [alpha[60] + K[60] * alpha[0]]

def constraintU61(alpha):
    return [K[61] * alpha[0] - alpha[61]]
def constraintL61(alpha):
    return [alpha[61] + K[61] * alpha[0]]

def constraintU62(alpha):
    return [K[62] * alpha[0] - alpha[62]]
def constraintL62(alpha):
    return [alpha[62] + K[62] * alpha[0]]

def constraintU63(alpha):
    return [K[63] * alpha[0] - alpha[63]]
def constraintL63(alpha):
    return [alpha[63] + K[63] * alpha[0]]

def constraintU64(alpha):
    return [K[64] * alpha[0] - alpha[64]]
def constraintL64(alpha):
    return [alpha[64] + K[64] * alpha[0]]

def constraintU65(alpha):
    return [K[65] * alpha[0] - alpha[65]]
def constraintL65(alpha):
    return [alpha[65] + K[65] * alpha[0]]

def constraintU66(alpha):
    return [K[66] * alpha[0] - alpha[66]]
def constraintL66(alpha):
    return [alpha[66] + K[66] * alpha[0]]

def constraintU67(alpha):
    return [K[67] * alpha[0] - alpha[67]]
def constraintL67(alpha):
    return [alpha[67] + K[67] * alpha[0]]

def constraintU68(alpha):
    return [K[68] * alpha[0] - alpha[68]]
def constraintL68(alpha):
    return [alpha[68] + K[68] * alpha[0]]

def constraintU69(alpha):
    return [K[69] * alpha[0] - alpha[69]]
def constraintL69(alpha):
    return [alpha[69] + K[69] * alpha[0]]

def constraintU70(alpha):
    return [K[70] * alpha[0] - alpha[70]]
def constraintL70(alpha):
    return [alpha[70] + K[70] * alpha[0]]

def constraintU71(alpha):
    return [K[71] * alpha[0] - alpha[71]]
def constraintL71(alpha):
    return [alpha[71] + K[71] * alpha[0]]

def constraintU72(alpha):
    return [K[72] * alpha[0] - alpha[72]]
def constraintL72(alpha):
    return [alpha[72] + K[72] * alpha[0]]

def constraintU73(alpha):
    return [K[73] * alpha[0] - alpha[73]]
def constraintL73(alpha):
    return [alpha[73] + K[73] * alpha[0]]

def constraintU74(alpha):
    return [K[74] * alpha[0] - alpha[74]]
def constraintL74(alpha):
    return [alpha[74] + K[74] * alpha[0]]

def constraintU75(alpha):
    return [K[75] * alpha[0] - alpha[75]]
def constraintL75(alpha):
    return [alpha[75] + K[75] * alpha[0]]

def constraintU76(alpha):
    return [K[76] * alpha[0] - alpha[76]]
def constraintL76(alpha):
    return [alpha[76] + K[76] * alpha[0]]

def constraintU77(alpha):
    return [K[77] * alpha[0] - alpha[77]]
def constraintL77(alpha):
    return [alpha[77] + K[77] * alpha[0]]

def constraintU78(alpha):
    return [K[78] * alpha[0] - alpha[78]]
def constraintL78(alpha):
    return [alpha[78] + K[78] * alpha[0]]

def constraintU79(alpha):
    return [K[79] * alpha[0] - alpha[79]]
def constraintL79(alpha):
    return [alpha[79] + K[79] * alpha[0]]

def constraintU80(alpha):
    return [K[80] * alpha[0] - alpha[80]]
def constraintL80(alpha):
    return [alpha[80] + K[80] * alpha[0]]

def constraintU81(alpha):
    return [K[81] * alpha[0] - alpha[81]]
def constraintL81(alpha):
    return [alpha[81] + K[81] * alpha[0]]

def constraintU82(alpha):
    return [K[82] * alpha[0] - alpha[82]]
def constraintL82(alpha):
    return [alpha[82] + K[82] * alpha[0]]

def constraintU83(alpha):
    return [K[83] * alpha[0] - alpha[83]]
def constraintL83(alpha):
    return [alpha[83] + K[83] * alpha[0]]

def constraintU84(alpha):
    return [K[84] * alpha[0] - alpha[84]]
def constraintL84(alpha):
    return [alpha[84] + K[84] * alpha[0]]

def constraintU85(alpha):
    return [K[85] * alpha[0] - alpha[85]]
def constraintL85(alpha):
    return [alpha[85] + K[85] * alpha[0]]

def constraintU86(alpha):
    return [K[86] * alpha[0] - alpha[86]]
def constraintL86(alpha):
    return [alpha[86] + K[86] * alpha[0]]

def constraintU87(alpha):
    return [K[87] * alpha[0] - alpha[87]]
def constraintL87(alpha):
    return [alpha[87] + K[87] * alpha[0]]

def constraintU88(alpha):
    return [K[88] * alpha[0] - alpha[88]]
def constraintL88(alpha):
    return [alpha[88] + K[88] * alpha[0]]

def constraintU89(alpha):
    return [K[89] * alpha[0] - alpha[89]]
def constraintL89(alpha):
    return [alpha[89] + K[89] * alpha[0]]

def constraintU90(alpha):
    return [K[90] * alpha[0] - alpha[90]]
def constraintL90(alpha):
    return [alpha[90] + K[90] * alpha[0]]

def constraintU91(alpha):
    return [K[91] * alpha[0] - alpha[91]]
def constraintL91(alpha):
    return [alpha[91] + K[91] * alpha[0]]

def constraintU92(alpha):
    return [K[92] * alpha[0] - alpha[92]]
def constraintL92(alpha):
    return [alpha[92] + K[92] * alpha[0]]

def constraintU93(alpha):
    return [K[93] * alpha[0] - alpha[93]]
def constraintL93(alpha):
    return [alpha[93] + K[93] * alpha[0]]

def constraintU94(alpha):
    return [K[94] * alpha[0] - alpha[94]]
def constraintL94(alpha):
    return [alpha[94] + K[94] * alpha[0]]

def constraintU95(alpha):
    return [K[95] * alpha[0] - alpha[95]]
def constraintL95(alpha):
    return [alpha[95] + K[95] * alpha[0]]

def constraintU96(alpha):
    return [K[96] * alpha[0] - alpha[96]]
def constraintL96(alpha):
    return [alpha[96] + K[96] * alpha[0]]

def constraintU97(alpha):
    return [K[97] * alpha[0] - alpha[97]]
def constraintL97(alpha):
    return [alpha[97] + K[97] * alpha[0]]

def constraintU98(alpha):
    return [K[98] * alpha[0] - alpha[98]]
def constraintL98(alpha):
    return [alpha[98] + K[98] * alpha[0]]

def constraintU99(alpha):
    return [K[99] * alpha[0] - alpha[99]]
def constraintL99(alpha):
    return [alpha[99] + K[99] * alpha[0]]

def constraintU100(alpha):
    return [K[100] * alpha[0] - alpha[100]]
def constraintL100(alpha):
    return [alpha[100] + K[100] * alpha[0]]
###########################

def generating_Gamma(alpha):
    alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
    values = np.concatenate((alpha, np.flip(np.array(alpha[1:]))))
    i, j = np.ones((N, N)).nonzero()
    values = values[j - i].reshape(N, N)
    B = values * B_mask
    values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
    i, j = np.ones((N, N)).nonzero()
    values_prime2 = values_prime[j - i].reshape(N, N)
    C = np.conj(values_prime2 * C_mask)
    alpha_0 = B[0, 0]
    Gamma = 1 / alpha_0[None, None] * (np.matmul(B, np.conj(B).T) - np.matmul(C, np.conj(C).T))
    return Gamma

def f(alpha):
    Gamma = generating_Gamma(alpha)
    #print('...')
    #print(Gamma)
    return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ sCov)

constraints = generating_constraints(N)

# MODEL
#AUTOREGRESSIVES MODEL GAUS
r = 0.7
C = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        C[i,j] = r**(np.abs(j-i))

#Brownian Motion (see shrinkage estimator original paper)
#H = 0.7
#C = np.zeros((N,N))
#for i in range(N):
#    for j in range(N):
#        C[i,j] = 0.5 * ( (np.abs(j-i) + 1)**(2*H) - 2 * np.abs(j-i)**(2*H) + np.abs((np.abs(j-i) - 1))**(2*H) )




N_SAMPLES = [4,8,10,16,32,64,100]
RUNS = 1000

MSE_sCov_n = []
MSE_toeplitz_n = []
MSE_OAS_n = []
MSE_toeplitz2_n = []

for n_samples in N_SAMPLES:

    MSE_sCov = []
    MSE_toeplitz = []
    MSE_OAS = []
    MSE_toeplitz2 = []
    for run in range(RUNS):
        K = K_dic[str(N)] * np.ones(N)
        if run%50 == 0:
            print(f'run {run}')
        samples = np.random.multivariate_normal(np.zeros(N),C,n_samples)


        #first comparison - sample Cov
        sCov = 1/n_samples * (samples.T @ samples)

        #second comparison - oracle approximating Shrinkage Estimator
        F = np.trace(sCov)/N * np.eye(N)
        rho = min(((1 - 2/N) * np.trace(sCov @ sCov) + np.trace(sCov)**2)/((n_samples + 1 - 2/N) * (np.trace(sCov @ sCov) - np.trace(sCov)**2/N)),1)
        OAS_C = (1 - rho) * sCov + rho * F

        # my method
        init_values = np.zeros(N)
        init_values[0] = np.random.uniform(low=1, high=20)
        for n in range(1,N):
            init_values[n] = np.random.uniform(low = - K[n] * init_values[0] + 0.0001, high = K[n] * init_values[0] - 0.0001)
        result = optimize.minimize(f, init_values, method="SLSQP",constraints=constraints)
        Gamma_est = generating_Gamma(result.x)

        MSE_sCov.append(np.sum((sCov - C)**2))
        MSE_toeplitz.append(np.sum((np.linalg.inv(Gamma_est) - C)**2))
        if np.sum((np.linalg.inv(Gamma_est) - C)**2) > 1000:
            w,v = np.linalg.eigh(Gamma_est)
            print(w)
            print(run)
            print(np.linalg.det(np.linalg.inv(Gamma_est)))
            print(np.linalg.det(Gamma_est))
            print(result.x)
            raise ValueError
        MSE_OAS.append(np.sum((OAS_C - C)**2))
        #print(f'MSE  Toeplitz: {np.sum((np.linalg.inv(Gamma_est) - C)**2)}')


        if IMPROVING:
            #print('start')
            K = adjustingK(K,f, result)
            alpha_0 = result.x[0]
            idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
            idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
            len_bounding = len(idx_bounding)
            len_interior = len(idx_interior)
            derivatives = optimize.approx_fprime(result.x, f, epsilon=10e-8)
            counter = 0
            while (len_interior > 0) & (len_bounding > 0):
                counter += 1
                if counter == 200:
                    break
                #print(result.x[0])
                result2 = optimize.minimize(f, result.x, method="SLSQP", constraints=constraints)
                #print('RESULTS')
                #print(result.fun)
                #print(result2.fun)
                result = result2
                K = adjustingK(K,f, result)
                idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
                idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
                len_bounding = len(idx_bounding)
                len_interior = len(idx_interior)
                derivatives = optimize.approx_fprime(result.x, f, epsilon=10e-8)
            Gamma_est2 = generating_Gamma(result.x)
            MSE_toeplitz2.append(np.sum((np.linalg.inv(Gamma_est2) - C) ** 2))
    print(f'MSE of sCov and real Cov: {np.mean(MSE_sCov):.4f}')
    print(f'MSE of Toep and real Cov: {np.mean(MSE_toeplitz):.4f}')
    print(f'MSE of Toep2 and real Cov: {np.mean(MSE_toeplitz2):.4f}')
    print(f'MSE of OAS and real Cov: {np.mean(MSE_OAS):.4f}')


    MSE_sCov_n.append(np.mean(MSE_sCov))
    MSE_toeplitz_n.append(np.mean(MSE_toeplitz))
    MSE_OAS_n.append(np.mean(MSE_OAS))
    MSE_toeplitz2_n.append(np.mean(MSE_toeplitz2))

MSE_sCov_n = np.array(MSE_sCov_n)
MSE_toeplitz_n = np.array(MSE_toeplitz_n)
MSE_OAS_n = np.array(MSE_OAS_n)
MSE_toeplitz2_n = np.array(MSE_toeplitz2_n)

csv_writer.writerow(N_SAMPLES)
csv_writer.writerow(MSE_sCov_n)
csv_writer.writerow(MSE_OAS_n)
csv_writer.writerow(MSE_toeplitz_n)
csv_writer.writerow(MSE_toeplitz2_n)
csv_file.close()

plt.plot(N_SAMPLES,MSE_sCov_n,label = 'sCov')
plt.plot(N_SAMPLES,MSE_toeplitz_n,label = 'Toeplitz')
plt.plot(N_SAMPLES,MSE_OAS_n,label = 'OAS')
plt.plot(N_SAMPLES,MSE_toeplitz2_n,label = 'Toeplitz2')
plt.legend()
plt.ylabel('MSE')
plt.xlabel('N_SAMPLES')
#plt.title(f'Dimension: {N}, r-value: {r}')
plt.show()