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
    '128':0.007624,
    '256':0.0037975,
}

IMPROVING = True
N = 256
K = K_dic[str(N)] * np.ones(N)
csv_file = open('./Autoregressive_DIM256_RUNS15_r0_4.txt','w')
#csv_file = open('./test.txt','w')
csv_writer = csv.writer(csv_file)

rand_matrix = np.random.randn(N, N)
B_mask = np.tril(rand_matrix)
B_mask[B_mask != 0] = 1
C_mask = np.tril(rand_matrix, k=-1)
C_mask[C_mask != 0] = 1

###########################
def generating_constraints_eig(N):
    c0_eig_dic = {
        'fun': constraintEig0,
        'type': 'ineq'
    }
    c1_eig_dic = {
        'fun': constraintEig1,
        'type': 'ineq'
    }
    c2_eig_dic = {
        'fun': constraintEig2,
        'type': 'ineq'
    }
    c3_eig_dic = {
        'fun': constraintEig3,
        'type': 'ineq'
    }
    c4_eig_dic = {
        'fun': constraintEig4,
        'type': 'ineq'
    }
    c5_eig_dic = {
        'fun': constraintEig5,
        'type': 'ineq'
    }
    c6_eig_dic = {
        'fun': constraintEig6,
        'type': 'ineq'
    }
    c7_eig_dic = {
        'fun': constraintEig7,
        'type': 'ineq'
    }
    c8_eig_dic = {
        'fun': constraintEig8,
        'type': 'ineq'
    }
    c9_eig_dic = {
        'fun': constraintEig9,
        'type': 'ineq'
    }
    c10_eig_dic = {
        'fun': constraintEig10,
        'type': 'ineq'
    }
    c11_eig_dic = {
        'fun': constraintEig11,
        'type': 'ineq'
    }
    c12_eig_dic = {
        'fun': constraintEig12,
        'type': 'ineq'
    }
    c13_eig_dic = {
        'fun': constraintEig13,
        'type': 'ineq'
    }
    c14_eig_dic = {
        'fun': constraintEig14,
        'type': 'ineq'
    }
    c15_eig_dic = {
        'fun': constraintEig15,
        'type': 'ineq'
    }
    c16_eig_dic = {
        'fun': constraintEig16,
        'type': 'ineq'
    }
    c17_eig_dic = {
        'fun': constraintEig17,
        'type': 'ineq'
    }
    c18_eig_dic = {
        'fun': constraintEig18,
        'type': 'ineq'
    }
    c19_eig_dic = {
        'fun': constraintEig19,
        'type': 'ineq'
    }
    c20_eig_dic = {
        'fun': constraintEig20,
        'type': 'ineq'
    }
    c21_eig_dic = {
        'fun': constraintEig21,
        'type': 'ineq'
    }
    c22_eig_dic = {
        'fun': constraintEig22,
        'type': 'ineq'
    }
    c23_eig_dic = {
        'fun': constraintEig23,
        'type': 'ineq'
    }
    c24_eig_dic = {
        'fun': constraintEig24,
        'type': 'ineq'
    }
    c25_eig_dic = {
        'fun': constraintEig25,
        'type': 'ineq'
    }
    c26_eig_dic = {
        'fun': constraintEig26,
        'type': 'ineq'
    }
    c27_eig_dic = {
        'fun': constraintEig27,
        'type': 'ineq'
    }
    c28_eig_dic = {
        'fun': constraintEig28,
        'type': 'ineq'
    }
    c29_eig_dic = {
        'fun': constraintEig29,
        'type': 'ineq'
    }
    c30_eig_dic = {
        'fun': constraintEig30,
        'type': 'ineq'
    }
    c31_eig_dic = {
        'fun': constraintEig31,
        'type': 'ineq'
    }
    c32_eig_dic = {
        'fun': constraintEig32,
        'type': 'ineq'
    }
    c33_eig_dic = {
        'fun': constraintEig33,
        'type': 'ineq'
    }
    c34_eig_dic = {
        'fun': constraintEig34,
        'type': 'ineq'
    }
    c35_eig_dic = {
        'fun': constraintEig35,
        'type': 'ineq'
    }
    c36_eig_dic = {
        'fun': constraintEig36,
        'type': 'ineq'
    }
    c37_eig_dic = {
        'fun': constraintEig37,
        'type': 'ineq'
    }
    c38_eig_dic = {
        'fun': constraintEig38,
        'type': 'ineq'
    }
    c39_eig_dic = {
        'fun': constraintEig39,
        'type': 'ineq'
    }
    c40_eig_dic = {
        'fun': constraintEig40,
        'type': 'ineq'
    }
    c41_eig_dic = {
        'fun': constraintEig41,
        'type': 'ineq'
    }
    c42_eig_dic = {
        'fun': constraintEig42,
        'type': 'ineq'
    }
    c43_eig_dic = {
        'fun': constraintEig43,
        'type': 'ineq'
    }
    c44_eig_dic = {
        'fun': constraintEig44,
        'type': 'ineq'
    }
    c45_eig_dic = {
        'fun': constraintEig45,
        'type': 'ineq'
    }
    c46_eig_dic = {
        'fun': constraintEig46,
        'type': 'ineq'
    }
    c47_eig_dic = {
        'fun': constraintEig47,
        'type': 'ineq'
    }
    c48_eig_dic = {
        'fun': constraintEig48,
        'type': 'ineq'
    }
    c49_eig_dic = {
        'fun': constraintEig49,
        'type': 'ineq'
    }
    c50_eig_dic = {
        'fun': constraintEig50,
        'type': 'ineq'
    }
    c51_eig_dic = {
        'fun': constraintEig51,
        'type': 'ineq'
    }
    c52_eig_dic = {
        'fun': constraintEig52,
        'type': 'ineq'
    }
    c53_eig_dic = {
        'fun': constraintEig53,
        'type': 'ineq'
    }
    c54_eig_dic = {
        'fun': constraintEig54,
        'type': 'ineq'
    }
    c55_eig_dic = {
        'fun': constraintEig55,
        'type': 'ineq'
    }
    c56_eig_dic = {
        'fun': constraintEig56,
        'type': 'ineq'
    }
    c57_eig_dic = {
        'fun': constraintEig57,
        'type': 'ineq'
    }
    c58_eig_dic = {
        'fun': constraintEig58,
        'type': 'ineq'
    }
    c59_eig_dic = {
        'fun': constraintEig59,
        'type': 'ineq'
    }
    c60_eig_dic = {
        'fun': constraintEig60,
        'type': 'ineq'
    }
    c61_eig_dic = {
        'fun': constraintEig61,
        'type': 'ineq'
    }
    c62_eig_dic = {
        'fun': constraintEig62,
        'type': 'ineq'
    }
    c63_eig_dic = {
        'fun': constraintEig63,
        'type': 'ineq'
    }
    c64_eig_dic = {
        'fun': constraintEig64,
        'type': 'ineq'
    }
    c65_eig_dic = {
        'fun': constraintEig65,
        'type': 'ineq'
    }
    c66_eig_dic = {
        'fun': constraintEig66,
        'type': 'ineq'
    }
    c67_eig_dic = {
        'fun': constraintEig67,
        'type': 'ineq'
    }
    c68_eig_dic = {
        'fun': constraintEig68,
        'type': 'ineq'
    }
    c69_eig_dic = {
        'fun': constraintEig69,
        'type': 'ineq'
    }
    c70_eig_dic = {
        'fun': constraintEig70,
        'type': 'ineq'
    }
    c71_eig_dic = {
        'fun': constraintEig71,
        'type': 'ineq'
    }
    c72_eig_dic = {
        'fun': constraintEig72,
        'type': 'ineq'
    }
    c73_eig_dic = {
        'fun': constraintEig73,
        'type': 'ineq'
    }
    c74_eig_dic = {
        'fun': constraintEig74,
        'type': 'ineq'
    }
    c75_eig_dic = {
        'fun': constraintEig75,
        'type': 'ineq'
    }
    c76_eig_dic = {
        'fun': constraintEig76,
        'type': 'ineq'
    }
    c77_eig_dic = {
        'fun': constraintEig77,
        'type': 'ineq'
    }
    c78_eig_dic = {
        'fun': constraintEig78,
        'type': 'ineq'
    }
    c79_eig_dic = {
        'fun': constraintEig79,
        'type': 'ineq'
    }
    c80_eig_dic = {
        'fun': constraintEig80,
        'type': 'ineq'
    }
    c81_eig_dic = {
        'fun': constraintEig81,
        'type': 'ineq'
    }
    c82_eig_dic = {
        'fun': constraintEig82,
        'type': 'ineq'
    }
    c83_eig_dic = {
        'fun': constraintEig83,
        'type': 'ineq'
    }
    c84_eig_dic = {
        'fun': constraintEig84,
        'type': 'ineq'
    }
    c85_eig_dic = {
        'fun': constraintEig85,
        'type': 'ineq'
    }
    c86_eig_dic = {
        'fun': constraintEig86,
        'type': 'ineq'
    }
    c87_eig_dic = {
        'fun': constraintEig87,
        'type': 'ineq'
    }
    c88_eig_dic = {
        'fun': constraintEig88,
        'type': 'ineq'
    }
    c89_eig_dic = {
        'fun': constraintEig89,
        'type': 'ineq'
    }
    c90_eig_dic = {
        'fun': constraintEig90,
        'type': 'ineq'
    }
    c91_eig_dic = {
        'fun': constraintEig91,
        'type': 'ineq'
    }
    c92_eig_dic = {
        'fun': constraintEig92,
        'type': 'ineq'
    }
    c93_eig_dic = {
        'fun': constraintEig93,
        'type': 'ineq'
    }
    c94_eig_dic = {
        'fun': constraintEig94,
        'type': 'ineq'
    }
    c95_eig_dic = {
        'fun': constraintEig95,
        'type': 'ineq'
    }
    c96_eig_dic = {
        'fun': constraintEig96,
        'type': 'ineq'
    }
    c97_eig_dic = {
        'fun': constraintEig97,
        'type': 'ineq'
    }
    c98_eig_dic = {
        'fun': constraintEig98,
        'type': 'ineq'
    }
    c99_eig_dic = {
        'fun': constraintEig99,
        'type': 'ineq'
    }
    c100_eig_dic = {
        'fun': constraintEig100,
        'type': 'ineq'
    }
    c101_eig_dic = {
        'fun': constraintEig101,
        'type': 'ineq'
    }
    c102_eig_dic = {
        'fun': constraintEig102,
        'type': 'ineq'
    }
    c103_eig_dic = {
        'fun': constraintEig103,
        'type': 'ineq'
    }
    c104_eig_dic = {
        'fun': constraintEig104,
        'type': 'ineq'
    }
    c105_eig_dic = {
        'fun': constraintEig105,
        'type': 'ineq'
    }
    c106_eig_dic = {
        'fun': constraintEig106,
        'type': 'ineq'
    }
    c107_eig_dic = {
        'fun': constraintEig107,
        'type': 'ineq'
    }
    c108_eig_dic = {
        'fun': constraintEig108,
        'type': 'ineq'
    }
    c109_eig_dic = {
        'fun': constraintEig109,
        'type': 'ineq'
    }
    c110_eig_dic = {
        'fun': constraintEig110,
        'type': 'ineq'
    }
    c111_eig_dic = {
        'fun': constraintEig111,
        'type': 'ineq'
    }
    c112_eig_dic = {
        'fun': constraintEig112,
        'type': 'ineq'
    }
    c113_eig_dic = {
        'fun': constraintEig113,
        'type': 'ineq'
    }
    c114_eig_dic = {
        'fun': constraintEig114,
        'type': 'ineq'
    }
    c115_eig_dic = {
        'fun': constraintEig115,
        'type': 'ineq'
    }
    c116_eig_dic = {
        'fun': constraintEig116,
        'type': 'ineq'
    }
    c117_eig_dic = {
        'fun': constraintEig117,
        'type': 'ineq'
    }
    c118_eig_dic = {
        'fun': constraintEig118,
        'type': 'ineq'
    }
    c119_eig_dic = {
        'fun': constraintEig119,
        'type': 'ineq'
    }
    c120_eig_dic = {
        'fun': constraintEig120,
        'type': 'ineq'
    }
    c121_eig_dic = {
        'fun': constraintEig121,
        'type': 'ineq'
    }
    c122_eig_dic = {
        'fun': constraintEig122,
        'type': 'ineq'
    }
    c123_eig_dic = {
        'fun': constraintEig123,
        'type': 'ineq'
    }
    c124_eig_dic = {
        'fun': constraintEig124,
        'type': 'ineq'
    }
    c125_eig_dic = {
        'fun': constraintEig125,
        'type': 'ineq'
    }
    c126_eig_dic = {
        'fun': constraintEig126,
        'type': 'ineq'
    }
    c127_eig_dic = {
        'fun': constraintEig127,
        'type': 'ineq'
    }
    c128_eig_dic = {
        'fun': constraintEig128,
        'type': 'ineq'
    }
    c129_eig_dic = {
        'fun': constraintEig129,
        'type': 'ineq'
    }
    c130_eig_dic = {
        'fun': constraintEig130,
        'type': 'ineq'
    }
    c131_eig_dic = {
        'fun': constraintEig131,
        'type': 'ineq'
    }
    c132_eig_dic = {
        'fun': constraintEig132,
        'type': 'ineq'
    }
    c133_eig_dic = {
        'fun': constraintEig133,
        'type': 'ineq'
    }
    c134_eig_dic = {
        'fun': constraintEig134,
        'type': 'ineq'
    }
    c135_eig_dic = {
        'fun': constraintEig135,
        'type': 'ineq'
    }
    c136_eig_dic = {
        'fun': constraintEig136,
        'type': 'ineq'
    }
    c137_eig_dic = {
        'fun': constraintEig137,
        'type': 'ineq'
    }
    c138_eig_dic = {
        'fun': constraintEig138,
        'type': 'ineq'
    }
    c139_eig_dic = {
        'fun': constraintEig139,
        'type': 'ineq'
    }
    c140_eig_dic = {
        'fun': constraintEig140,
        'type': 'ineq'
    }
    c141_eig_dic = {
        'fun': constraintEig141,
        'type': 'ineq'
    }
    c142_eig_dic = {
        'fun': constraintEig142,
        'type': 'ineq'
    }
    c143_eig_dic = {
        'fun': constraintEig143,
        'type': 'ineq'
    }
    c144_eig_dic = {
        'fun': constraintEig144,
        'type': 'ineq'
    }
    c145_eig_dic = {
        'fun': constraintEig145,
        'type': 'ineq'
    }
    c146_eig_dic = {
        'fun': constraintEig146,
        'type': 'ineq'
    }
    c147_eig_dic = {
        'fun': constraintEig147,
        'type': 'ineq'
    }
    c148_eig_dic = {
        'fun': constraintEig148,
        'type': 'ineq'
    }
    c149_eig_dic = {
        'fun': constraintEig149,
        'type': 'ineq'
    }
    c150_eig_dic = {
        'fun': constraintEig150,
        'type': 'ineq'
    }
    c151_eig_dic = {
        'fun': constraintEig151,
        'type': 'ineq'
    }
    c152_eig_dic = {
        'fun': constraintEig152,
        'type': 'ineq'
    }
    c153_eig_dic = {
        'fun': constraintEig153,
        'type': 'ineq'
    }
    c154_eig_dic = {
        'fun': constraintEig154,
        'type': 'ineq'
    }
    c155_eig_dic = {
        'fun': constraintEig155,
        'type': 'ineq'
    }
    c156_eig_dic = {
        'fun': constraintEig156,
        'type': 'ineq'
    }
    c157_eig_dic = {
        'fun': constraintEig157,
        'type': 'ineq'
    }
    c158_eig_dic = {
        'fun': constraintEig158,
        'type': 'ineq'
    }
    c159_eig_dic = {
        'fun': constraintEig159,
        'type': 'ineq'
    }
    c160_eig_dic = {
        'fun': constraintEig160,
        'type': 'ineq'
    }
    c161_eig_dic = {
        'fun': constraintEig161,
        'type': 'ineq'
    }
    c162_eig_dic = {
        'fun': constraintEig162,
        'type': 'ineq'
    }
    c163_eig_dic = {
        'fun': constraintEig163,
        'type': 'ineq'
    }
    c164_eig_dic = {
        'fun': constraintEig164,
        'type': 'ineq'
    }
    c165_eig_dic = {
        'fun': constraintEig165,
        'type': 'ineq'
    }
    c166_eig_dic = {
        'fun': constraintEig166,
        'type': 'ineq'
    }
    c167_eig_dic = {
        'fun': constraintEig167,
        'type': 'ineq'
    }
    c168_eig_dic = {
        'fun': constraintEig168,
        'type': 'ineq'
    }
    c169_eig_dic = {
        'fun': constraintEig169,
        'type': 'ineq'
    }
    c170_eig_dic = {
        'fun': constraintEig170,
        'type': 'ineq'
    }
    c171_eig_dic = {
        'fun': constraintEig171,
        'type': 'ineq'
    }
    c172_eig_dic = {
        'fun': constraintEig172,
        'type': 'ineq'
    }
    c173_eig_dic = {
        'fun': constraintEig173,
        'type': 'ineq'
    }
    c174_eig_dic = {
        'fun': constraintEig174,
        'type': 'ineq'
    }
    c175_eig_dic = {
        'fun': constraintEig175,
        'type': 'ineq'
    }
    c176_eig_dic = {
        'fun': constraintEig176,
        'type': 'ineq'
    }
    c177_eig_dic = {
        'fun': constraintEig177,
        'type': 'ineq'
    }
    c178_eig_dic = {
        'fun': constraintEig178,
        'type': 'ineq'
    }
    c179_eig_dic = {
        'fun': constraintEig179,
        'type': 'ineq'
    }
    c180_eig_dic = {
        'fun': constraintEig180,
        'type': 'ineq'
    }
    c181_eig_dic = {
        'fun': constraintEig181,
        'type': 'ineq'
    }
    c182_eig_dic = {
        'fun': constraintEig182,
        'type': 'ineq'
    }
    c183_eig_dic = {
        'fun': constraintEig183,
        'type': 'ineq'
    }
    c184_eig_dic = {
        'fun': constraintEig184,
        'type': 'ineq'
    }
    c185_eig_dic = {
        'fun': constraintEig185,
        'type': 'ineq'
    }
    c186_eig_dic = {
        'fun': constraintEig186,
        'type': 'ineq'
    }
    c187_eig_dic = {
        'fun': constraintEig187,
        'type': 'ineq'
    }
    c188_eig_dic = {
        'fun': constraintEig188,
        'type': 'ineq'
    }
    c189_eig_dic = {
        'fun': constraintEig189,
        'type': 'ineq'
    }
    c190_eig_dic = {
        'fun': constraintEig190,
        'type': 'ineq'
    }
    c191_eig_dic = {
        'fun': constraintEig191,
        'type': 'ineq'
    }
    c192_eig_dic = {
        'fun': constraintEig192,
        'type': 'ineq'
    }
    c193_eig_dic = {
        'fun': constraintEig193,
        'type': 'ineq'
    }
    c194_eig_dic = {
        'fun': constraintEig194,
        'type': 'ineq'
    }
    c195_eig_dic = {
        'fun': constraintEig195,
        'type': 'ineq'
    }
    c196_eig_dic = {
        'fun': constraintEig196,
        'type': 'ineq'
    }
    c197_eig_dic = {
        'fun': constraintEig197,
        'type': 'ineq'
    }
    c198_eig_dic = {
        'fun': constraintEig198,
        'type': 'ineq'
    }
    c199_eig_dic = {
        'fun': constraintEig199,
        'type': 'ineq'
    }
    c200_eig_dic = {
        'fun': constraintEig200,
        'type': 'ineq'
    }
    c201_eig_dic = {
        'fun': constraintEig201,
        'type': 'ineq'
    }
    c202_eig_dic = {
        'fun': constraintEig202,
        'type': 'ineq'
    }
    c203_eig_dic = {
        'fun': constraintEig203,
        'type': 'ineq'
    }
    c204_eig_dic = {
        'fun': constraintEig204,
        'type': 'ineq'
    }
    c205_eig_dic = {
        'fun': constraintEig205,
        'type': 'ineq'
    }
    c206_eig_dic = {
        'fun': constraintEig206,
        'type': 'ineq'
    }
    c207_eig_dic = {
        'fun': constraintEig207,
        'type': 'ineq'
    }
    c208_eig_dic = {
        'fun': constraintEig208,
        'type': 'ineq'
    }
    c209_eig_dic = {
        'fun': constraintEig209,
        'type': 'ineq'
    }
    c210_eig_dic = {
        'fun': constraintEig210,
        'type': 'ineq'
    }
    c211_eig_dic = {
        'fun': constraintEig211,
        'type': 'ineq'
    }
    c212_eig_dic = {
        'fun': constraintEig212,
        'type': 'ineq'
    }
    c213_eig_dic = {
        'fun': constraintEig213,
        'type': 'ineq'
    }
    c214_eig_dic = {
        'fun': constraintEig214,
        'type': 'ineq'
    }
    c215_eig_dic = {
        'fun': constraintEig215,
        'type': 'ineq'
    }
    c216_eig_dic = {
        'fun': constraintEig216,
        'type': 'ineq'
    }
    c217_eig_dic = {
        'fun': constraintEig217,
        'type': 'ineq'
    }
    c218_eig_dic = {
        'fun': constraintEig218,
        'type': 'ineq'
    }
    c219_eig_dic = {
        'fun': constraintEig219,
        'type': 'ineq'
    }
    c220_eig_dic = {
        'fun': constraintEig220,
        'type': 'ineq'
    }
    c221_eig_dic = {
        'fun': constraintEig221,
        'type': 'ineq'
    }
    c222_eig_dic = {
        'fun': constraintEig222,
        'type': 'ineq'
    }
    c223_eig_dic = {
        'fun': constraintEig223,
        'type': 'ineq'
    }
    c224_eig_dic = {
        'fun': constraintEig224,
        'type': 'ineq'
    }
    c225_eig_dic = {
        'fun': constraintEig225,
        'type': 'ineq'
    }
    c226_eig_dic = {
        'fun': constraintEig226,
        'type': 'ineq'
    }
    c227_eig_dic = {
        'fun': constraintEig227,
        'type': 'ineq'
    }
    c228_eig_dic = {
        'fun': constraintEig228,
        'type': 'ineq'
    }
    c229_eig_dic = {
        'fun': constraintEig229,
        'type': 'ineq'
    }
    c230_eig_dic = {
        'fun': constraintEig230,
        'type': 'ineq'
    }
    c231_eig_dic = {
        'fun': constraintEig231,
        'type': 'ineq'
    }
    c232_eig_dic = {
        'fun': constraintEig232,
        'type': 'ineq'
    }
    c233_eig_dic = {
        'fun': constraintEig233,
        'type': 'ineq'
    }
    c234_eig_dic = {
        'fun': constraintEig234,
        'type': 'ineq'
    }
    c235_eig_dic = {
        'fun': constraintEig235,
        'type': 'ineq'
    }
    c236_eig_dic = {
        'fun': constraintEig236,
        'type': 'ineq'
    }
    c237_eig_dic = {
        'fun': constraintEig237,
        'type': 'ineq'
    }
    c238_eig_dic = {
        'fun': constraintEig238,
        'type': 'ineq'
    }
    c239_eig_dic = {
        'fun': constraintEig239,
        'type': 'ineq'
    }
    c240_eig_dic = {
        'fun': constraintEig240,
        'type': 'ineq'
    }
    c241_eig_dic = {
        'fun': constraintEig241,
        'type': 'ineq'
    }
    c242_eig_dic = {
        'fun': constraintEig242,
        'type': 'ineq'
    }
    c243_eig_dic = {
        'fun': constraintEig243,
        'type': 'ineq'
    }
    c244_eig_dic = {
        'fun': constraintEig244,
        'type': 'ineq'
    }
    c245_eig_dic = {
        'fun': constraintEig245,
        'type': 'ineq'
    }
    c246_eig_dic = {
        'fun': constraintEig246,
        'type': 'ineq'
    }
    c247_eig_dic = {
        'fun': constraintEig247,
        'type': 'ineq'
    }
    c248_eig_dic = {
        'fun': constraintEig248,
        'type': 'ineq'
    }
    c249_eig_dic = {
        'fun': constraintEig249,
        'type': 'ineq'
    }
    c250_eig_dic = {
        'fun': constraintEig250,
        'type': 'ineq'
    }
    c251_eig_dic = {
        'fun': constraintEig251,
        'type': 'ineq'
    }
    c252_eig_dic = {
        'fun': constraintEig252,
        'type': 'ineq'
    }
    c253_eig_dic = {
        'fun': constraintEig253,
        'type': 'ineq'
    }
    c254_eig_dic = {
        'fun': constraintEig254,
        'type': 'ineq'
    }
    c255_eig_dic = {
        'fun': constraintEig255,
        'type': 'ineq'
    }

    constraintsEig = [c0_eig_dic,c1_eig_dic,c2_eig_dic,c3_eig_dic,c4_eig_dic,c5_eig_dic,c6_eig_dic,c7_eig_dic,c8_eig_dic,c9_eig_dic,c10_eig_dic,c11_eig_dic,c12_eig_dic,c13_eig_dic,c14_eig_dic,c15_eig_dic,c16_eig_dic,c17_eig_dic,c18_eig_dic,c19_eig_dic,c20_eig_dic,c21_eig_dic,c22_eig_dic,c23_eig_dic,c24_eig_dic,c25_eig_dic,c26_eig_dic,c27_eig_dic,c28_eig_dic,c29_eig_dic,c30_eig_dic,c31_eig_dic,c32_eig_dic,c33_eig_dic,c34_eig_dic,c35_eig_dic,c36_eig_dic,c37_eig_dic,c38_eig_dic,c39_eig_dic,c40_eig_dic,c41_eig_dic,c42_eig_dic,c43_eig_dic,c44_eig_dic,c45_eig_dic,c46_eig_dic,c47_eig_dic,c48_eig_dic,c49_eig_dic,c50_eig_dic,c51_eig_dic,c52_eig_dic,c53_eig_dic,c54_eig_dic,c55_eig_dic,c56_eig_dic,c57_eig_dic,c58_eig_dic,c59_eig_dic,c60_eig_dic,c61_eig_dic,c62_eig_dic,c63_eig_dic]
    constraints1Eig = [c64_eig_dic, c65_eig_dic, c66_eig_dic, c67_eig_dic, c68_eig_dic, c69_eig_dic, c70_eig_dic, c71_eig_dic,
                      c72_eig_dic, c73_eig_dic, c74_eig_dic, c75_eig_dic, c76_eig_dic, c77_eig_dic, c78_eig_dic,
                      c79_eig_dic, c80_eig_dic, c81_eig_dic, c82_eig_dic, c83_eig_dic, c84_eig_dic, c85_eig_dic,
                      c86_eig_dic, c87_eig_dic, c88_eig_dic, c89_eig_dic, c90_eig_dic, c91_eig_dic, c92_eig_dic,
                      c93_eig_dic, c94_eig_dic, c95_eig_dic, c96_eig_dic, c97_eig_dic, c98_eig_dic, c99_eig_dic,
                      c100_eig_dic, c101_eig_dic, c102_eig_dic, c103_eig_dic, c104_eig_dic, c105_eig_dic, c106_eig_dic,
                      c107_eig_dic, c108_eig_dic, c109_eig_dic, c110_eig_dic, c111_eig_dic, c112_eig_dic, c113_eig_dic,
                      c114_eig_dic, c115_eig_dic, c116_eig_dic, c117_eig_dic, c118_eig_dic, c119_eig_dic, c120_eig_dic,
                      c121_eig_dic, c122_eig_dic, c123_eig_dic, c124_eig_dic, c125_eig_dic, c126_eig_dic, c127_eig_dic]
    constraints2Eig = [c128_eig_dic, c129_eig_dic, c130_eig_dic,c131_eig_dic,
                       c132_eig_dic, c133_eig_dic, c134_eig_dic, c135_eig_dic, c136_eig_dic, c137_eig_dic, c138_eig_dic,
                       c139_eig_dic, c140_eig_dic, c141_eig_dic, c142_eig_dic, c143_eig_dic, c144_eig_dic, c145_eig_dic,
                       c146_eig_dic, c147_eig_dic, c148_eig_dic, c149_eig_dic, c150_eig_dic, c151_eig_dic, c152_eig_dic,
                       c153_eig_dic, c154_eig_dic, c155_eig_dic, c156_eig_dic, c157_eig_dic, c158_eig_dic, c159_eig_dic,
                       c160_eig_dic, c161_eig_dic, c162_eig_dic, c163_eig_dic, c164_eig_dic, c165_eig_dic, c166_eig_dic,
                       c167_eig_dic, c168_eig_dic, c169_eig_dic, c170_eig_dic, c171_eig_dic, c172_eig_dic, c173_eig_dic,
                       c174_eig_dic, c175_eig_dic, c176_eig_dic, c177_eig_dic, c178_eig_dic, c179_eig_dic, c180_eig_dic,
                       c181_eig_dic, c182_eig_dic, c183_eig_dic, c184_eig_dic, c185_eig_dic, c186_eig_dic, c187_eig_dic]
    constraints3Eig = [c188_eig_dic, c189_eig_dic, c190_eig_dic,c191_eig_dic,
                       c192_eig_dic, c193_eig_dic, c194_eig_dic, c195_eig_dic, c196_eig_dic, c197_eig_dic, c198_eig_dic,
                       c199_eig_dic, c200_eig_dic, c201_eig_dic, c202_eig_dic, c203_eig_dic, c204_eig_dic, c205_eig_dic,
                       c206_eig_dic, c207_eig_dic, c208_eig_dic, c209_eig_dic, c210_eig_dic, c211_eig_dic, c212_eig_dic,
                       c213_eig_dic, c214_eig_dic, c215_eig_dic, c216_eig_dic, c217_eig_dic, c218_eig_dic, c219_eig_dic,
                       c220_eig_dic, c221_eig_dic, c222_eig_dic, c223_eig_dic, c224_eig_dic, c225_eig_dic, c226_eig_dic,
                       c227_eig_dic, c228_eig_dic, c229_eig_dic, c230_eig_dic, c231_eig_dic, c232_eig_dic, c233_eig_dic,
                       c234_eig_dic, c235_eig_dic, c236_eig_dic, c237_eig_dic, c238_eig_dic, c239_eig_dic, c240_eig_dic,
                       c241_eig_dic, c242_eig_dic, c243_eig_dic, c244_eig_dic, c245_eig_dic, c246_eig_dic, c247_eig_dic,
                       c248_eig_dic, c249_eig_dic, c250_eig_dic, c251_eig_dic, c252_eig_dic, c253_eig_dic, c254_eig_dic,
                       c255_eig_dic]
    constraintsEig.extend(constraints1Eig)
    constraintsEig.extend(constraints2Eig)
    constraintsEig.extend(constraints3Eig)

    return constraintsEig[: N - 1]

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
    cL101_dic = {
        'fun': constraintL101,
        'type': 'ineq',
    }
    cU101_dic = {
        'fun': constraintU101,
        'type': 'ineq',
    }
    cL102_dic = {
        'fun': constraintL102,
        'type': 'ineq',
    }
    cU102_dic = {
        'fun': constraintU102,
        'type': 'ineq',
    }
    cL103_dic = {
        'fun': constraintL103,
        'type': 'ineq',
    }
    cU103_dic = {
        'fun': constraintU103,
        'type': 'ineq',
    }
    cL104_dic = {
        'fun': constraintL104,
        'type': 'ineq',
    }
    cU104_dic = {
        'fun': constraintU104,
        'type': 'ineq',
    }
    cL105_dic = {
        'fun': constraintL105,
        'type': 'ineq',
    }
    cU105_dic = {
        'fun': constraintU105,
        'type': 'ineq',
    }
    cL106_dic = {
        'fun': constraintL106,
        'type': 'ineq',
    }
    cU106_dic = {
        'fun': constraintU106,
        'type': 'ineq',
    }

    cL107_dic = {
        'fun': constraintL107,
        'type': 'ineq',
    }
    cU107_dic = {
        'fun': constraintU107,
        'type': 'ineq',
    }
    cL108_dic = {
        'fun': constraintL108,
        'type': 'ineq',
    }
    cU108_dic = {
        'fun': constraintU108,
        'type': 'ineq',
    }
    cL109_dic = {
        'fun': constraintL109,
        'type': 'ineq',
    }
    cU109_dic = {
        'fun': constraintU109,
        'type': 'ineq',
    }
    cL110_dic = {
        'fun': constraintL110,
        'type': 'ineq',
    }
    cU110_dic = {
        'fun': constraintU110,
        'type': 'ineq',
    }
    cL111_dic = {
        'fun': constraintL111,
        'type': 'ineq',
    }
    cU111_dic = {
        'fun': constraintU111,
        'type': 'ineq',
    }
    cL112_dic = {
        'fun': constraintL112,
        'type': 'ineq',
    }
    cU112_dic = {
        'fun': constraintU112,
        'type': 'ineq',
    }
    cL113_dic = {
        'fun': constraintL113,
        'type': 'ineq',
    }
    cU113_dic = {
        'fun': constraintU113,
        'type': 'ineq',
    }
    cL114_dic = {
        'fun': constraintL114,
        'type': 'ineq',
    }
    cU114_dic = {
        'fun': constraintU114,
        'type': 'ineq',
    }
    cL115_dic = {
        'fun': constraintL115,
        'type': 'ineq',
    }
    cU115_dic = {
        'fun': constraintU115,
        'type': 'ineq',
    }
    cL116_dic = {
        'fun': constraintL116,
        'type': 'ineq',
    }
    cU116_dic = {
        'fun': constraintU116,
        'type': 'ineq',
    }
    cL117_dic = {
        'fun': constraintL117,
        'type': 'ineq',
    }
    cU117_dic = {
        'fun': constraintU117,
        'type': 'ineq',
    }
    cL118_dic = {
        'fun': constraintL118,
        'type': 'ineq',
    }
    cU118_dic = {
        'fun': constraintU118,
        'type': 'ineq',
    }
    cL119_dic = {
        'fun': constraintL119,
        'type': 'ineq',
    }
    cU119_dic = {
        'fun': constraintU119,
        'type': 'ineq',
    }
    cL120_dic = {
        'fun': constraintL120,
        'type': 'ineq',
    }
    cU120_dic = {
        'fun': constraintU120,
        'type': 'ineq',
    }
    cL121_dic = {
        'fun': constraintL121,
        'type': 'ineq',
    }
    cU121_dic = {
        'fun': constraintU121,
        'type': 'ineq',
    }
    cL122_dic = {
        'fun': constraintL122,
        'type': 'ineq',
    }
    cU122_dic = {
        'fun': constraintU122,
        'type': 'ineq',
    }
    cL123_dic = {
        'fun': constraintL123,
        'type': 'ineq',
    }
    cU123_dic = {
        'fun': constraintU123,
        'type': 'ineq',
    }
    cL124_dic = {
        'fun': constraintL124,
        'type': 'ineq',
    }
    cU124_dic = {
        'fun': constraintU124,
        'type': 'ineq',
    }
    cL125_dic = {
        'fun': constraintL125,
        'type': 'ineq',
    }
    cU125_dic = {
        'fun': constraintU125,
        'type': 'ineq',
    }
    cL126_dic = {
        'fun': constraintL126,
        'type': 'ineq',
    }
    cU126_dic = {
        'fun': constraintU126,
        'type': 'ineq',
    }
    cL127_dic = {
        'fun': constraintL127,
        'type': 'ineq',
    }
    cU127_dic = {
        'fun': constraintU127,
        'type': 'ineq',
    }
    cL128_dic = {
        'fun': constraintL128,
        'type': 'ineq',
    }
    cU128_dic = {
        'fun': constraintU128,
        'type': 'ineq',
    }
    cL129_dic = {
        'fun': constraintL129,
        'type': 'ineq',
    }
    cU129_dic = {
        'fun': constraintU129,
        'type': 'ineq',
    }
    cL130_dic = {
        'fun': constraintL130,
        'type': 'ineq',
    }
    cU130_dic = {
        'fun': constraintU130,
        'type': 'ineq',
    }
    cL131_dic = {
        'fun': constraintL131,
        'type': 'ineq',
    }
    cU131_dic = {
        'fun': constraintU131,
        'type': 'ineq',
    }
    cL132_dic = {
        'fun': constraintL132,
        'type': 'ineq',
    }
    cU132_dic = {
        'fun': constraintU132,
        'type': 'ineq',
    }
    cL133_dic = {
        'fun': constraintL133,
        'type': 'ineq',
    }
    cU133_dic = {
        'fun': constraintU133,
        'type': 'ineq',
    }
    cL134_dic = {
        'fun': constraintL134,
        'type': 'ineq',
    }
    cU134_dic = {
        'fun': constraintU134,
        'type': 'ineq',
    }
    cL135_dic = {
        'fun': constraintL135,
        'type': 'ineq',
    }
    cU135_dic = {
        'fun': constraintU135,
        'type': 'ineq',
    }
    cL136_dic = {
        'fun': constraintL136,
        'type': 'ineq',
    }
    cU136_dic = {
        'fun': constraintU136,
        'type': 'ineq',
    }
    cL137_dic = {
        'fun': constraintL137,
        'type': 'ineq',
    }
    cU137_dic = {
        'fun': constraintU137,
        'type': 'ineq',
    }
    cL138_dic = {
        'fun': constraintL138,
        'type': 'ineq',
    }
    cU138_dic = {
        'fun': constraintU138,
        'type': 'ineq',
    }
    cL139_dic = {
        'fun': constraintL139,
        'type': 'ineq',
    }
    cU139_dic = {
        'fun': constraintU139,
        'type': 'ineq',
    }
    cL140_dic = {
        'fun': constraintL140,
        'type': 'ineq',
    }
    cU140_dic = {
        'fun': constraintU140,
        'type': 'ineq',
    }
    cL141_dic = {
        'fun': constraintL141,
        'type': 'ineq',
    }
    cU141_dic = {
        'fun': constraintU141,
        'type': 'ineq',
    }
    cL142_dic = {
        'fun': constraintL142,
        'type': 'ineq',
    }
    cU142_dic = {
        'fun': constraintU142,
        'type': 'ineq',
    }
    cL143_dic = {
        'fun': constraintL143,
        'type': 'ineq',
    }
    cU143_dic = {
        'fun': constraintU143,
        'type': 'ineq',
    }
    cL144_dic = {
        'fun': constraintL144,
        'type': 'ineq',
    }
    cU144_dic = {
        'fun': constraintU144,
        'type': 'ineq',
    }
    cL145_dic = {
        'fun': constraintL145,
        'type': 'ineq',
    }
    cU145_dic = {
        'fun': constraintU145,
        'type': 'ineq',
    }
    cL146_dic = {
        'fun': constraintL146,
        'type': 'ineq',
    }
    cU146_dic = {
        'fun': constraintU146,
        'type': 'ineq',
    }
    cL147_dic = {
        'fun': constraintL147,
        'type': 'ineq',
    }
    cU147_dic = {
        'fun': constraintU147,
        'type': 'ineq',
    }
    cL148_dic = {
        'fun': constraintL148,
        'type': 'ineq',
    }
    cU148_dic = {
        'fun': constraintU148,
        'type': 'ineq',
    }
    cL149_dic = {
        'fun': constraintL149,
        'type': 'ineq',
    }
    cU149_dic = {
        'fun': constraintU149,
        'type': 'ineq',
    }
    cL150_dic = {
        'fun': constraintL150,
        'type': 'ineq',
    }
    cU150_dic = {
        'fun': constraintU150,
        'type': 'ineq',
    }
    cL151_dic = {
        'fun': constraintL151,
        'type': 'ineq',
    }
    cU151_dic = {
        'fun': constraintU151,
        'type': 'ineq',
    }
    cL152_dic = {
        'fun': constraintL152,
        'type': 'ineq',
    }
    cU152_dic = {
        'fun': constraintU152,
        'type': 'ineq',
    }
    cL153_dic = {
        'fun': constraintL153,
        'type': 'ineq',
    }
    cU153_dic = {
        'fun': constraintU153,
        'type': 'ineq',
    }
    cL154_dic = {
        'fun': constraintL154,
        'type': 'ineq',
    }
    cU154_dic = {
        'fun': constraintU154,
        'type': 'ineq',
    }
    cL155_dic = {
        'fun': constraintL155,
        'type': 'ineq',
    }
    cU155_dic = {
        'fun': constraintU155,
        'type': 'ineq',
    }
    cL156_dic = {
        'fun': constraintL156,
        'type': 'ineq',
    }
    cU156_dic = {
        'fun': constraintU156,
        'type': 'ineq',
    }
    cL157_dic = {
        'fun': constraintL157,
        'type': 'ineq',
    }
    cU157_dic = {
        'fun': constraintU157,
        'type': 'ineq',
    }
    cL158_dic = {
        'fun': constraintL158,
        'type': 'ineq',
    }
    cU158_dic = {
        'fun': constraintU158,
        'type': 'ineq',
    }
    cL159_dic = {
        'fun': constraintL159,
        'type': 'ineq',
    }
    cU159_dic = {
        'fun': constraintU159,
        'type': 'ineq',
    }
    cL160_dic = {
        'fun': constraintL160,
        'type': 'ineq',
    }
    cU160_dic = {
        'fun': constraintU160,
        'type': 'ineq',
    }
    cL161_dic = {
        'fun': constraintL161,
        'type': 'ineq',
    }
    cU161_dic = {
        'fun': constraintU161,
        'type': 'ineq',
    }
    cL162_dic = {
        'fun': constraintL162,
        'type': 'ineq',
    }
    cU162_dic = {
        'fun': constraintU162,
        'type': 'ineq',
    }
    cL163_dic = {
        'fun': constraintL163,
        'type': 'ineq',
    }
    cU163_dic = {
        'fun': constraintU163,
        'type': 'ineq',
    }
    cL164_dic = {
        'fun': constraintL164,
        'type': 'ineq',
    }
    cU164_dic = {
        'fun': constraintU164,
        'type': 'ineq',
    }
    cL165_dic = {
        'fun': constraintL165,
        'type': 'ineq',
    }
    cU165_dic = {
        'fun': constraintU165,
        'type': 'ineq',
    }
    cL166_dic = {
        'fun': constraintL166,
        'type': 'ineq',
    }
    cU166_dic = {
        'fun': constraintU166,
        'type': 'ineq',
    }
    cL167_dic = {
        'fun': constraintL167,
        'type': 'ineq',
    }
    cU167_dic = {
        'fun': constraintU167,
        'type': 'ineq',
    }
    cL168_dic = {
        'fun': constraintL168,
        'type': 'ineq',
    }
    cU168_dic = {
        'fun': constraintU168,
        'type': 'ineq',
    }
    cL169_dic = {
        'fun': constraintL169,
        'type': 'ineq',
    }
    cU169_dic = {
        'fun': constraintU169,
        'type': 'ineq',
    }
    cL170_dic = {
        'fun': constraintL170,
        'type': 'ineq',
    }
    cU170_dic = {
        'fun': constraintU170,
        'type': 'ineq',
    }
    cL171_dic = {
        'fun': constraintL171,
        'type': 'ineq',
    }
    cU171_dic = {
        'fun': constraintU171,
        'type': 'ineq',
    }
    cL172_dic = {
        'fun': constraintL172,
        'type': 'ineq',
    }
    cU172_dic = {
        'fun': constraintU172,
        'type': 'ineq',
    }
    cL173_dic = {
        'fun': constraintL173,
        'type': 'ineq',
    }
    cU173_dic = {
        'fun': constraintU160,
        'type': 'ineq',
    }
    cL174_dic = {
        'fun': constraintL174,
        'type': 'ineq',
    }
    cU174_dic = {
        'fun': constraintU174,
        'type': 'ineq',
    }
    cL175_dic = {
        'fun': constraintL175,
        'type': 'ineq',
    }
    cU175_dic = {
        'fun': constraintU175,
        'type': 'ineq',
    }
    cL176_dic = {
        'fun': constraintL176,
        'type': 'ineq',
    }
    cU176_dic = {
        'fun': constraintU176,
        'type': 'ineq',
    }
    cL177_dic = {
        'fun': constraintL177,
        'type': 'ineq',
    }
    cU177_dic = {
        'fun': constraintU177,
        'type': 'ineq',
    }
    cL178_dic = {
        'fun': constraintL178,
        'type': 'ineq',
    }
    cU178_dic = {
        'fun': constraintU178,
        'type': 'ineq',
    }
    cL179_dic = {
        'fun': constraintL179,
        'type': 'ineq',
    }
    cU179_dic = {
        'fun': constraintU179,
        'type': 'ineq',
    }
    cL180_dic = {
        'fun': constraintL180,
        'type': 'ineq',
    }
    cU180_dic = {
        'fun': constraintU180,
        'type': 'ineq',
    }
    cL181_dic = {
        'fun': constraintL181,
        'type': 'ineq',
    }
    cU181_dic = {
        'fun': constraintU181,
        'type': 'ineq',
    }
    cL182_dic = {
        'fun': constraintL182,
        'type': 'ineq',
    }
    cU182_dic = {
        'fun': constraintU182,
        'type': 'ineq',
    }
    cL183_dic = {
        'fun': constraintL183,
        'type': 'ineq',
    }
    cU183_dic = {
        'fun': constraintU183,
        'type': 'ineq',
    }
    cL184_dic = {
        'fun': constraintL184,
        'type': 'ineq',
    }
    cU184_dic = {
        'fun': constraintU184,
        'type': 'ineq',
    }
    cL185_dic = {
        'fun': constraintL185,
        'type': 'ineq',
    }
    cU185_dic = {
        'fun': constraintU185,
        'type': 'ineq',
    }
    cL186_dic = {
        'fun': constraintL186,
        'type': 'ineq',
    }
    cU186_dic = {
        'fun': constraintU186,
        'type': 'ineq',
    }
    cL187_dic = {
        'fun': constraintL187,
        'type': 'ineq',
    }
    cU187_dic = {
        'fun': constraintU187,
        'type': 'ineq',
    }
    cL188_dic = {
        'fun': constraintL188,
        'type': 'ineq',
    }
    cU188_dic = {
        'fun': constraintU188,
        'type': 'ineq',
    }
    cL189_dic = {
        'fun': constraintL189,
        'type': 'ineq',
    }
    cU189_dic = {
        'fun': constraintU189,
        'type': 'ineq',
    }
    cL190_dic = {
        'fun': constraintL190,
        'type': 'ineq',
    }
    cU190_dic = {
        'fun': constraintU190,
        'type': 'ineq',
    }
    cL191_dic = {
        'fun': constraintL191,
        'type': 'ineq',
    }
    cU191_dic = {
        'fun': constraintU191,
        'type': 'ineq',
    }
    cL192_dic = {
        'fun': constraintL192,
        'type': 'ineq',
    }
    cU192_dic = {
        'fun': constraintU192,
        'type': 'ineq',
    }
    cL193_dic = {
        'fun': constraintL193,
        'type': 'ineq',
    }
    cU193_dic = {
        'fun': constraintU193,
        'type': 'ineq',
    }
    cL194_dic = {
        'fun': constraintL194,
        'type': 'ineq',
    }
    cU194_dic = {
        'fun': constraintU194,
        'type': 'ineq',
    }
    cL195_dic = {
        'fun': constraintL195,
        'type': 'ineq',
    }
    cU195_dic = {
        'fun': constraintU195,
        'type': 'ineq',
    }
    cL196_dic = {
        'fun': constraintL196,
        'type': 'ineq',
    }
    cU196_dic = {
        'fun': constraintU196,
        'type': 'ineq',
    }
    cL197_dic = {
        'fun': constraintL197,
        'type': 'ineq',
    }
    cU197_dic = {
        'fun': constraintU197,
        'type': 'ineq',
    }
    cL198_dic = {
        'fun': constraintL198,
        'type': 'ineq',
    }
    cU198_dic = {
        'fun': constraintU198,
        'type': 'ineq',
    }
    cL199_dic = {
        'fun': constraintL199,
        'type': 'ineq',
    }
    cU199_dic = {
        'fun': constraintU199,
        'type': 'ineq',
    }
    cL200_dic = {
        'fun': constraintL200,
        'type': 'ineq',
    }
    cU200_dic = {
        'fun': constraintU200,
        'type': 'ineq',
    }
    cL201_dic = {
        'fun': constraintL201,
        'type': 'ineq',
    }
    cU201_dic = {
        'fun': constraintU201,
        'type': 'ineq',
    }
    cL202_dic = {
        'fun': constraintL202,
        'type': 'ineq',
    }
    cU202_dic = {
        'fun': constraintU202,
        'type': 'ineq',
    }
    cL203_dic = {
        'fun': constraintL203,
        'type': 'ineq',
    }
    cU203_dic = {
        'fun': constraintU203,
        'type': 'ineq',
    }
    cL204_dic = {
        'fun': constraintL204,
        'type': 'ineq',
    }
    cU204_dic = {
        'fun': constraintU204,
        'type': 'ineq',
    }
    cL205_dic = {
        'fun': constraintL205,
        'type': 'ineq',
    }
    cU205_dic = {
        'fun': constraintU205,
        'type': 'ineq',
    }
    cL206_dic = {
        'fun': constraintL206,
        'type': 'ineq',
    }
    cU206_dic = {
        'fun': constraintU206,
        'type': 'ineq',
    }
    cL207_dic = {
        'fun': constraintL207,
        'type': 'ineq',
    }
    cU207_dic = {
        'fun': constraintU207,
        'type': 'ineq',
    }
    cL208_dic = {
        'fun': constraintL208,
        'type': 'ineq',
    }
    cU208_dic = {
        'fun': constraintU208,
        'type': 'ineq',
    }
    cL209_dic = {
        'fun': constraintL209,
        'type': 'ineq',
    }
    cU209_dic = {
        'fun': constraintU209,
        'type': 'ineq',
    }
    cL210_dic = {
        'fun': constraintL210,
        'type': 'ineq',
    }
    cU210_dic = {
        'fun': constraintU210,
        'type': 'ineq',
    }
    cL211_dic = {
        'fun': constraintL211,
        'type': 'ineq',
    }
    cU211_dic = {
        'fun': constraintU211,
        'type': 'ineq',
    }
    cL212_dic = {
        'fun': constraintL212,
        'type': 'ineq',
    }
    cU212_dic = {
        'fun': constraintU212,
        'type': 'ineq',
    }
    cL213_dic = {
        'fun': constraintL213,
        'type': 'ineq',
    }
    cU213_dic = {
        'fun': constraintU213,
        'type': 'ineq',
    }
    cL214_dic = {
        'fun': constraintL214,
        'type': 'ineq',
    }
    cU214_dic = {
        'fun': constraintU214,
        'type': 'ineq',
    }
    cL215_dic = {
        'fun': constraintL215,
        'type': 'ineq',
    }
    cU215_dic = {
        'fun': constraintU215,
        'type': 'ineq',
    }
    cL216_dic = {
        'fun': constraintL216,
        'type': 'ineq',
    }
    cU216_dic = {
        'fun': constraintU216,
        'type': 'ineq',
    }
    cL217_dic = {
        'fun': constraintL217,
        'type': 'ineq',
    }
    cU217_dic = {
        'fun': constraintU217,
        'type': 'ineq',
    }
    cL218_dic = {
        'fun': constraintL218,
        'type': 'ineq',
    }
    cU218_dic = {
        'fun': constraintU218,
        'type': 'ineq',
    }
    cL219_dic = {
        'fun': constraintL219,
        'type': 'ineq',
    }
    cU219_dic = {
        'fun': constraintU219,
        'type': 'ineq',
    }
    cL220_dic = {
        'fun': constraintL220,
        'type': 'ineq',
    }
    cU220_dic = {
        'fun': constraintU220,
        'type': 'ineq',
    }
    cL221_dic = {
        'fun': constraintL221,
        'type': 'ineq',
    }
    cU221_dic = {
        'fun': constraintU221,
        'type': 'ineq',
    }
    cL222_dic = {
        'fun': constraintL222,
        'type': 'ineq',
    }
    cU222_dic = {
        'fun': constraintU222,
        'type': 'ineq',
    }
    cL223_dic = {
        'fun': constraintL223,
        'type': 'ineq',
    }
    cU223_dic = {
        'fun': constraintU223,
        'type': 'ineq',
    }
    cL224_dic = {
        'fun': constraintL224,
        'type': 'ineq',
    }
    cU224_dic = {
        'fun': constraintU224,
        'type': 'ineq',
    }
    cL225_dic = {
        'fun': constraintL225,
        'type': 'ineq',
    }
    cU225_dic = {
        'fun': constraintU225,
        'type': 'ineq',
    }
    cL226_dic = {
        'fun': constraintL226,
        'type': 'ineq',
    }
    cU226_dic = {
        'fun': constraintU226,
        'type': 'ineq',
    }
    cL227_dic = {
        'fun': constraintL227,
        'type': 'ineq',
    }
    cU227_dic = {
        'fun': constraintU227,
        'type': 'ineq',
    }
    cL228_dic = {
        'fun': constraintL228,
        'type': 'ineq',
    }
    cU228_dic = {
        'fun': constraintU228,
        'type': 'ineq',
    }
    cL229_dic = {
        'fun': constraintL229,
        'type': 'ineq',
    }
    cU229_dic = {
        'fun': constraintU229,
        'type': 'ineq',
    }
    cL230_dic = {
        'fun': constraintL230,
        'type': 'ineq',
    }
    cU230_dic = {
        'fun': constraintU230,
        'type': 'ineq',
    }
    cL231_dic = {
        'fun': constraintL231,
        'type': 'ineq',
    }
    cU231_dic = {
        'fun': constraintU231,
        'type': 'ineq',
    }
    cL232_dic = {
        'fun': constraintL232,
        'type': 'ineq',
    }
    cU232_dic = {
        'fun': constraintU232,
        'type': 'ineq',
    }
    cL233_dic = {
        'fun': constraintL233,
        'type': 'ineq',
    }
    cU233_dic = {
        'fun': constraintU233,
        'type': 'ineq',
    }
    cL234_dic = {
        'fun': constraintL234,
        'type': 'ineq',
    }
    cU234_dic = {
        'fun': constraintU234,
        'type': 'ineq',
    }
    cL235_dic = {
        'fun': constraintL235,
        'type': 'ineq',
    }
    cU235_dic = {
        'fun': constraintU235,
        'type': 'ineq',
    }
    cL236_dic = {
        'fun': constraintL236,
        'type': 'ineq',
    }
    cU236_dic = {
        'fun': constraintU236,
        'type': 'ineq',
    }
    cL237_dic = {
        'fun': constraintL237,
        'type': 'ineq',
    }
    cU237_dic = {
        'fun': constraintU237,
        'type': 'ineq',
    }
    cL238_dic = {
        'fun': constraintL238,
        'type': 'ineq',
    }
    cU238_dic = {
        'fun': constraintU238,
        'type': 'ineq',
    }
    cL239_dic = {
        'fun': constraintL239,
        'type': 'ineq',
    }
    cU239_dic = {
        'fun': constraintU239,
        'type': 'ineq',
    }
    cL240_dic = {
        'fun': constraintL240,
        'type': 'ineq',
    }
    cU240_dic = {
        'fun': constraintU240,
        'type': 'ineq',
    }
    cL241_dic = {
        'fun': constraintL241,
        'type': 'ineq',
    }
    cU241_dic = {
        'fun': constraintU241,
        'type': 'ineq',
    }
    cL242_dic = {
        'fun': constraintL242,
        'type': 'ineq',
    }
    cU242_dic = {
        'fun': constraintU242,
        'type': 'ineq',
    }
    cL243_dic = {
        'fun': constraintL243,
        'type': 'ineq',
    }
    cU243_dic = {
        'fun': constraintU243,
        'type': 'ineq',
    }
    cL244_dic = {
        'fun': constraintL244,
        'type': 'ineq',
    }
    cU244_dic = {
        'fun': constraintU244,
        'type': 'ineq',
    }
    cL245_dic = {
        'fun': constraintL245,
        'type': 'ineq',
    }
    cU245_dic = {
        'fun': constraintU245,
        'type': 'ineq',
    }
    cL246_dic = {
        'fun': constraintL246,
        'type': 'ineq',
    }
    cU246_dic = {
        'fun': constraintU246,
        'type': 'ineq',
    }
    cL247_dic = {
        'fun': constraintL247,
        'type': 'ineq',
    }
    cU247_dic = {
        'fun': constraintU247,
        'type': 'ineq',
    }
    cL248_dic = {
        'fun': constraintL248,
        'type': 'ineq',
    }
    cU248_dic = {
        'fun': constraintU248,
        'type': 'ineq',
    }
    cL249_dic = {
        'fun': constraintL249,
        'type': 'ineq',
    }
    cU249_dic = {
        'fun': constraintU249,
        'type': 'ineq',
    }
    cL250_dic = {
        'fun': constraintL250,
        'type': 'ineq',
    }
    cU250_dic = {
        'fun': constraintU250,
        'type': 'ineq',
    }
    cL251_dic = {
        'fun': constraintL251,
        'type': 'ineq',
    }
    cU251_dic = {
        'fun': constraintU251,
        'type': 'ineq',
    }
    cL252_dic = {
        'fun': constraintL252,
        'type': 'ineq',
    }
    cU252_dic = {
        'fun': constraintU252,
        'type': 'ineq',
    }
    cL253_dic = {
        'fun': constraintL253,
        'type': 'ineq',
    }
    cU253_dic = {
        'fun': constraintU253,
        'type': 'ineq',
    }
    cL254_dic = {
        'fun': constraintL254,
        'type': 'ineq',
    }
    cU254_dic = {
        'fun': constraintU254,
        'type': 'ineq',
    }
    cL255_dic = {
        'fun': constraintL255,
        'type': 'ineq',
    }
    cU255_dic = {
        'fun': constraintU255,
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
    constraints4 = [cL101_dic, cU101_dic, cL102_dic, cU102_dic, cL103_dic, cU103_dic, cL104_dic, cU104_dic, cL105_dic, cU105_dic,
                    cL106_dic, cU106_dic, cL107_dic, cU107_dic, cL108_dic, cU108_dic, cL109_dic, cU109_dic, cL110_dic, cU110_dic,
                    cL111_dic, cU111_dic, cL112_dic, cU112_dic, cL113_dic, cU113_dic, cL114_dic, cU114_dic, cL115_dic, cU115_dic,
                    cL116_dic, cU116_dic, cL117_dic, cU117_dic, cL118_dic, cU118_dic, cL119_dic, cU119_dic, cL120_dic, cU120_dic,
                    cL121_dic, cU121_dic, cL122_dic, cU122_dic, cL123_dic, cU123_dic, cL124_dic, cU124_dic, cL125_dic, cU125_dic,
                    cL126_dic, cU126_dic, cL127_dic, cU127_dic, cL128_dic, cU128_dic, cL129_dic, cU129_dic, cL130_dic,cU130_dic]
    constraints5 = [cL131_dic, cU131_dic, cL132_dic, cU132_dic, cL133_dic, cU133_dic, cL134_dic, cU134_dic, cL135_dic, cU135_dic,
                    cL136_dic, cU136_dic, cL137_dic, cU137_dic, cL138_dic, cU138_dic, cL139_dic, cU139_dic, cL140_dic, cU140_dic,
                    cL141_dic, cU141_dic, cL142_dic, cU142_dic, cL143_dic, cU143_dic, cL144_dic, cU144_dic, cL145_dic, cU145_dic,
                    cL146_dic, cU146_dic, cL147_dic, cU147_dic, cL148_dic, cU148_dic, cL149_dic, cU149_dic, cL150_dic, cU150_dic,
                    cL151_dic, cU151_dic, cL152_dic, cU152_dic, cL153_dic, cU153_dic, cL154_dic, cU154_dic, cL155_dic, cU155_dic,
                    cL156_dic, cU156_dic, cL157_dic, cU157_dic, cL158_dic, cU158_dic, cL159_dic, cU159_dic, cL160_dic,cU160_dic]
    constraints6 = [cL161_dic, cU161_dic, cL162_dic, cU162_dic, cL163_dic, cU163_dic, cL164_dic, cU164_dic, cL165_dic, cU165_dic,
                    cL166_dic, cU166_dic, cL167_dic, cU167_dic, cL168_dic, cU168_dic, cL169_dic, cU169_dic, cL170_dic, cU170_dic,
                    cL171_dic, cU171_dic, cL172_dic, cU172_dic, cL173_dic, cU173_dic, cL174_dic, cU174_dic, cL175_dic, cU175_dic,
                    cL176_dic, cU176_dic, cL177_dic, cU177_dic, cL178_dic, cU178_dic, cL179_dic, cU179_dic, cL180_dic, cU180_dic,
                    cL181_dic, cU181_dic, cL182_dic, cU182_dic, cL183_dic, cU183_dic, cL184_dic, cU184_dic, cL185_dic, cU185_dic,
                    cL186_dic, cU186_dic, cL187_dic, cU187_dic, cL188_dic, cU188_dic, cL189_dic, cU189_dic, cL190_dic,cU190_dic]
    constraints7 = [cL191_dic, cU191_dic, cL192_dic, cU192_dic, cL193_dic, cU193_dic, cL194_dic, cU194_dic, cL195_dic, cU195_dic,
                    cL196_dic, cU196_dic, cL197_dic, cU197_dic, cL198_dic, cU198_dic, cL199_dic, cU199_dic, cL200_dic, cU200_dic,
                    cL201_dic, cU201_dic, cL202_dic, cU202_dic, cL203_dic, cU203_dic, cL204_dic, cU204_dic, cL205_dic, cU205_dic,
                    cL206_dic, cU206_dic, cL207_dic, cU207_dic, cL208_dic, cU208_dic, cL209_dic, cU209_dic, cL210_dic, cU210_dic,
                    cL211_dic, cU211_dic, cL212_dic, cU212_dic, cL213_dic, cU213_dic, cL214_dic, cU214_dic, cL215_dic, cU215_dic,
                    cL216_dic, cU216_dic, cL217_dic, cU217_dic, cL218_dic, cU218_dic, cL219_dic, cU219_dic, cL220_dic,cU220_dic]
    constraints8 = [cL221_dic, cU221_dic, cL222_dic, cU222_dic, cL223_dic, cU223_dic, cL224_dic, cU224_dic, cL225_dic, cU225_dic,
                    cL226_dic, cU226_dic, cL227_dic, cU227_dic, cL228_dic, cU228_dic, cL229_dic, cU229_dic, cL230_dic, cU230_dic,
                    cL231_dic, cU231_dic, cL232_dic, cU232_dic, cL233_dic, cU233_dic, cL234_dic, cU234_dic, cL235_dic, cU235_dic,
                    cL236_dic, cU236_dic, cL237_dic, cU237_dic, cL238_dic, cU238_dic, cL239_dic, cU239_dic, cL240_dic, cU240_dic,
                    cL241_dic, cU241_dic, cL242_dic, cU242_dic, cL243_dic, cU243_dic, cL244_dic, cU244_dic, cL245_dic, cU245_dic,
                    cL246_dic, cU246_dic, cL247_dic, cU247_dic, cL248_dic, cU248_dic, cL249_dic, cU249_dic, cL250_dic,cU250_dic,
                    cL251_dic, cU251_dic, cL252_dic, cU252_dic, cL253_dic, cU253_dic, cL254_dic, cU254_dic, cL255_dic,cU255_dic]
    constraints.extend(constraints2)
    constraints.extend(constraints3)
    constraints.extend(constraints4)
    constraints.extend(constraints5)
    constraints.extend(constraints6)
    constraints.extend(constraints7)
    constraints.extend(constraints8)
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
    return [alpha[38] + K[38] * alpha[0]]

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

def constraintU101(alpha):
    return [K[101] * alpha[0] - alpha[101]]
def constraintL101(alpha):
    return [alpha[101] + K[101] * alpha[0]]

def constraintU102(alpha):
    return [K[102] * alpha[0] - alpha[102]]
def constraintL102(alpha):
    return [alpha[102] + K[102] * alpha[0]]

def constraintU103(alpha):
    return [K[103] * alpha[0] - alpha[103]]
def constraintL103(alpha):
    return [alpha[103] + K[103] * alpha[0]]

def constraintU104(alpha):
    return [K[104] * alpha[0] - alpha[104]]
def constraintL104(alpha):
    return [alpha[104] + K[104] * alpha[0]]

def constraintU105(alpha):
    return [K[105] * alpha[0] - alpha[105]]
def constraintL105(alpha):
    return [alpha[105] + K[105] * alpha[0]]

def constraintU106(alpha):
    return [K[106] * alpha[0] - alpha[106]]
def constraintL106(alpha):
    return [alpha[106] + K[106] * alpha[0]]

def constraintU107(alpha):
    return [K[107] * alpha[0] - alpha[107]]
def constraintL107(alpha):
    return [alpha[107] + K[107] * alpha[0]]

def constraintU108(alpha):
    return [K[108] * alpha[0] - alpha[108]]
def constraintL108(alpha):
    return [alpha[108] + K[108] * alpha[0]]

def constraintU109(alpha):
    return [K[109] * alpha[0] - alpha[109]]
def constraintL109(alpha):
    return [alpha[109] + K[109] * alpha[0]]

def constraintU110(alpha):
    return [K[110] * alpha[0] - alpha[110]]
def constraintL110(alpha):
    return [alpha[110] + K[110] * alpha[0]]

def constraintU111(alpha):
    return [K[111] * alpha[0] - alpha[111]]
def constraintL111(alpha):
    return [alpha[111] + K[111] * alpha[0]]

def constraintU112(alpha):
    return [K[112] * alpha[0] - alpha[112]]
def constraintL112(alpha):
    return [alpha[112] + K[112] * alpha[0]]

def constraintU113(alpha):
    return [K[113] * alpha[0] - alpha[113]]
def constraintL113(alpha):
    return [alpha[113] + K[113] * alpha[0]]

def constraintU114(alpha):
    return [K[114] * alpha[0] - alpha[114]]
def constraintL114(alpha):
    return [alpha[114] + K[114] * alpha[0]]

def constraintU115(alpha):
    return [K[115] * alpha[0] - alpha[115]]
def constraintL115(alpha):
    return [alpha[115] + K[115] * alpha[0]]

def constraintU116(alpha):
    return [K[116] * alpha[0] - alpha[116]]
def constraintL116(alpha):
    return [alpha[116] + K[116] * alpha[0]]

def constraintU117(alpha):
    return [K[117] * alpha[0] - alpha[117]]
def constraintL117(alpha):
    return [alpha[117] + K[117] * alpha[0]]

def constraintU118(alpha):
    return [K[118] * alpha[0] - alpha[118]]
def constraintL118(alpha):
    return [alpha[118] + K[118] * alpha[0]]

def constraintU119(alpha):
    return [K[119] * alpha[0] - alpha[119]]
def constraintL119(alpha):
    return [alpha[119] + K[119] * alpha[0]]

def constraintU120(alpha):
    return [K[120] * alpha[0] - alpha[120]]
def constraintL120(alpha):
    return [alpha[120] + K[120] * alpha[0]]

def constraintU121(alpha):
    return [K[121] * alpha[0] - alpha[121]]
def constraintL121(alpha):
    return [alpha[121] + K[121] * alpha[0]]

def constraintU122(alpha):
    return [K[122] * alpha[0] - alpha[122]]
def constraintL122(alpha):
    return [alpha[122] + K[122] * alpha[0]]

def constraintU123(alpha):
    return [K[123] * alpha[0] - alpha[123]]
def constraintL123(alpha):
    return [alpha[123] + K[123] * alpha[0]]

def constraintU124(alpha):
    return [K[124] * alpha[0] - alpha[124]]
def constraintL124(alpha):
    return [alpha[124] + K[124] * alpha[0]]

def constraintU125(alpha):
    return [K[125] * alpha[0] - alpha[125]]
def constraintL125(alpha):
    return [alpha[125] + K[125] * alpha[0]]

def constraintU126(alpha):
    return [K[126] * alpha[0] - alpha[126]]
def constraintL126(alpha):
    return [alpha[126] + K[126] * alpha[0]]

def constraintU127(alpha):
    return [K[127] * alpha[0] - alpha[127]]
def constraintL127(alpha):
    return [alpha[127] + K[127] * alpha[0]]

def constraintU128(alpha):
    return [K[128] * alpha[0] - alpha[128]]
def constraintL128(alpha):
    return [alpha[128] + K[128] * alpha[0]]

def constraintU129(alpha):
    return [K[129] * alpha[0] - alpha[129]]
def constraintL129(alpha):
    return [alpha[129] + K[129] * alpha[0]]

def constraintU130(alpha):
    return [K[130] * alpha[0] - alpha[130]]
def constraintL130(alpha):
    return [alpha[130] + K[130] * alpha[0]]

def constraintU131(alpha):
    return [K[131] * alpha[0] - alpha[131]]
def constraintL131(alpha):
    return [alpha[131] + K[131] * alpha[0]]

def constraintU132(alpha):
    return [K[132] * alpha[0] - alpha[132]]
def constraintL132(alpha):
    return [alpha[132] + K[132] * alpha[0]]

def constraintU133(alpha):
    return [K[133] * alpha[0] - alpha[133]]
def constraintL133(alpha):
    return [alpha[133] + K[133] * alpha[0]]

def constraintU134(alpha):
    return [K[134] * alpha[0] - alpha[134]]
def constraintL134(alpha):
    return [alpha[134] + K[134] * alpha[0]]

def constraintU135(alpha):
    return [K[135] * alpha[0] - alpha[135]]
def constraintL135(alpha):
    return [alpha[135] + K[135] * alpha[0]]

def constraintU136(alpha):
    return [K[136] * alpha[0] - alpha[136]]
def constraintL136(alpha):
    return [alpha[136] + K[136] * alpha[0]]

def constraintU137(alpha):
    return [K[137] * alpha[0] - alpha[137]]
def constraintL137(alpha):
    return [alpha[137] + K[137] * alpha[0]]

def constraintU138(alpha):
    return [K[138] * alpha[0] - alpha[138]]
def constraintL138(alpha):
    return [alpha[138] + K[138] * alpha[0]]

def constraintU139(alpha):
    return [K[139] * alpha[0] - alpha[139]]
def constraintL139(alpha):
    return [alpha[139] + K[139] * alpha[0]]

def constraintU140(alpha):
    return [K[140] * alpha[0] - alpha[140]]
def constraintL140(alpha):
    return [alpha[140] + K[140] * alpha[0]]

def constraintU141(alpha):
    return [K[141] * alpha[0] - alpha[141]]
def constraintL141(alpha):
    return [alpha[141] + K[141] * alpha[0]]

def constraintU142(alpha):
    return [K[142] * alpha[0] - alpha[142]]
def constraintL142(alpha):
    return [alpha[142] + K[142] * alpha[0]]

def constraintU143(alpha):
    return [K[143] * alpha[0] - alpha[143]]
def constraintL143(alpha):
    return [alpha[143] + K[143] * alpha[0]]

def constraintU144(alpha):
    return [K[144] * alpha[0] - alpha[144]]
def constraintL144(alpha):
    return [alpha[144] + K[144] * alpha[0]]

def constraintU145(alpha):
    return [K[145] * alpha[0] - alpha[145]]
def constraintL145(alpha):
    return [alpha[145] + K[145] * alpha[0]]

def constraintU146(alpha):
    return [K[146] * alpha[0] - alpha[146]]
def constraintL146(alpha):
    return [alpha[146] + K[146] * alpha[0]]

def constraintU147(alpha):
    return [K[147] * alpha[0] - alpha[147]]
def constraintL147(alpha):
    return [alpha[147] + K[147] * alpha[0]]

def constraintU148(alpha):
    return [K[148] * alpha[0] - alpha[148]]
def constraintL148(alpha):
    return [alpha[148] + K[148] * alpha[0]]

def constraintU149(alpha):
    return [K[149] * alpha[0] - alpha[149]]
def constraintL149(alpha):
    return [alpha[149] + K[149] * alpha[0]]

def constraintU150(alpha):
    return [K[150]* alpha[0] - alpha[150]]
def constraintL150(alpha):
    return [alpha[150] + K[150] * alpha[0]]

def constraintU151(alpha):
    return [K[151] * alpha[0] - alpha[151]]
def constraintL151(alpha):
    return [alpha[151] + K[151] * alpha[0]]

def constraintU152(alpha):
    return [K[152] * alpha[0] - alpha[152]]
def constraintL152(alpha):
    return [alpha[152] + K[152] * alpha[0]]

def constraintU153(alpha):
    return [K[153] * alpha[0] - alpha[153]]
def constraintL153(alpha):
    return [alpha[153] + K[153] * alpha[0]]

def constraintU154(alpha):
    return [K[154] * alpha[0] - alpha[154]]
def constraintL154(alpha):
    return [alpha[154] + K[154] * alpha[0]]

def constraintU155(alpha):
    return [K[155] * alpha[0] - alpha[155]]
def constraintL155(alpha):
    return [alpha[155] + K[155] * alpha[0]]

def constraintU156(alpha):
    return [K[156] * alpha[0] - alpha[156]]
def constraintL156(alpha):
    return [alpha[156] + K[156] * alpha[0]]

def constraintU157(alpha):
    return [K[157] * alpha[0] - alpha[157]]
def constraintL157(alpha):
    return [alpha[157] + K[157] * alpha[0]]

def constraintU158(alpha):
    return [K[158] * alpha[0] - alpha[158]]
def constraintL158(alpha):
    return [alpha[158] + K[158] * alpha[0]]

def constraintU159(alpha):
    return [K[159] * alpha[0] - alpha[159]]
def constraintL159(alpha):
    return [alpha[159] + K[159] * alpha[0]]

def constraintU160(alpha):
    return [K[160] * alpha[0] - alpha[160]]
def constraintL160(alpha):
    return [alpha[160] + K[160] * alpha[0]]

def constraintU161(alpha):
    return [K[161] * alpha[0] - alpha[161]]
def constraintL161(alpha):
    return [alpha[161] + K[161] * alpha[0]]

def constraintU162(alpha):
    return [K[162] * alpha[0] - alpha[162]]
def constraintL162(alpha):
    return [alpha[162] + K[162] * alpha[0]]

def constraintU163(alpha):
    return [K[163] * alpha[0] - alpha[163]]
def constraintL163(alpha):
    return [alpha[163] + K[163] * alpha[0]]

def constraintU164(alpha):
    return [K[164] * alpha[0] - alpha[164]]
def constraintL164(alpha):
    return [alpha[164] + K[164] * alpha[0]]

def constraintU165(alpha):
    return [K[165] * alpha[0] - alpha[165]]
def constraintL165(alpha):
    return [alpha[165] + K[165] * alpha[0]]

def constraintU166(alpha):
    return [K[166] * alpha[0] - alpha[166]]
def constraintL166(alpha):
    return [alpha[166] + K[166] * alpha[0]]

def constraintU167(alpha):
    return [K[167] * alpha[0] - alpha[167]]
def constraintL167(alpha):
    return [alpha[167] + K[167] * alpha[0]]

def constraintU168(alpha):
    return [K[168] * alpha[0] - alpha[168]]
def constraintL168(alpha):
    return [alpha[168] + K[168] * alpha[0]]

def constraintU169(alpha):
    return [K[169] * alpha[0] - alpha[169]]
def constraintL169(alpha):
    return [alpha[169] + K[169] * alpha[0]]

def constraintU170(alpha):
    return [K[170] * alpha[0] - alpha[170]]
def constraintL170(alpha):
    return [alpha[170] + K[170] * alpha[0]]

def constraintU171(alpha):
    return [K[171] * alpha[0] - alpha[171]]
def constraintL171(alpha):
    return [alpha[171] + K[171] * alpha[0]]

def constraintU172(alpha):
    return [K[172] * alpha[0] - alpha[172]]
def constraintL172(alpha):
    return [alpha[172] + K[172] * alpha[0]]

def constraintU173(alpha):
    return [K[173] * alpha[0] - alpha[173]]
def constraintL173(alpha):
    return [alpha[173] + K[173] * alpha[0]]

def constraintU174(alpha):
    return [K[174] * alpha[0] - alpha[174]]
def constraintL174(alpha):
    return [alpha[174] + K[174] * alpha[0]]

def constraintU175(alpha):
    return [K[175] * alpha[0] - alpha[175]]
def constraintL175(alpha):
    return [alpha[175] + K[175] * alpha[0]]

def constraintU176(alpha):
    return [K[176] * alpha[0] - alpha[176]]
def constraintL176(alpha):
    return [alpha[176] + K[176] * alpha[0]]

def constraintU177(alpha):
    return [K[177] * alpha[0] - alpha[177]]
def constraintL177(alpha):
    return [alpha[177] + K[177] * alpha[0]]

def constraintU178(alpha):
    return [K[178] * alpha[0] - alpha[178]]
def constraintL178(alpha):
    return [alpha[178] + K[178] * alpha[0]]

def constraintU179(alpha):
    return [K[179] * alpha[0] - alpha[179]]
def constraintL179(alpha):
    return [alpha[179] + K[179] * alpha[0]]

def constraintU180(alpha):
    return [K[180] * alpha[0] - alpha[180]]
def constraintL180(alpha):
    return [alpha[180] + K[180] * alpha[0]]

def constraintU181(alpha):
    return [K[181] * alpha[0] - alpha[181]]
def constraintL181(alpha):
    return [alpha[181] + K[181] * alpha[0]]

def constraintU182(alpha):
    return [K[182] * alpha[0] - alpha[182]]
def constraintL182(alpha):
    return [alpha[182] + K[182] * alpha[0]]

def constraintU183(alpha):
    return [K[183] * alpha[0] - alpha[183]]
def constraintL183(alpha):
    return [alpha[183] + K[183] * alpha[0]]

def constraintU184(alpha):
    return [K[184] * alpha[0] - alpha[184]]
def constraintL184(alpha):
    return [alpha[184] + K[184] * alpha[0]]

def constraintU185(alpha):
    return [K[185] * alpha[0] - alpha[185]]
def constraintL185(alpha):
    return [alpha[185] + K[185] * alpha[0]]

def constraintU186(alpha):
    return [K[186] * alpha[0] - alpha[186]]
def constraintL186(alpha):
    return [alpha[186] + K[186] * alpha[0]]

def constraintU187(alpha):
    return [K[187] * alpha[0] - alpha[187]]
def constraintL187(alpha):
    return [alpha[187] + K[187] * alpha[0]]

def constraintU188(alpha):
    return [K[188] * alpha[0] - alpha[188]]
def constraintL188(alpha):
    return [alpha[188] + K[188] * alpha[0]]

def constraintU189(alpha):
    return [K[189] * alpha[0] - alpha[189]]
def constraintL189(alpha):
    return [alpha[189] + K[189] * alpha[0]]

def constraintU190(alpha):
    return [K[190] * alpha[0] - alpha[190]]
def constraintL190(alpha):
    return [alpha[190] + K[190] * alpha[0]]

def constraintU191(alpha):
    return [K[191] * alpha[0] - alpha[191]]
def constraintL191(alpha):
    return [alpha[191] + K[191] * alpha[0]]

def constraintU192(alpha):
    return [K[192] * alpha[0] - alpha[192]]
def constraintL192(alpha):
    return [alpha[192] + K[192] * alpha[0]]

def constraintU193(alpha):
    return [K[193] * alpha[0] - alpha[193]]
def constraintL193(alpha):
    return [alpha[193] + K[193] * alpha[0]]

def constraintU194(alpha):
    return [K[194] * alpha[0] - alpha[194]]
def constraintL194(alpha):
    return [alpha[194] + K[194] * alpha[0]]

def constraintU195(alpha):
    return [K[195] * alpha[0] - alpha[195]]
def constraintL195(alpha):
    return [alpha[195] + K[195] * alpha[0]]

def constraintU196(alpha):
    return [K[196] * alpha[0] - alpha[196]]
def constraintL196(alpha):
    return [alpha[196] + K[196] * alpha[0]]

def constraintU197(alpha):
    return [K[197] * alpha[0] - alpha[197]]
def constraintL197(alpha):
    return [alpha[197] + K[197] * alpha[0]]

def constraintU198(alpha):
    return [K[198] * alpha[0] - alpha[198]]
def constraintL198(alpha):
    return [alpha[198] + K[198] * alpha[0]]

def constraintU199(alpha):
    return [K[199] * alpha[0] - alpha[199]]
def constraintL199(alpha):
    return [alpha[199] + K[199] * alpha[0]]

def constraintU200(alpha):
    return [K[200] * alpha[0] - alpha[200]]
def constraintL200(alpha):
    return [alpha[200] + K[200] * alpha[0]]

def constraintU201(alpha):
    return [K[201] * alpha[0] - alpha[201]]
def constraintL201(alpha):
    return [alpha[201] + K[201] * alpha[0]]

def constraintU202(alpha):
    return [K[202] * alpha[0] - alpha[202]]
def constraintL202(alpha):
    return [alpha[202] + K[202] * alpha[0]]

def constraintU203(alpha):
    return [K[203] * alpha[0] - alpha[203]]
def constraintL203(alpha):
    return [alpha[203] + K[203] * alpha[0]]

def constraintU204(alpha):
    return [K[204] * alpha[0] - alpha[204]]
def constraintL204(alpha):
    return [alpha[204] + K[204] * alpha[0]]

def constraintU205(alpha):
    return [K[205] * alpha[0] - alpha[205]]
def constraintL205(alpha):
    return [alpha[205] + K[205] * alpha[0]]

def constraintU206(alpha):
    return [K[206] * alpha[0] - alpha[206]]
def constraintL206(alpha):
    return [alpha[206] + K[206] * alpha[0]]

def constraintU207(alpha):
    return [K[207] * alpha[0] - alpha[207]]
def constraintL207(alpha):
    return [alpha[207] + K[207] * alpha[0]]

def constraintU208(alpha):
    return [K[208] * alpha[0] - alpha[208]]
def constraintL208(alpha):
    return [alpha[208] + K[208] * alpha[0]]

def constraintU209(alpha):
    return [K[209] * alpha[0] - alpha[209]]
def constraintL209(alpha):
    return [alpha[209] + K[209] * alpha[0]]

def constraintU210(alpha):
    return [K[210] * alpha[0] - alpha[210]]
def constraintL210(alpha):
    return [alpha[210] + K[210] * alpha[0]]

def constraintU211(alpha):
    return [K[211] * alpha[0] - alpha[211]]
def constraintL211(alpha):
    return [alpha[211] + K[211] * alpha[0]]

def constraintU212(alpha):
    return [K[212] * alpha[0] - alpha[212]]
def constraintL212(alpha):
    return [alpha[212] + K[212] * alpha[0]]

def constraintU213(alpha):
    return [K[213] * alpha[0] - alpha[213]]
def constraintL213(alpha):
    return [alpha[213] + K[213] * alpha[0]]

def constraintU214(alpha):
    return [K[214] * alpha[0] - alpha[214]]
def constraintL214(alpha):
    return [alpha[214] + K[214] * alpha[0]]

def constraintU215(alpha):
    return [K[215] * alpha[0] - alpha[215]]
def constraintL215(alpha):
    return [alpha[215] + K[215] * alpha[0]]

def constraintU216(alpha):
    return [K[216] * alpha[0] - alpha[216]]
def constraintL216(alpha):
    return [alpha[216] + K[216] * alpha[0]]

def constraintU217(alpha):
    return [K[217] * alpha[0] - alpha[217]]
def constraintL217(alpha):
    return [alpha[217] + K[217] * alpha[0]]

def constraintU218(alpha):
    return [K[218] * alpha[0] - alpha[218]]
def constraintL218(alpha):
    return [alpha[218] + K[218] * alpha[0]]

def constraintU219(alpha):
    return [K[219] * alpha[0] - alpha[219]]
def constraintL219(alpha):
    return [alpha[219] + K[219] * alpha[0]]

def constraintU220(alpha):
    return [K[220] * alpha[0] - alpha[220]]
def constraintL220(alpha):
    return [alpha[220] + K[220] * alpha[0]]

def constraintU221(alpha):
    return [K[221] * alpha[0] - alpha[221]]
def constraintL221(alpha):
    return [alpha[221] + K[221] * alpha[0]]

def constraintU222(alpha):
    return [K[222] * alpha[0] - alpha[222]]
def constraintL222(alpha):
    return [alpha[222] + K[222] * alpha[0]]

def constraintU223(alpha):
    return [K[223] * alpha[0] - alpha[223]]
def constraintL223(alpha):
    return [alpha[223] + K[223] * alpha[0]]

def constraintU224(alpha):
    return [K[224] * alpha[0] - alpha[224]]
def constraintL224(alpha):
    return [alpha[224] + K[224] * alpha[0]]

def constraintU225(alpha):
    return [K[225] * alpha[0] - alpha[225]]
def constraintL225(alpha):
    return [alpha[225] + K[225] * alpha[0]]

def constraintU226(alpha):
    return [K[226] * alpha[0] - alpha[226]]
def constraintL226(alpha):
    return [alpha[226] + K[226] * alpha[0]]

def constraintU227(alpha):
    return [K[227] * alpha[0] - alpha[227]]
def constraintL227(alpha):
    return [alpha[227] + K[227] * alpha[0]]

def constraintU228(alpha):
    return [K[228] * alpha[0] - alpha[228]]
def constraintL228(alpha):
    return [alpha[228] + K[228] * alpha[0]]

def constraintU229(alpha):
    return [K[229] * alpha[0] - alpha[229]]
def constraintL229(alpha):
    return [alpha[229] + K[229] * alpha[0]]

def constraintU230(alpha):
    return [K[230] * alpha[0] - alpha[230]]
def constraintL230(alpha):
    return [alpha[230] + K[230] * alpha[0]]

def constraintU231(alpha):
    return [K[231] * alpha[0] - alpha[231]]
def constraintL231(alpha):
    return [alpha[231] + K[231] * alpha[0]]

def constraintU232(alpha):
    return [K[232] * alpha[0] - alpha[232]]
def constraintL232(alpha):
    return [alpha[232] + K[232] * alpha[0]]

def constraintU233(alpha):
    return [K[233] * alpha[0] - alpha[233]]
def constraintL233(alpha):
    return [alpha[233] + K[233] * alpha[0]]

def constraintU234(alpha):
    return [K[234] * alpha[0] - alpha[234]]
def constraintL234(alpha):
    return [alpha[234] + K[234] * alpha[0]]

def constraintU235(alpha):
    return [K[235] * alpha[0] - alpha[235]]
def constraintL235(alpha):
    return [alpha[235] + K[235] * alpha[0]]

def constraintU236(alpha):
    return [K[236] * alpha[0] - alpha[236]]
def constraintL236(alpha):
    return [alpha[236] + K[236] * alpha[0]]

def constraintU237(alpha):
    return [K[237] * alpha[0] - alpha[237]]
def constraintL237(alpha):
    return [alpha[237] + K[237] * alpha[0]]

def constraintU238(alpha):
    return [K[238] * alpha[0] - alpha[238]]
def constraintL238(alpha):
    return [alpha[238] + K[238] * alpha[0]]

def constraintU239(alpha):
    return [K[239] * alpha[0] - alpha[239]]
def constraintL239(alpha):
    return [alpha[239] + K[239] * alpha[0]]

def constraintU240(alpha):
    return [K[240] * alpha[0] - alpha[240]]
def constraintL240(alpha):
    return [alpha[240] + K[240] * alpha[0]]

def constraintU241(alpha):
    return [K[241] * alpha[0] - alpha[241]]
def constraintL241(alpha):
    return [alpha[241] + K[241] * alpha[0]]

def constraintU242(alpha):
    return [K[242] * alpha[0] - alpha[242]]
def constraintL242(alpha):
    return [alpha[242] + K[242] * alpha[0]]

def constraintU243(alpha):
    return [K[243] * alpha[0] - alpha[243]]
def constraintL243(alpha):
    return [alpha[243] + K[243] * alpha[0]]

def constraintU244(alpha):
    return [K[244] * alpha[0] - alpha[244]]
def constraintL244(alpha):
    return [alpha[244] + K[244] * alpha[0]]

def constraintU245(alpha):
    return [K[245] * alpha[0] - alpha[245]]
def constraintL245(alpha):
    return [alpha[245] + K[245] * alpha[0]]

def constraintU246(alpha):
    return [K[246] * alpha[0] - alpha[246]]
def constraintL246(alpha):
    return [alpha[246] + K[246] * alpha[0]]

def constraintU247(alpha):
    return [K[247] * alpha[0] - alpha[247]]
def constraintL247(alpha):
    return [alpha[247] + K[247] * alpha[0]]

def constraintU248(alpha):
    return [K[248] * alpha[0] - alpha[248]]
def constraintL248(alpha):
    return [alpha[248] + K[248] * alpha[0]]

def constraintU249(alpha):
    return [K[249] * alpha[0] - alpha[249]]
def constraintL249(alpha):
    return [alpha[249] + K[249] * alpha[0]]

def constraintU250(alpha):
    return [K[250] * alpha[0] - alpha[250]]
def constraintL250(alpha):
    return [alpha[250] + K[250] * alpha[0]]

def constraintU251(alpha):
    return [K[251] * alpha[0] - alpha[251]]
def constraintL251(alpha):
    return [alpha[251] + K[251] * alpha[0]]

def constraintU252(alpha):
    return [K[252] * alpha[0] - alpha[252]]
def constraintL252(alpha):
    return [alpha[252] + K[252] * alpha[0]]

def constraintU253(alpha):
    return [K[253] * alpha[0] - alpha[253]]
def constraintL253(alpha):
    return [alpha[253] + K[253] * alpha[0]]

def constraintU254(alpha):
    return [K[254] * alpha[0] - alpha[254]]
def constraintL254(alpha):
    return [alpha[254] + K[254] * alpha[0]]

def constraintU255(alpha):
    return [K[255] * alpha[0] - alpha[255]]
def constraintL255(alpha):
    return [alpha[255] + K[255] * alpha[0]]

###########################

###########################
def constraintEig0(eig):
    return eig[0]

def constraintEig1(eig):
    return eig[1]

def constraintEig2(eig):
    return eig[2]

def constraintEig3(eig):
    return eig[3]

def constraintEig4(eig):
    return eig[4]

def constraintEig5(eig):
    return eig[5]

def constraintEig6(eig):
    return eig[6]

def constraintEig7(eig):
    return eig[7]

def constraintEig8(eig):
    return eig[8]

def constraintEig9(eig):
    return eig[9]

def constraintEig10(eig):
    return eig[10]

def constraintEig11(eig):
    return eig[11]

def constraintEig12(eig):
    return eig[12]

def constraintEig13(eig):
    return eig[13]

def constraintEig14(eig):
    return eig[14]

def constraintEig15(eig):
    return eig[15]

def constraintEig16(eig):
    return eig[16]

def constraintEig17(eig):
    return eig[17]

def constraintEig18(eig):
    return eig[18]

def constraintEig19(eig):
    return eig[19]

def constraintEig20(eig):
    return eig[20]

def constraintEig21(eig):
    return eig[21]

def constraintEig22(eig):
    return eig[22]

def constraintEig23(eig):
    return eig[23]

def constraintEig24(eig):
    return eig[24]

def constraintEig25(eig):
    return eig[25]

def constraintEig26(eig):
    return eig[26]

def constraintEig27(eig):
    return eig[27]

def constraintEig28(eig):
    return eig[28]

def constraintEig29(eig):
    return eig[29]

def constraintEig30(eig):
    return eig[30]

def constraintEig31(eig):
    return eig[31]

def constraintEig32(eig):
    return eig[32]

def constraintEig33(eig):
    return eig[33]

def constraintEig34(eig):
    return eig[34]

def constraintEig35(eig):
    return eig[35]

def constraintEig36(eig):
    return eig[36]

def constraintEig37(eig):
    return eig[37]

def constraintEig38(eig):
    return eig[38]

def constraintEig39(eig):
    return eig[39]

def constraintEig40(eig):
    return eig[40]

def constraintEig41(eig):
    return eig[41]

def constraintEig42(eig):
    return eig[42]

def constraintEig43(eig):
    return eig[43]

def constraintEig44(eig):
    return eig[44]

def constraintEig45(eig):
    return eig[45]

def constraintEig46(eig):
    return eig[46]

def constraintEig47(eig):
    return eig[47]

def constraintEig48(eig):
    return eig[48]

def constraintEig49(eig):
    return eig[49]

def constraintEig50(eig):
    return eig[50]

def constraintEig51(eig):
    return eig[51]

def constraintEig52(eig):
    return eig[52]

def constraintEig53(eig):
    return eig[53]

def constraintEig54(eig):
    return eig[54]

def constraintEig55(eig):
    return eig[55]

def constraintEig56(eig):
    return eig[56]

def constraintEig57(eig):
    return eig[57]

def constraintEig58(eig):
    return eig[58]

def constraintEig59(eig):
    return eig[59]

def constraintEig60(eig):
    return eig[60]

def constraintEig61(eig):
    return eig[61]

def constraintEig62(eig):
    return eig[62]

def constraintEig63(eig):
    return eig[63]

def constraintEig64(eig):
    return eig[64]

def constraintEig65(eig):
    return eig[65]

def constraintEig66(eig):
    return eig[66]

def constraintEig67(eig):
    return eig[67]

def constraintEig68(eig):
    return eig[68]

def constraintEig69(eig):
    return eig[69]

def constraintEig70(eig):
    return eig[70]

def constraintEig71(eig):
    return eig[71]

def constraintEig72(eig):
    return eig[72]

def constraintEig73(eig):
    return eig[73]

def constraintEig74(eig):
    return eig[74]

def constraintEig75(eig):
    return eig[75]

def constraintEig76(eig):
    return eig[76]

def constraintEig77(eig):
    return eig[77]

def constraintEig78(eig):
    return eig[78]

def constraintEig79(eig):
    return eig[79]

def constraintEig80(eig):
    return eig[80]

def constraintEig81(eig):
    return eig[81]

def constraintEig82(eig):
    return eig[82]

def constraintEig83(eig):
    return eig[83]

def constraintEig84(eig):
    return eig[84]

def constraintEig85(eig):
    return eig[85]

def constraintEig86(eig):
    return eig[86]

def constraintEig87(eig):
    return eig[87]

def constraintEig88(eig):
    return eig[88]

def constraintEig89(eig):
    return eig[89]

def constraintEig90(eig):
    return eig[90]

def constraintEig91(eig):
    return eig[91]

def constraintEig92(eig):
    return eig[92]

def constraintEig93(eig):
    return eig[93]

def constraintEig94(eig):
    return eig[94]

def constraintEig95(eig):
    return eig[95]

def constraintEig96(eig):
    return eig[96]

def constraintEig97(eig):
    return eig[97]

def constraintEig98(eig):
    return eig[98]

def constraintEig99(eig):
    return eig[99]

def constraintEig100(eig):
    return eig[100]

def constraintEig101(eig):
    return eig[101]

def constraintEig102(eig):
    return eig[102]

def constraintEig103(eig):
    return eig[103]

def constraintEig104(eig):
    return eig[104]

def constraintEig105(eig):
    return eig[105]

def constraintEig106(eig):
    return eig[106]

def constraintEig107(eig):
    return eig[107]

def constraintEig108(eig):
    return eig[108]

def constraintEig109(eig):
    return eig[109]

def constraintEig110(eig):
    return eig[110]

def constraintEig111(eig):
    return eig[111]

def constraintEig112(eig):
    return eig[112]

def constraintEig113(eig):
    return eig[113]

def constraintEig114(eig):
    return eig[114]

def constraintEig115(eig):
    return eig[115]

def constraintEig116(eig):
    return eig[116]

def constraintEig117(eig):
    return eig[117]

def constraintEig118(eig):
    return eig[118]

def constraintEig119(eig):
    return eig[119]

def constraintEig120(eig):
    return eig[120]

def constraintEig121(eig):
    return eig[121]

def constraintEig122(eig):
    return eig[122]

def constraintEig123(eig):
    return eig[123]

def constraintEig124(eig):
    return eig[124]

def constraintEig125(eig):
    return eig[125]

def constraintEig126(eig):
    return eig[126]

def constraintEig127(eig):
    return eig[127]

def constraintEig128(eig):
    return eig[128]

def constraintEig129(eig):
    return eig[129]

def constraintEig130(eig):
    return eig[130]

def constraintEig131(eig):
    return eig[131]

def constraintEig132(eig):
    return eig[132]

def constraintEig133(eig):
    return eig[133]

def constraintEig134(eig):
    return eig[134]

def constraintEig135(eig):
    return eig[135]

def constraintEig136(eig):
    return eig[136]

def constraintEig137(eig):
    return eig[137]

def constraintEig138(eig):
    return eig[138]

def constraintEig139(eig):
    return eig[139]

def constraintEig140(eig):
    return eig[140]

def constraintEig141(eig):
    return eig[141]

def constraintEig142(eig):
    return eig[142]

def constraintEig143(eig):
    return eig[143]

def constraintEig144(eig):
    return eig[144]

def constraintEig145(eig):
    return eig[145]

def constraintEig146(eig):
    return eig[146]

def constraintEig147(eig):
    return eig[147]

def constraintEig148(eig):
    return eig[148]

def constraintEig149(eig):
    return eig[149]

def constraintEig150(eig):
    return eig[150]

def constraintEig151(eig):
    return eig[151]

def constraintEig152(eig):
    return eig[152]

def constraintEig153(eig):
    return eig[153]

def constraintEig154(eig):
    return eig[154]

def constraintEig155(eig):
    return eig[155]

def constraintEig156(eig):
    return eig[156]

def constraintEig157(eig):
    return eig[157]

def constraintEig158(eig):
    return eig[158]

def constraintEig159(eig):
    return eig[159]

def constraintEig160(eig):
    return eig[160]

def constraintEig161(eig):
    return eig[161]

def constraintEig162(eig):
    return eig[162]

def constraintEig163(eig):
    return eig[163]

def constraintEig164(eig):
    return eig[164]

def constraintEig165(eig):
    return eig[165]

def constraintEig166(eig):
    return eig[166]

def constraintEig167(eig):
    return eig[167]

def constraintEig168(eig):
    return eig[168]

def constraintEig169(eig):
    return eig[169]

def constraintEig170(eig):
    return eig[170]

def constraintEig171(eig):
    return eig[171]

def constraintEig172(eig):
    return eig[172]

def constraintEig173(eig):
    return eig[173]

def constraintEig174(eig):
    return eig[174]

def constraintEig175(eig):
    return eig[175]

def constraintEig176(eig):
    return eig[176]

def constraintEig177(eig):
    return eig[177]

def constraintEig178(eig):
    return eig[178]

def constraintEig179(eig):
    return eig[179]

def constraintEig180(eig):
    return eig[180]

def constraintEig181(eig):
    return eig[181]

def constraintEig182(eig):
    return eig[182]

def constraintEig183(eig):
    return eig[183]

def constraintEig184(eig):
    return eig[184]

def constraintEig185(eig):
    return eig[185]

def constraintEig186(eig):
    return eig[186]

def constraintEig187(eig):
    return eig[187]

def constraintEig188(eig):
    return eig[188]

def constraintEig189(eig):
    return eig[189]

def constraintEig190(eig):
    return eig[190]

def constraintEig191(eig):
    return eig[191]

def constraintEig192(eig):
    return eig[192]

def constraintEig193(eig):
    return eig[193]

def constraintEig194(eig):
    return eig[194]

def constraintEig195(eig):
    return eig[195]

def constraintEig196(eig):
    return eig[196]

def constraintEig197(eig):
    return eig[197]

def constraintEig198(eig):
    return eig[198]

def constraintEig199(eig):
    return eig[199]

def constraintEig200(eig):
    return eig[200]

def constraintEig201(eig):
    return eig[201]

def constraintEig202(eig):
    return eig[202]

def constraintEig203(eig):
    return eig[203]

def constraintEig204(eig):
    return eig[204]

def constraintEig205(eig):
    return eig[205]

def constraintEig206(eig):
    return eig[206]

def constraintEig207(eig):
    return eig[207]

def constraintEig208(eig):
    return eig[208]

def constraintEig209(eig):
    return eig[209]

def constraintEig210(eig):
    return eig[210]

def constraintEig211(eig):
    return eig[211]

def constraintEig212(eig):
    return eig[212]

def constraintEig213(eig):
    return eig[213]

def constraintEig214(eig):
    return eig[214]

def constraintEig215(eig):
    return eig[215]

def constraintEig216(eig):
    return eig[216]

def constraintEig217(eig):
    return eig[217]

def constraintEig218(eig):
    return eig[218]

def constraintEig219(eig):
    return eig[219]

def constraintEig220(eig):
    return eig[220]

def constraintEig221(eig):
    return eig[221]

def constraintEig222(eig):
    return eig[222]

def constraintEig223(eig):
    return eig[223]

def constraintEig224(eig):
    return eig[224]

def constraintEig225(eig):
    return eig[225]

def constraintEig226(eig):
    return eig[226]

def constraintEig227(eig):
    return eig[227]

def constraintEig228(eig):
    return eig[228]

def constraintEig229(eig):
    return eig[229]

def constraintEig230(eig):
    return eig[230]

def constraintEig231(eig):
    return eig[231]

def constraintEig232(eig):
    return eig[232]

def constraintEig233(eig):
    return eig[233]

def constraintEig234(eig):
    return eig[234]

def constraintEig235(eig):
    return eig[235]

def constraintEig236(eig):
    return eig[236]

def constraintEig237(eig):
    return eig[237]

def constraintEig238(eig):
    return eig[238]

def constraintEig239(eig):
    return eig[239]

def constraintEig240(eig):
    return eig[240]

def constraintEig241(eig):
    return eig[241]

def constraintEig242(eig):
    return eig[242]

def constraintEig243(eig):
    return eig[243]

def constraintEig244(eig):
    return eig[244]

def constraintEig245(eig):
    return eig[245]

def constraintEig246(eig):
    return eig[246]

def constraintEig247(eig):
    return eig[247]

def constraintEig248(eig):
    return eig[248]

def constraintEig249(eig):
    return eig[249]

def constraintEig250(eig):
    return eig[250]

def constraintEig251(eig):
    return eig[251]

def constraintEig252(eig):
    return eig[252]

def constraintEig253(eig):
    return eig[253]

def constraintEig254(eig):
    return eig[254]

def constraintEig255(eig):
    return eig[255]
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

def f_eig(eig):
    return - np.sum(np.log(eig)) + np.trace(U_Toeplitz @ np.diag(eig) @ U_Toeplitz.T @ sCov)

def f_eig2(eig):
    return - np.sum(np.log(eig)) + np.trace(U_Toeplitz2 @ np.diag(eig) @ U_Toeplitz2.T @ sCov)

constraints = generating_constraints(N)
constraints_Eig = generating_constraints_eig(N)

# MODEL
#AUTOREGRESSIVES MODEL GAUS
r = 0.4
C = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        C[i,j] = r**(np.abs(j-i))

#Brownian Motion (see shrinkage estimator original paper)
#H = 0.8
#C = np.zeros((N,N))
#for i in range(N):
#    for j in range(N):
#        C[i,j] = 0.5 * ( (np.abs(j-i) + 1)**(2*H) - 2 * np.abs(j-i)**(2*H) + np.abs((np.abs(j-i) - 1))**(2*H) )




#N_SAMPLES = [4,8,10,16,32,64,100]
N_SAMPLES = [4,8,16,32,64,128]
#N_SAMPLES = [16]
RUNS = 15
print(f'r {r}, N: {N}, RUNS: {RUNS}')
MSE_sCov_n = []
MSE_toeplitz_n = []
MSE_OAS_n = []
MSE_toeplitz2_n = []

MSE_SVD_sCov_n = []
MSE_SVD_toeplitz_n = []
MSE_SVD_OAS_n = []
MSE_SVD_toeplitz2_n = []

MSE_toeplitz_eig_n = []
MSE_toeplitz2_eig_n = []
MSE_toeplitz3_eig_n = []


n_outliers = 0
skipping = False

#SVD echte Cov

U,S,VH = np.linalg.svd(C)
tot_e = np.sum(S**2)
boundary = 0.9 * tot_e
n_eig = 0
e = 0
for i in range(N):
    if e < boundary:
        n_eig += 1
        e += S[i]**2
    else:
        n_eig -= 1
        e -= S[i-1]**2
        break


for n_samples in N_SAMPLES:

    MSE_sCov = []
    MSE_toeplitz = []
    MSE_OAS = []
    MSE_toeplitz2 = []

    MSE_SVD_sCov = []
    MSE_SVD_toeplitz = []
    MSE_SVD_OAS = []
    MSE_SVD_toeplitz2 = []

    MSE_toeplitz_eig = []
    MSE_toeplitz2_eig = []
    MSE_toeplitz3_eig = []

    for run in range(RUNS):
        K = K_dic[str(N)] * np.ones(N)
        if run%4 == 0:
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

        if np.sum((np.linalg.inv(Gamma_est) - C)**2) > 1000:
            n_outliers += 1
            skipping = True
            w,v = np.linalg.eigh(Gamma_est)
            print(w)
            print(run)
            print(np.linalg.det(np.linalg.inv(Gamma_est)))
            print(np.linalg.det(Gamma_est))
            print(result.x)

        if skipping == False:
            MSE_sCov.append(np.sum((sCov - C) ** 2))
            MSE_toeplitz.append(np.sum((np.linalg.inv(Gamma_est) - C) ** 2))
            MSE_OAS.append(np.sum((OAS_C - C)**2))

            U_sCov,_,_ = np.linalg.svd(sCov)
            U_toeplitz,S_toeplitz,_ = np.linalg.svd(Gamma_est)
            U_OAS,_,_ = np.linalg.svd(OAS_C)

            U_Toeplitz = U_toeplitz

            mse_scov = []
            mse_toeplitz = []
            mse_oas = []
            for eig in range(int(n_eig)):

                mse_scov.append(np.min([np.sum(np.abs(-U[:,eig][:,None] - U_sCov)**2,axis=0),np.sum(np.abs(U[:,eig][:,None] - U_sCov)**2,axis=0)]))
                mse_toeplitz.append(np.min([np.sum(np.abs(-U[:, eig][:,None] - U_toeplitz) ** 2,axis=0),np.sum(np.abs(U[:, eig][:,None] - U_toeplitz)**2,axis=0)]))
                mse_oas.append(np.min([np.sum(np.abs(-U[:, eig][:,None] - U_OAS) ** 2,axis=0),np.sum(np.abs(U[:, eig][:,None] - U_OAS) ** 2,axis=0)]))
                arg_sCov = np.argmin([np.sum(np.abs(-U[:,eig][:,None] - U_sCov)**2,axis=0),np.sum(np.abs(U[:,eig][:,None] - U_sCov)**2,axis=0)]) % (N-eig)
                arg_toep = np.argmin([np.sum(np.abs(-U[:, eig][:,None] - U_toeplitz) ** 2,axis=0),np.sum(np.abs(U[:, eig][:,None] - U_toeplitz)**2,axis=0)]) % (N-eig)
                arg_oas = np.argmin([np.sum(np.abs(-U[:, eig][:,None] - U_OAS) ** 2,axis=0),np.sum(np.abs(U[:, eig][:,None] - U_OAS) ** 2,axis=0)]) % (N-eig)

                U_sCov = np.delete(U_sCov,arg_sCov,1)
                U_toeplitz = np.delete(U_toeplitz, arg_toep, 1)
                U_OAS = np.delete(U_OAS, arg_oas, 1)

            mse_scov = np.mean(mse_scov)
            mse_toeplitz = np.mean(mse_toeplitz)
            mse_oas = np.mean(mse_oas)

            result_eig1 = optimize.minimize(f_eig, S_toeplitz, method="SLSQP", constraints=constraints_Eig)
            Gamma_est_eig = U_Toeplitz @ np.diag(result_eig1.x) @ U_Toeplitz.T

            MSE_toeplitz_eig.append(np.sum((np.linalg.inv(Gamma_est_eig) - C) ** 2))

        #print(f'MSE  Toeplitz: {np.sum((np.linalg.inv(Gamma_est) - C)**2)}')


        if IMPROVING & (skipping == False):
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
            U_toeplitz2, S_toeplitz2, _ = np.linalg.svd(Gamma_est2)
            U_Toeplitz2 = U_toeplitz2
            mse_toeplitz2 = []
            for eig in range(int(n_eig)):
                mse_toeplitz2.append(np.min([np.min(np.sum(np.abs(-U[:, eig][:,None] - U_toeplitz2) ** 2,axis=0)),np.min(np.sum(np.abs(U[:, eig][:,None] - U_toeplitz2)**2,axis=0))]))
                arg_toep2 = np.argmin([np.sum(np.abs(-U[:, eig][:, None] - U_toeplitz2) ** 2, axis=0),np.sum(np.abs(U[:, eig][:, None] - U_toeplitz2) ** 2, axis=0)]) % (N - eig)
                U_toeplitz2 = np.delete(U_toeplitz2, arg_toep2, 1)
            mse_toeplitz2 = np.mean(mse_toeplitz2)

            result_eig2 = optimize.minimize(f_eig2, S_toeplitz2, method="SLSQP", constraints=constraints_Eig)
            Gamma_est_eig2 = U_Toeplitz2 @ np.diag(result_eig2.x) @ U_Toeplitz2.T

            MSE_toeplitz2_eig.append(np.sum((np.linalg.inv(Gamma_est_eig2) - C) ** 2))

            #result_eig3 = optimize.minimize(f_eig, S_toeplitz2, method="SLSQP", constraints=constraints_Eig)
            #Gamma_est_eig3 = U_Toeplitz2 @ np.diag(result_eig3.x) @ U_Toeplitz2.T

            #MSE_toeplitz3_eig.append(np.sum((np.linalg.inv(Gamma_est_eig3) - C) ** 2))

            MSE_SVD_sCov.append(mse_scov)
            MSE_SVD_toeplitz.append(mse_toeplitz)
            MSE_SVD_OAS.append(mse_oas)
            MSE_SVD_toeplitz2.append(mse_toeplitz2)

            MSE_toeplitz2.append(np.sum((np.linalg.inv(Gamma_est2) - C) ** 2))
            MSE_toeplitz2_eig.append(np.sum((np.linalg.inv(Gamma_est_eig2) - C) ** 2))
            #MSE_toeplitz3_eig.append(np.sum((np.linalg.inv(Gamma_est_eig3) - C) ** 2))



        skipping = False
    print(f'MSE of sCov and real Cov: {np.mean(MSE_sCov):.4f}')
    print(f'MSE of Toep and real Cov: {np.mean(MSE_toeplitz):.4f}')
    print(f'MSE of Toep2 and real Cov: {np.mean(MSE_toeplitz2):.4f}')
    print(f'MSE of OAS and real Cov: {np.mean(MSE_OAS):.4f}')
    print(f'MSE of ToepEig and real Cov: {np.mean(MSE_toeplitz_eig):.4f}')
    print(f'MSE of ToepEig2 and real Cov: {np.mean(MSE_toeplitz2_eig):.4f}')
    #print(f'MSE of ToepEig3 and real Cov: {np.mean(MSE_toeplitz3_eig):.4f}')

    print(f'\nMSE of SVD sCov and real Cov: {np.mean(MSE_SVD_sCov):.4f}')
    print(f'MSE of SVD Toep and real Cov: {np.mean(MSE_SVD_toeplitz):.4f}')
    print(f'MSE of SVD Toep2 and real Cov: {np.mean(MSE_SVD_toeplitz2):.4f}')
    print(f'MSE of SVD OAS and real Cov: {np.mean(MSE_SVD_OAS):.4f}')
    print(f'Outliers: {n_outliers}')

    MSE_sCov_n.append(np.mean(MSE_sCov))
    MSE_toeplitz_n.append(np.mean(MSE_toeplitz))
    MSE_OAS_n.append(np.mean(MSE_OAS))
    MSE_toeplitz2_n.append(np.mean(MSE_toeplitz2))
    MSE_toeplitz_eig_n.append(np.mean(MSE_toeplitz_eig))
    MSE_toeplitz2_eig_n.append(np.mean(MSE_toeplitz2_eig))
    #MSE_toeplitz3_eig_n.append(np.mean(MSE_toeplitz3_eig))

    MSE_SVD_sCov_n.append(np.mean(MSE_SVD_sCov))
    MSE_SVD_toeplitz_n.append(np.mean(MSE_SVD_toeplitz))
    MSE_SVD_OAS_n.append(np.mean(MSE_SVD_OAS))
    MSE_SVD_toeplitz2_n.append(np.mean(MSE_SVD_toeplitz2))

MSE_sCov_n = np.array(MSE_sCov_n)
MSE_toeplitz_n = np.array(MSE_toeplitz_n)
MSE_OAS_n = np.array(MSE_OAS_n)
MSE_toeplitz2_n = np.array(MSE_toeplitz2_n)
MSE_toeplitz_eig_n = np.array(MSE_toeplitz_eig_n)
MSE_toeplitz2_eig_n = np.array(MSE_toeplitz2_eig_n)
#MSE_toeplitz3_eig_n = np.array(MSE_toeplitz3_eig_n)

MSE_SVD_sCov_n = np.array(MSE_SVD_sCov_n)
MSE_SVD_toeplitz_n = np.array(MSE_SVD_toeplitz_n)
MSE_SVD_OAS_n = np.array(MSE_SVD_OAS_n)
MSE_SVD_toeplitz2_n = np.array(MSE_SVD_toeplitz2_n)

csv_writer.writerow(N_SAMPLES)
csv_writer.writerow(MSE_SVD_sCov_n)
csv_writer.writerow(MSE_SVD_OAS_n)
csv_writer.writerow(MSE_SVD_toeplitz_n)
csv_writer.writerow(MSE_SVD_toeplitz2_n)
csv_writer.writerow(MSE_OAS_n)
csv_writer.writerow(MSE_toeplitz_n)
csv_writer.writerow(MSE_toeplitz2_n)
csv_writer.writerow(MSE_toeplitz_eig_n)
csv_writer.writerow(MSE_toeplitz2_eig_n)
csv_file.close()

# plt.plot(N_SAMPLES,MSE_sCov_n,label = 'sCov')
# plt.plot(N_SAMPLES,MSE_toeplitz_n,label = 'Toeplitz')
# plt.plot(N_SAMPLES,MSE_OAS_n,label = 'OAS')
# plt.plot(N_SAMPLES,MSE_toeplitz2_n,label = 'Toeplitz2')
# plt.plot(N_SAMPLES,MSE_toeplitz_eig_n,label = 'ToepEig')
# plt.plot(N_SAMPLES,MSE_toeplitz2_eig_n,label = 'ToepEig2')
# plt.legend()
# plt.ylabel('MSE')
# plt.xlabel('N_SAMPLES')
# #plt.title(f'Dimension: {N}, r-value: {r}')
# plt.show()
#
# plt.plot(N_SAMPLES,MSE_SVD_sCov_n,label = 'sCov')
# plt.plot(N_SAMPLES,MSE_SVD_toeplitz_n,label = 'Toeplitz')
# plt.plot(N_SAMPLES,MSE_SVD_OAS_n,label = 'OAS')
# plt.plot(N_SAMPLES,MSE_SVD_toeplitz2_n,label = 'Toeplitz2')
# plt.legend()
# plt.ylabel('MSE SVD')
# plt.xlabel('N_SAMPLES')
# #plt.title(f'Dimension: {N}, r-value: {r}')
# plt.show()