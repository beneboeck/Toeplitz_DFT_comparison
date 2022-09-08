import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from ComputingTheKBound import *

IMPROVING = True

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

N = 100
K = K_dic[str(N)] * np.ones(N)

rand_matrix = np.random.randn(N, N)
B_mask = np.tril(rand_matrix)
B_mask[B_mask != 0] = 1
C_mask = np.tril(rand_matrix, k=-1)
C_mask[C_mask != 0] = 1

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

def adjustingK(K,result):
    alpha_0 = result.x[0]
    idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
    idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)


    len_bounding = len(idx_bounding)
    len_interior = len(idx_interior)
    derivatives = optimize.approx_fprime(result.x, f, epsilon=10e-8)
    while (len_interior > 0) & (len_bounding > 0):
        max_bounding_idx = idx_bounding[np.argmax(np.abs(derivatives[idx_bounding]))]
        max_interior_idx = idx_interior[np.argmax(K[idx_interior] * alpha_0 - np.abs(result.x[idx_interior]))]

        K[max_interior_idx] = (0.7 * (K[max_interior_idx] * result.x[0] - np.abs(result.x[max_interior_idx])) + np.abs(result.x[max_interior_idx])) / result.x[0]
        K_test = K
        while bound(K_test) < 1:
            K_test[max_bounding_idx] = 1.01 * K_test[max_bounding_idx]
        K_test[max_bounding_idx] = 1 / 1.01 * K_test[max_bounding_idx]
        K = K_test

        idx_bounding = np.delete(idx_bounding,np.where(idx_bounding == max_bounding_idx))
        idx_interior = np.delete(idx_interior,np.where(idx_interior ==  max_interior_idx))
        len_bounding = len(idx_bounding)
        len_interior = len(idx_interior)

    return K


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


constraints=[c0_dic,cL1_dic,cU1_dic,cL2_dic,cU2_dic,cL3_dic,cU3_dic,cL4_dic,cU4_dic,cL5_dic,cU5_dic,cL6_dic,cU6_dic,cL7_dic,cU7_dic,cL8_dic,cU8_dic,cL9_dic,cU9_dic,cL10_dic,cU10_dic,cL11_dic,cU11_dic,cL12_dic,cU12_dic,cL13_dic,cU13_dic,cL14_dic,cU14_dic,cL15_dic,cU15_dic,cL16_dic,cU16_dic,cL17_dic,cU17_dic,cL18_dic,cU18_dic,cL19_dic,cU19_dic,cL20_dic,cU20_dic,cL21_dic,cU21_dic,cL22_dic,cU22_dic,cL23_dic,cU23_dic,cL24_dic,cU24_dic,cL25_dic,cU25_dic,cL26_dic,cU26_dic,cL27_dic,cU27_dic,cL28_dic,cU28_dic,cL29_dic,cU29_dic,cL30_dic,cU30_dic,cL31_dic,cU31_dic,cL32_dic,cU32_dic]
constraints2 = [cL33_dic,cU33_dic,cL34_dic,cU34_dic,cL35_dic,cU35_dic,cL36_dic,cU36_dic,cL37_dic,cU37_dic,cL38_dic,cU38_dic,cL39_dic,cU39_dic,cL40_dic,cU40_dic,cL41_dic,cU41_dic,cL42_dic,cU42_dic,cL43_dic,cU43_dic,cL44_dic,cU44_dic,cL45_dic,cU45_dic,cL46_dic,cU46_dic,cL47_dic,cU47_dic,cL48_dic,cU48_dic,cL49_dic,cU49_dic,cL50_dic,cU50_dic,cL51_dic,cU51_dic,cL52_dic,cU52_dic,cL53_dic,cU53_dic,cL54_dic,cU54_dic,cL55_dic,cU55_dic,cL56_dic,cU56_dic,cL57_dic,cU57_dic,cL58_dic,cU58_dic,cL59_dic,cU59_dic,cL60_dic,cU60_dic,cL61_dic,cU61_dic,cL62_dic,cU62_dic,cL63_dic,cU63_dic,cL64_dic,cU64_dic]
constraints.extend(constraints2)
constraints = constraints[:2*N - 1]
# MODEL
N_SAMPLES = [4,8,16,32,64]
RUNS = 10
#AUTOREGRESSIVES MODEL GAUS
r = 0.3
C = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        C[i,j] = r**(np.abs(j-i))

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
        rho = min(((1 - 2/N) * np.trace(sCov @ sCov) + np.trace(sCov)**2)/((n_samples + 1 - 2/N) * (np.trace(sCov @ sCov) - np.trace(sCov)/N)),1)
        OAS_C = (1 - rho) * sCov + rho * F

        # my method
        init_values = np.zeros(N)
        init_values[0] = np.random.uniform(low=1, high=20)
        for n in range(1,N):
            init_values[n] = np.random.uniform(low = - K[n] * init_values[0] + 0.01, high = K[n] * init_values[0] - 0.01 )
        result = optimize.minimize(f, init_values, method="SLSQP",constraints=constraints)
        Gamma_est = generating_Gamma(result.x)

        MSE_sCov.append(np.sum((sCov - C)**2))
        MSE_toeplitz.append(np.sum((np.linalg.inv(Gamma_est) - C)**2))
        if np.sum((np.linalg.inv(Gamma_est) - C)**2) > 100:
            print(run)
            print(np.linalg.det(np.linalg.inv(Gamma_est)))
            print(np.linalg.det(Gamma_est))
            print(result.x)
            raise ValueError
        MSE_OAS.append(np.sum((OAS_C - C)**2))
        #print(f'MSE  Toeplitz: {np.sum((np.linalg.inv(Gamma_est) - C)**2)}')


        if IMPROVING:
            #print('start')
            K = adjustingK(K, result)
            alpha_0 = result.x[0]
            idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
            idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
            len_bounding = len(idx_bounding)
            len_interior = len(idx_interior)
            derivatives = optimize.approx_fprime(result.x, f, epsilon=10e-8)
            while (len_interior > 0) & (len_bounding > 0):
                #print(result.x[0])
                result2 = optimize.minimize(f, result.x, method="SLSQP", constraints=constraints)
                #print('RESULTS')
                #print(result.fun)
                #print(result2.fun)
                result = result2
                K = adjustingK(K, result)
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

plt.plot(N_SAMPLES,MSE_sCov_n,label = 'sCov')
plt.plot(N_SAMPLES,MSE_toeplitz_n,label = 'Toeplitz')
plt.plot(N_SAMPLES,MSE_OAS_n,label = 'OAS')
plt.plot(N_SAMPLES,MSE_toeplitz2_n,label = 'OAS')
plt.legend()
plt.ylabel('MSE')
plt.xlabel('N_SAMPLES')
plt.title(f'Dimension: {N}, r-value: {r}')
plt.show()




# MSE_sCov = []
# MSE_toeplitz = []
# C = [[1,0.4,0.2,0.1],[0.4,1,0.4,0.2],[0.2,0.4,1,0.4],[0.1,0.2,0.4,1]]
# for k in range(1):
#     x_new = np.random.multivariate_normal([0,0,0,0],C,16)
#     sCov = 1/16 * (x_new.T @ x_new)
#     alpha_vec = []
#     for i in range(200):
#         alpha_0 = np.random.uniform(low=0.05,high=20)
#         alpha_1 = np.random.uniform(low = - K * alpha_0 + 0.01, high = K * alpha_0 - 0.01 )
#         alpha_2 = np.random.uniform(low=- K * alpha_0 + 0.01, high=K * alpha_0 - 0.01)
#         alpha_3 = np.random.uniform(low=- K * alpha_0 + 0.01, high=K * alpha_0 - 0.01)
#
#         result = optimize.minimize(f, np.array([alpha_0, alpha_1, alpha_2, alpha_3]), method="SLSQP",constraints=constraints)
#         alpha_vec.append(result.x)
#
#     print('MEAN')
#     print(np.sum(np.abs(np.array(alpha_vec) - np.mean(np.array(alpha_vec),axis=0))**2))
#     #result2 = optimize.minimize(f, np.array([3, 0.4, -0.9, 0.05]), method="SLSQP",constraints=[constraint1_dic, constraint2_dic, constraint3_dic, constraint4_dic, constraint5_dic, constraint6_dic, constraint7_dic])
#     #print(result)
#     #print(type(result))
#     #print(result.x)
#     Gamma_est = generating_Gamma(result.x)
#     #print(result.x[0])
#     #Gamma_est2 = generating_Gamma(result2.x)
#     #print(result2.x[0])
#     #print(Gamma_est - Gamma_est2)
#     #print('..')
#     #print(C)
#     #print(np.sum((np.linalg.inv(Gamma_est) - C)**2))
#     #print(np.sum((sCov - C)**2))
#     MSE_sCov.append(np.sum((sCov - C)**2))
#     MSE_toeplitz.append(np.sum((np.linalg.inv(Gamma_est) - C)**2))
#
# print(np.mean(MSE_sCov))
# print(np.mean(MSE_toeplitz))
# accumulator = []
#
# def f(x):
#     accumulator.append(x)
#     return (x[0] - 2)**2 + (x[1] - 3)**2
#
# def constraint1(x):
#     return [x[0]]
# def constraint2(x):
#     return [x[1]]
# def constraint3(x):
#     return [1 - x[0] - x[1]]
#
# constraint1_dic = {
#     'fun': constraint1,
#     'type': 'ineq',
# }
# constraint2_dic = {
#     'fun': constraint2,
#     'type': 'ineq',
# }
# constraint3_dic = {
#     'fun': constraint3,
#     'type': 'ineq',
# }
#
# result = optimize.minimize(f, np.array([0, 0]), method="SLSQP",
#                      constraints=[constraint1_dic,constraint2_dic,constraint3_dic])
# print(result)
# #print(accumulator)
#
# print('----------')
# C = np.array([[1,0.1],[0.1,1]])
# L,U = np.linalg.eigh(C)
# print(U @ np.diag(L) @ U.T)
# x = np.random.randn(10,2)
# x_new = np.einsum('ij,mj->mi',np.diag(np.sqrt(L)) @ U,x)
# sCov = 1/10 * np.mean(np.einsum('ij,ik->ijk',x_new,x_new),axis=0)
# print(sCov)
# def f(theta):
#     Gamma = np.array([[theta[0],theta[1]],[theta[2],theta[3]]])
#
#     return - 5 * np.log(np.linalg.det(Gamma)) + 5 * np.trace(Gamma @ sCov)
#
# result = optimize.minimize(f, np.array([1,0,0,1]), method="SLSQP")
# print(result)
# print(np.linalg.inv(sCov))