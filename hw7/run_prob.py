p_rh_cnotm = 0.9 * 0.6
print("P(R = 1, H = 1 | C = 1, M = 0) = P(R = 1 | C = 1, M = 0) * P(H = 1 | C = 1, M = 0) ")
print(" = ", p_rh_cnotm, '\n')
p_rh_notcnotm = 0.05 * 0.07
print("P(R = 1, H = 1 | C = 0, M = 0) = P(R = 1 | C = 0, M = 0) * P(H = 1 | C = 0, M = 0)")
print(" = ", p_rh_notcnotm,'\n')

p_m = 0.999 * 0.0001
print("P(M = 1) = (P(M = 1 | V = 1) * P(V = 1)) + (P(M = 1 | V = 0) * P(V = 0))")
print(" = ", p_m, '\n')

p_rh_notm = (p_rh_cnotm *0.05) + (p_rh_notcnotm*0.95) 
print("P(R = 1, H = 1 | M = 0) = (P(R = 1, H = 1 | C = 1, M = 0) * P(C = 1)) + (P(R = 1, H = 1 | C = 0, M = 0) * P(C = 0))")
print(" = ", p_rh_notm,'\n')

p_cnotm = 0.05*(1-p_m)
print("P(C = 1, M = 0) = P(C = 1) * P(M = 0)")
print(" = ", p_cnotm, '\n')

p_crhnotm = p_rh_cnotm * p_cnotm
print("P(C = 1, R = 1, H = 1, M = 0) = P(R = 1, H = 1 | C = 1, M = 0) * P(C = 1, M = 0)")
print(" = ", p_crhnotm, '\n')

p_rhnotm = p_rh_notm * (1-p_m)
print("P(R = 1, H = 1, M = 0) = P(R = 1, H = 1 | M = 0) * P(M = 0)")
print(" = ", p_rhnotm, '\n')
p_c_rhnotm = p_crhnotm / p_rhnotm
print("P(C = 1 | R = 1, H = 1, M = 0) = P(C = 1, R = 1, H = 1, M = 0) / P(R = 1, H = 1, M = 0)")
print("===========================================")
print("     Part A: ")
print("     P(C = 1 | R = 1, H = 1, M = 0): ",p_c_rhnotm)
print("===========================================\n")

p_h_mnotc = 0.98
p_h_notcnotm = 0.07
p_h_notc = (p_h_mnotc * p_m + p_h_notcnotm * (1-p_m))
print("P(H = 1 | C = 0) = (P(H = 1 | C = 0, M = 1) * P(M = 1)) + (P(H = 1 | C = 0, M = 0) * P(M = 0))")
print(" = ", p_h_notc, '\n')

p_mnotc = p_m * 0.95
print("P(C = 0, M = 1) = P(C = 0) * P(M = 1)")
print(" = ", p_mnotc, '\n')

p_mhnotc = p_h_mnotc * p_mnotc
print("P(H = 1, M = 1, C = 0) = P(H = 1 | M = 1, C = 0) * P(M = 1, C = 0)")
print(" = ", p_mhnotc,'\n')
p_hnotc = p_h_notc * 0.95
print("P(H = 1, C = 0) = P(H = 1 | C = 0) * P(C = 0)")
print(" = ", p_hnotc,'\n')
p_m_hnotc = p_mhnotc / p_hnotc
print("P(M = 1 | H = 1, C = 0) = P(H = 1, M = 1, C = 0) / P(H = 1, C = 0)")
print("===========================================")
print("     Part B: ")
print("     P(M = 1 | H = 1, C = 0): ",p_m_hnotc)
print("===========================================")