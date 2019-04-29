p_rh_cnotm = 0.9 * 0.6
print("P(R = 1, H = 1 | C = 1, M = 0): ", p_rh_cnotm)
p_rh_notcnotm = 0.05 * 0.07
print("P(R = 1, H = 1 | C = 0, M = 0): ", p_rh_notcnotm)

p_m = 0.999 * 0.0001
print("P(M = 1): ", p_m)

p_rh_notm = (p_rh_cnotm *0.05) + (p_rh_notcnotm*0.95) 
print("P(R = 1, H = 1 | M = 0): ", p_rh_notm)

p_cnotm = 0.05*(1-p_m)
print("P(C = 1, M = 0): ", p_cnotm)

p_crhnotm = p_rh_cnotm * p_cnotm
print("P(C = 1, R = 1, H = 1, M = 0): ", p_crhnotm)

p_rhnotm = p_rh_notm * (1-p_m)
print("P(R = 1, H = 1, M = 0): ", p_rhnotm)
p_c_rhnotm = p_crhnotm / p_rhnotm
print("===========================================")
print("     Part A: ")
print("     P(C = 1 | R = 1, H = 1, M = 0): ",p_c_rhnotm)
print("===========================================")

p_h_mnotc = 0.98
p_h_notcnotm = 0.07
p_h_notc = (p_h_mnotc * p_m + p_h_notcnotm * (1-p_m))
print("P(H = 1 | C = 0): ", p_h_notc)

p_mnotc = p_m * 0.95
print("P(M = 1, C = 0): ", p_mnotc)

p_mhnotc = p_h_mnotc * p_mnotc
print("P(H = 1 | M = 1, C = 0): ", p_mhnotc)
p_hnotc = p_h_notc * 0.95
print("P(H = 1, C = 0): ", p_hnotc)
p_m_hnotc = p_mhnotc / p_hnotc
print("===========================================")
print("     Part B: ")
print("     P(M = 1 | H = 1, C = 0): ",p_m_hnotc)
print("===========================================")