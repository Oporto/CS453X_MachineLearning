

p_rh_cnotm = 0.9 * 0.6
p_rh_notcnotm = 0.05 * 0.07
p_rh_cm = 0.98 * 0.99
p_rh_notcm = 0.98 * 0.5

p_m = 0.999 * 0.0001

p_rh_c = (p_rh_cm *p_m) + (p_rh_cnotm*(1-p_m)) 
p_rh_notc = (p_rh_notcm *p_m) + (p_rh_notcnotm*(1-p_m)) 

p_rh_m = (p_rh_cm *0.05) + (p_rh_notcm*0.95) 
p_rh_notm = (p_rh_cnotm *0.05) + (p_rh_notcnotm*0.95) 

p_cnotm = 0.05*(1-p_m)

p_crhnotm = p_rh_cnotm * p_cnotm

p_rhnotm = p_rh_notm * (1-p_m)
p_c_rhnotm = p_crhnotm / p_rhnotm
print("Part A: ")
print(p_c_rhnotm)

p_noth_mc = 0.01
p_noth_cnotm = 0.4
p_noth_c = (p_noth_mc * p_m + p_noth_cnotm * (1-p_m))

p_mnothc = p_noth_mc
p_nothc = p_noth_c * 0.05
p_m_nothc = p_mnothc / p_nothc
print("Part B: ")
print(p_m_nothc)