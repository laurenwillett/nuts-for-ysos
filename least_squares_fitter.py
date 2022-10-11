from scipy.optimize import least_squares
from scipy import linalg
import numpy as np
import math
from astropy.table import Table
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt
from scipy import interpolate


from import_templates import prep_scale_templates
from features_to_evaluate import avg_value_feature, slope_feature, photometry_feature

def chi2_like(observed, errs, model):
    chi2 = 0
    N = len(observed)
    for n in range(0, N):
        y_val = observed[n]
        err = errs[n]
        est_val = model[n]
        chi2_element = ((y_val - est_val)/err)**2
        chi2 += chi2_element
    return chi2

def K_solver(def_wave_model, init_slab_model, init_photosphere, def_wave_data, YSO, Av, Rv=3.1):
    
    #set to round to nearest even number
    f360_YSO = np.mean(YSO[int(np.where(np.rint(def_wave_data/2)*2 == 3568)[0][0]):int(np.where(np.rint(def_wave_data/2)*2 == 3588)[0][0])])
    f550_YSO = np.mean(YSO[int(np.where(np.rint(def_wave_data/2)*2 == 5090)[0][0]):int(np.where(np.rint(def_wave_data/2)*2 == 5110)[0][0])])
    
    #Cardelli et al 1989 reddening law
    c = 2.99792458 * (1e10)
    wavelength_micron = def_wave_model / 10000
    wave_inv = np.arange(0.3, 3.3, .0001)
    opacity = []
    for x in wave_inv:
        y = x-1.82
        a = 1 + (0.17699*y) - (0.50447*(y**2)) - (0.02427*(y**3)) + (0.72085*(y**4)) + (0.01979*(y**5)) - (0.77530*(y**6)) + (.32999*(y**7))
        b = (1.41338*y) + (2.28305*(y**2)) + (1.07233*(y**3)) - (5.38434*(y**4)) - (0.62251*(y**5)) + (5.30260*(y**6)) - (2.09002*(y**7))
        z = a + (b/Rv)
        opacity.append(z)
    red_law_OPT = interpolate.interp1d(wave_inv, opacity)
    opacity = []
    for x in wave_inv:
        a = 0.574*(x**1.61)
        b = -0.527*(x**1.61)
        z = a + (b/Rv)
        opacity.append(z)
    red_law_IR = interpolate.interp1d(wave_inv, opacity)
    OPTmask = [(wavelength_micron**-1) >= 1.1]
    IRmask = [(wavelength_micron**-1) < 1.1]
    A_specific = Av * ((red_law_OPT(1/wavelength_micron)*OPTmask) + (red_law_IR(1/wavelength_micron)*IRmask))
    
    init_slab_model_red = np.array((init_slab_model * (10 ** (-0.4 * A_specific)))[0])
    init_photosphere_red = np.array((init_photosphere * (10 ** (-0.4 * A_specific)))[0])
    f360_slab = init_slab_model_red[(np.where(np.rint(def_wave_model/2)*2 == 3578)[0][0])]
    f550_slab = init_slab_model_red[(np.where(np.rint(def_wave_model/2)*2 == 5100)[0][0])]
    f360_phot = np.mean(init_photosphere_red[int(np.where(np.rint(def_wave_model/2)*2 == 3568)[0][0]):int(np.where(np.rint(def_wave_model/2)*2 == 3588)[0][0])])
    f550_phot = np.mean(init_photosphere_red[int(np.where(np.rint(def_wave_model/2)*2 == 5090)[0][0]):int(np.where(np.rint(def_wave_model/2)*2 == 5110)[0][0])])
    b = np.array([f360_YSO, f550_YSO])
    a = np.array([[f360_phot, f360_slab], [f550_phot, f550_slab]])
    X = np.linalg.solve(a,b)
    Kphot_0 = X[0]
    Kslab_0 = X[1]
    return Kslab_0, Kphot_0


def least_squares_fit_function(def_wave_data, mean_resolution, YSO, YSO_spectrum_features, YSO_spectrum_features_errs, feature_types, feature_bounds, Rv=3.1, plot=True):
    print('performing initial least squares fit')
    template_Teffs, template_lums, def_wave, templates_scaled = prep_scale_templates(def_wave_data, mean_resolution)
    
    T = tt.scalar('T')
    n_e = tt.scalar('n_e')
    tau_0 = tt.scalar('tau_0')
    Kslab = tt.scalar('Kslab')
    Kphot = tt.scalar('Kphot')
    Av = tt.scalar('Av')
    photosphere = tt.vector('photosphere')
    
    c = 2.99792458 * (1e10)
    h = 6.62607004 * (1e-27)
    k_B = 1.38064852 * (1e-16)
    nu_0 = 3.28795 * (1e15) #hz -- ionization frequency for hydrogen
    mH = 1.6735 * (1e-24)
    me = 9.10938356 * (1e-28)
    mp = 1.6726219 * (1e-24)
    Z_i = 1

    lamb_0 = 1.6419 #photodetachment threshold in microns
    alpha = 1.439 * (1e4)
    lambda_0 = 300e-7 #defined by Manara 2013b
    freq_0 = c/lambda_0
    
    nu = c*(1e8) / def_wave
    diff = def_wave[1]-def_wave[0]
    wavelength_spacing_model = diff
    full_wave = wavelength_spacing_model * np.arange(500/wavelength_spacing_model, 25000/wavelength_spacing_model)
    nu_2 = c*(1e8) / full_wave
    wave_cm_2 = (full_wave*(1e-8))

    #valid 0.182 to 0.3645 um 
    Ans_ff_1 = [518.1021,  473.2636, -482.2089 , 115.5291, 0, 0]
    Bns_ff_1 = [-734.8667, 1443.4137 , -737.1616,  169.6374, 0, 0]
    Cns_ff_1 = [1021.1775, -1977.3395,  1096.8827, -245.6490, 0, 0]
    Dns_ff_1 = [-479.0721, 922.3575, -521.1341, 114.2430, 0, 0]
    Ens_ff_1 = [93.1373, -178.9275, 101.7963, -21.9972, 0, 0]
    Fns_ff_1 = [-6.4285, 12.3600, -7.0571,  1.5097, 0, 0]

    #valid > 0.3645 um 
    Ans_ff_2 = [0, 2483.3460, -3449.8890, 2200.0400, -696.2710, 88.2830]
    Bns_ff_2 = [0, 285.8270, -1158.3820, 2427.7190, -1841.4000, 444.5170]
    Cns_ff_2 = [0, -2054.2910, 8746.5230, -13651.1050, 8624.9700, -1863.8640]
    Dns_ff_2 = [0, 2827.7760, -11485.6320, 16755.5240, -10051.5300, 2095.2880]
    Ens_ff_2 = [0, -1341.5370, 5303.6090, -7510.4940, 4400.0670, -901.7880]
    Fns_ff_2 = [0, 208.9520, -812.9390, 1132.7380, -655.0200, 132.9850]

    Cns_fb = [152.519, 49.534, -118.858, 92.536, -34.194, 4.982] #valid from 0.125 to 1.6419 um

    B_out_2 = 2*h*(nu_2**3)*(1/((tt.exp((h*nu_2)/(k_B*T)))-1))/(c**2)
    coeff = (2*h*nu_0*(Z_i**2)) / (k_B*T)
    m_2 = tt.floor((nu_0*(Z_i**2)/nu_2)**(1/2)+1)
    frac1_2 = tt.switch(tt.eq(m_2, 1.0), (nu_2/(nu_0 * (Z_i**2))), 0)
    frac1_2 = frac1_2[frac1_2.nonzero()]
    frac2_2 = tt.switch(tt.eq(m_2, 2.0), (nu_2/(nu_0 * (Z_i**2))), 0)
    frac2_2 = frac2_2[frac2_2.nonzero()]
    frac3_2 = tt.switch(tt.eq(m_2, 3.0), (nu_2/(nu_0 * (Z_i**2))), 0)
    frac3_2 = frac3_2[frac3_2.nonzero()]
    frac4_2 = tt.switch(tt.eq(m_2, 4.0), (nu_2/(nu_0 * (Z_i**2))), 0)
    frac4_2 = frac4_2[frac4_2.nonzero()]
    frac5_2 = tt.switch(tt.eq(m_2, 5.0), (nu_2/(nu_0 * (Z_i**2))), 0)
    frac5_2 = frac5_2[frac5_2.nonzero()]
    frac6_2 = tt.switch(tt.eq(m_2, 6.0), (nu_2/(nu_0 * (Z_i**2))), 0)
    frac6_2 = frac6_2[frac6_2.nonzero()]
    y1_2 = tt.zeros(frac1_2.shape)
    y2_2 = tt.zeros(frac2_2.shape)
    y3_2 = tt.zeros(frac3_2.shape)
    y4_2 = tt.zeros(frac4_2.shape)
    y5_2 = tt.zeros(frac5_2.shape)
    y6_2 = tt.zeros(frac6_2.shape)
    for n in range(1, 20):
        gn = 1 + (0.1728*(frac1_2**(1/3) - (2*(frac1_2**(-2/3))/(n**2)))) - .0496*(frac1_2**(2/3) - (2*(frac1_2**(-1/3))/((3*(n**2)))) + (2*(frac1_2**(-4/3))/((3*(n**4)))))
        exp = (h*nu_0) / ((n**2)*k_B*T)
        y1_2 += (n**-3)*(tt.exp(exp))*gn
    for n in range(2, 20):
        gn = 1 + (0.1728*(frac2_2**(1/3) - (2*(frac2_2**(-2/3))/(n**2)))) - .0496*(frac2_2**(2/3) - (2*(frac2_2**(-1/3))/((3*(n**2)))) + (2*(frac2_2**(-4/3))/((3*(n**4)))))
        exp = (h*nu_0) / ((n**2)*k_B*T)
        y2_2 += (n**-3)*(tt.exp(exp))*gn
    for n in range(3, 20):
        gn = 1 + (0.1728*(frac3_2**(1/3) - (2*(frac3_2**(-2/3))/(n**2)))) - .0496*(frac3_2**(2/3) - (2*(frac3_2**(-1/3))/((3*(n**2)))) + (2*(frac3_2**(-4/3))/((3*(n**4)))))
        exp = (h*nu_0) / ((n**2)*k_B*T)
        y3_2 += (n**-3)*(tt.exp(exp))*gn    
    for n in range(4, 20):
        gn = 1 + (0.1728*(frac4_2**(1/3) - (2*(frac4_2**(-2/3))/(n**2)))) - .0496*(frac4_2**(2/3) - (2*(frac4_2**(-1/3))/((3*(n**2)))) + (2*(frac4_2**(-4/3))/((3*(n**4)))))
        exp = (h*nu_0) / ((n**2)*k_B*T)
        y4_2 += (n**-3)*(tt.exp(exp))*gn 
    for n in range(5, 20):
        gn = 1 + (0.1728*(frac5_2**(1/3) - (2*(frac5_2**(-2/3))/(n**2)))) - .0496*(frac5_2**(2/3) - (2*(frac5_2**(-1/3))/((3*(n**2)))) + (2*(frac5_2**(-4/3))/((3*(n**4)))))
        exp = (h*nu_0) / ((n**2)*k_B*T)
        y5_2 += (n**-3)*(tt.exp(exp))*gn 
    for n in range(6, 20):
        gn = 1 + (0.1728*(frac6_2**(1/3) - (2*(frac6_2**(-2/3))/(n**2)))) - .0496*(frac6_2**(2/3) - (2*(frac6_2**(-1/3))/((3*(n**2)))) + (2*(frac6_2**(-4/3))/((3*(n**4)))))
        exp = (h*nu_0) / ((n**2)*k_B*T)
        y6_2 += (n**-3)*(tt.exp(exp))*gn 
    g_fb_out_2=(tt.concatenate((y1_2,y2_2,y3_2,y4_2,y5_2,y6_2)))*coeff
    g_ff_out_2 = 1 + (0.1728* ((nu_2/(nu_0*(Z_i**2)))**(1/3)) *(1+(2*k_B*T/(h*nu_2)))) - (.0496*((nu_2/(nu_0*(Z_i**2)))**(2/3)) * (1+(2*k_B*T/(3*h*nu_2)) +((4/3)*((k_B*T/(h*nu_2))**2)) ))
    j_out_2 = 5.44*(1e-39)*(tt.exp((-h*nu_2)/(k_B*T)))*(T**(-1/2)) * (n_e**2) * (g_ff_out_2 + g_fb_out_2)
    coeff = (2*h*nu_0*(Z_i**2)) / (k_B*T)
    B_Lslab_out = 2*h*(freq_0**3)*(1/((tt.exp((h*freq_0)/(k_B*T)))-1))/(c**2)
    g_ff_Lslab_out = 1 + (0.1728* ((freq_0/(nu_0*(Z_i**2)))**(1/3)) *(1+(2*k_B*T/(h*freq_0)))) - (.0496*((freq_0/(nu_0*(Z_i**2)))**(2/3)) * (1+(2*k_B*T/(3*h*freq_0)) +((4/3)*((k_B*T/(h*freq_0))**2)) ))
    g_fb_Lslab_out = 0
    for n in range(2, 20):
        gn = 1 + (0.1728*((freq_0/(nu_0 * (Z_i**2)))**(1/3) - (2*((freq_0/(nu_0 * (Z_i**2)))**(-2/3))/(n**2)))) - .0496*((freq_0/(nu_0 * (Z_i**2)))**(2/3) - (2*((freq_0/(nu_0 * (Z_i**2)))**(-1/3))/((3*(n**2)))) + (2*((freq_0/(nu_0 * (Z_i**2)))**(-4/3))/((3*(n**4)))))
        g_fb_Lslab_out += (n**-3)*(tt.exp((h*nu_0) / ((n**2)*k_B*T)))*gn
    g_fb_Lslab_out *= coeff 
    j_Lslab_out = 5.44*(1e-39)*(tt.exp((-h*freq_0)/(k_B*T)))*(T**(-1/2)) * (n_e**2) * (g_ff_Lslab_out + g_fb_Lslab_out)
    Lslab_out = tau_0 * B_Lslab_out / j_Lslab_out
    tau_H_out_2 = j_out_2 * Lslab_out / B_out_2 
    lamb_2 = (c/nu_2) *(1e4)
    f_out_2 = tt.zeros(len(nu_2))
    for n in range(0,6):
        Cn = Cns_fb[n]
        f_out_2+= tt.switch(lamb_2 < lamb_0, Cn * ((1/lamb_2) - (1/lamb_0))**((n)/2),0)
    sigma_out_2 = tt.switch(lamb_2 < lamb_0, (1e-18)*(lamb_2**3)*(((1/lamb_2) - (1/lamb_0))**(3/2))*(f_out_2),0)
    k_fb__out_2 = 0.750*(T**(-5/2))*(tt.exp(alpha/(lamb_0*T))) * (1-(tt.exp(-alpha/(lamb_2*T)))) * sigma_out_2
    lamb1_2 = tt.switch((lamb_2 <= 0.182), lamb_2, 0)
    lamb1_2 = lamb1_2[lamb1_2.nonzero()]  
    lamb2_2 = tt.switch((lamb_2 < 0.3645), lamb_2, 0)
    lamb2_2 = tt.switch((lamb2_2 > 0.182), lamb2_2, 0)
    lamb2_2 = lamb2_2[lamb2_2.nonzero()] 
    lamb3_2 = tt.switch((lamb_2 >= 0.3645), lamb_2, 0)
    lamb3_2 = lamb3_2[lamb3_2.nonzero()] 
    y1_2 = tt.zeros(lamb1_2.shape)
    y2_2 = tt.zeros(lamb2_2.shape)
    y3_2 = tt.zeros(lamb3_2.shape)
    for n in range(0,6):
        y2_2+= ((5040/T)**((n+1)/2)) * (((lamb2_2**2)*Ans_ff_1[n]) + Bns_ff_1[n] + (Cns_ff_1[n]/lamb2_2) + (Dns_ff_1[n]/(lamb2_2**2)) + (Ens_ff_1[n]/(lamb2_2**3)) + (Fns_ff_1[n]/(lamb2_2**4)))
        y3_2+= ((5040/T)**((n+1)/2)) * (((lamb3_2**2)*Ans_ff_2[n]) + Bns_ff_2[n] + (Cns_ff_2[n]/lamb3_2) + (Dns_ff_2[n]/(lamb3_2**2)) + (Ens_ff_2[n]/(lamb3_2**3)) + (Fns_ff_2[n]/(lamb3_2**4)))
    k_ff__out_2 = (tt.concatenate((y1_2, y2_2, y3_2)))*(1e-29)
    coeff2 = ((h**3)/((2*math.pi*me*k_B)**(3/2)))
    n_H_out=0
    for n in range(1,20):
        n_H_out += (n**2)*(tt.exp(h*nu_0/((n**2)*k_B*T)))
    n_H_out*= coeff2*(T**(-3/2))*(n_e**2)
    k_H__out_2 = (k_fb__out_2 + k_ff__out_2)*n_e*n_H_out*k_B*T
    tau_H__out_2 = k_H__out_2 * Lslab_out
    I_H_out_2 = tau_H_out_2 * B_out_2 * ((1-(tt.exp(-tau_H_out_2)))/tau_H_out_2)
    I_H__out_2 = tau_H__out_2 * B_out_2 * ((1-(tt.exp(-tau_H__out_2)))/tau_H__out_2)
    tau_total_2 = tau_H_out_2 + tau_H__out_2
    beta_tau_total_out_2 = (1-(tt.exp(-tau_total_2)))/tau_total_2
    I_both_out_2 = tau_total_2 * B_out_2 * beta_tau_total_out_2

    generate_slab_out_2 = (c*I_both_out_2/((wave_cm_2)**2)) * (1e-8)
    slab_shortened = generate_slab_out_2[(tt.eq(nu_2, nu[0])).nonzero()[0][0]:((tt.eq(nu_2, nu[-1])).nonzero()[0][0]+int(diff/wavelength_spacing_model)):int(diff/wavelength_spacing_model)]
    generate_slab = theano.function([T, n_e, tau_0], slab_shortened)
    
    #Cardelli et al 1989 reddening law (theano version)
    wavelength_micron = def_wave / 10000
    wave_inv = (wavelength_micron**-1)
    x_OPT = wave_inv-1.82
    a_OPT = 1 + (0.17699*x_OPT) - (0.50447*(x_OPT**2)) - (0.02427*(x_OPT**3)) + (0.72085*(x_OPT**4)) + (0.01979*(x_OPT**5)) - (0.77530*(x_OPT**6)) + (.32999*(x_OPT**7))
    b_OPT = (1.41338*x_OPT) + (2.28305*(x_OPT**2)) + (1.07233*(x_OPT**3)) - (5.38434*(x_OPT**4)) - (0.62251*(x_OPT**5)) + (5.30260*(x_OPT**6)) - (2.09002*(x_OPT**7))
    z_OPT = a_OPT + (b_OPT/Rv)
    x_IR = wave_inv
    a_IR = 0.574*(x_IR**1.61)
    b_IR = -0.527*(x_IR**1.61)
    z_IR = a_IR + (b_IR/Rv)
    A_specific  = (Av)* tt.switch(((wavelength_micron**-1) >= 1.1), z_OPT, z_IR)
    
    reddened_slab = (slab_shortened * (10 ** (-0.4 * A_specific)))
    reddened_photosphere = (photosphere * (10 ** (-0.4 * A_specific)))
    model = reddened_slab*Kslab + reddened_photosphere*Kphot
    generate_model = theano.function([T, n_e, tau_0, Kslab, Kphot, Av, photosphere], [reddened_slab*Kslab, reddened_photosphere*Kphot, model])
    
    x0 = [7000.0, 1e13, 1, 'Kslab', 'Kphot', 0.0]
    init_slab_model = np.array(generate_slab(x0[0], x0[1], x0[2]))
    ftol_mine = 1e-04
    
    fit_photospheres = []
    chi_squares = []
    fit_infos = []
    fit_Teffs = []
    params_saved = []
    
    def residuals(model_params):
        model = generate_model(model_params[0], model_params[1], model_params[2], model_params[3], model_params[4], model_params[5], init_photosphere)[2]
        #model_features = np.array(make_feature_list(def_wave, model, err=False))
        model_features = []
        for f in range(0, len(feature_types)):
            if feature_types[f] == 'value':
                model_features.append(avg_value_feature(def_wave, feature_bounds[f][0], feature_bounds[f][1], model))
            if feature_types[f] == 'slope':
                model_features.append(slope_feature(def_wave, feature_bounds[f][0], feature_bounds[f][1], feature_bounds[f][2], feature_bounds[f][3], model))
            if feature_types[f] == 'photometry':
                model_features.append(photometry_feature(def_wave, feature_bounds[f], model))
        model_features = np.array(model_features)
                
        residual = (YSO_spectrum_features - model_features) / YSO_spectrum_features_errs
        return residual
    
    for t in range(0, len(templates_scaled)):
        init_photosphere = templates_scaled[t]
        init_photosphere_lum = template_lums[t]
        init_Teff = template_Teffs[t]
    
        Kslab_0, Kphot_0 = K_solver(def_wave, init_slab_model, init_photosphere, def_wave_data, YSO, 0)
        Kphot_upper = 1000
        Kphot_lower = 0
        #make a procedure for if an initial guess w/ Av=0 is infeasible
        Av_try = 0
        while Kphot_0>Kphot_upper or Kphot_0<Kphot_lower or Kslab_0 <0:
            Av_try += 1
            Kslab_0, Kphot_0 = K_solver(def_wave, init_slab_model, init_photosphere, YSO, Av_try)
            if Av_try == 10:
                break
        x0[3] = Kslab_0 
        x0[4] = Kphot_0
        x0[5] = Av_try
        
        try:
            x = least_squares(residuals, x0, xtol = None, ftol = ftol_mine, gtol = None, method = 'trf', x_scale = [1000, x0[1], 1, Kslab_0, Kphot_0, 1], bounds = ([5000, 1e11,.01,0,0,0],[11000,1e16,5.0,np.inf,np.inf,10]))
            fit_infos.append(x)
            model_params_results = x['x']
            params_saved.append([model_params_results[0], model_params_results[1], model_params_results[2], model_params_results[3], model_params_results[4], model_params_results[5], init_Teff])
            good_photosphere = init_photosphere
            good_model = generate_model(model_params_results[0], model_params_results[1], model_params_results[2], model_params_results[3], model_params_results[4], model_params_results[5], good_photosphere)[2]
            #good_model_features =  np.array(make_feature_list(def_wave, good_model, err=False))
            good_model_features = []
            for f in range(0, len(feature_types)):
                if feature_types[f] == 'value':
                    good_model_features.append(avg_value_feature(def_wave, feature_bounds[f][0], feature_bounds[f][1], good_model))
                if feature_types[f] == 'slope':
                    good_model_features.append(slope_feature(def_wave, feature_bounds[f][0], feature_bounds[f][1], feature_bounds[f][2], feature_bounds[f][3], good_model))
                if feature_types[f] == 'photometry':
                    good_model_features.append(photometry_feature(def_wave, feature_bounds[f], good_model))
            good_model_features = np.array(good_model_features)
            chisq = chi2_like(YSO_spectrum_features, YSO_spectrum_features_errs, good_model_features)
            chi_squares.append(chisq)
            fit_photospheres.append(init_photosphere)
            fit_Teffs.append(init_Teff)
        except Exception:
            #print('infeasible SpT \n')
            pass
            
    if len(chi_squares) !=0:
        best_fit_Teff= fit_Teffs[np.where(chi_squares == np.min(chi_squares))[0][0]]
        best_fit_info = (fit_infos[np.where(chi_squares == np.min(chi_squares))[0][0]])
        best_fit_params = best_fit_info['x']
        best_fit_params= np.concatenate((best_fit_params, [best_fit_Teff]))
        
        if plot==True:
            best_photosphere = fit_photospheres[np.where(chi_squares == np.min(chi_squares))[0][0]]
            best_model_components = generate_model(best_fit_params[0], best_fit_params[1], best_fit_params[2], best_fit_params[3], best_fit_params[4], best_fit_params[5], best_photosphere)
            best_slab_reddened = best_model_components[0]
            best_photosphere_reddened = best_model_components[1]
            best_model = best_model_components[2]
            plt.plot(def_wave, best_model, label = 'model fit', zorder=4, c = 'blue', lw = 1, alpha = 0.5)
            plt.plot(def_wave, best_slab_reddened, label = 'slab', zorder=3, c = 'black', lw = 1)
            plt.plot(def_wave, best_photosphere_reddened, label = 'photosphere', zorder=2, c = 'lime', lw = 1)
            plt.plot(def_wave_data, YSO, label = 'data', zorder=1, c = 'red', lw = 1, alpha = 0.5)
            plt.xlim(def_wave_data[0], def_wave_data[-1])
            plt.xlabel('wavelength (Angstrom)')
            plt.ylabel('flux (e-17 ergs/s/cm2/A)')
            plt.title('least squares fit result')
            plot_ylim = 1.2*np.max(YSO)
            plt.ylim(plot_ylim/(-100),plot_ylim)
            plt.legend()
            plt.show()
        
        return(best_fit_params)
    
    else:
        print('no good least square fit')





