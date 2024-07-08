import numpy as np
import math
from astropy.table import Table
import arviz
import pymc as pm
import pytensor
import pytensor.tensor as tt
import scipy.optimize as optimize
import scipy.stats as stats
from scipy.stats.kde import gaussian_kde
from import_templates import prep_scale_templates

def make_gamma_dist(x1, p1, x2, p2): #p1 and p2 are percentiles (eg. 16 and 84) with respective values x1 and x2
    # Standardize so that x1 < x2 and p1 < p2
    if p1 > p2:
        (p1, p2) = (p2, p1)
        (x1, x2) = (x2, x1)
    # function to find roots of for gamma distribution parameters
    def objective(inputted):
        return [stats.gengamma.ppf(p2, inputted[0], inputted[1]) / stats.gengamma.ppf(p1, inputted[0], inputted[1]) - x2/x1,0]
    initialGuess = [1,2]
    alpha = optimize.root(objective, initialGuess)
    a = alpha['x'][0]
    c = alpha['x'][1]
    beta = x1 / stats.gengamma.ppf(p1,a,c)
    gamma_dist = stats.gengamma.rvs(a,c, size=32000)*beta
    kde = gaussian_kde(gamma_dist)
    dist_space = np.linspace( min(gamma_dist), max(gamma_dist), 1000 )
    return dist_space, kde(dist_space)


def pymc_NUTS_fitting(def_wave_data, mean_resolution, YSO_spectrum_features, YSO_spectrum_features_errs, feature_types, feature_bounds, distance_info, name, init_params, target_accept_set, length, chains, cores, Rv=3.1):
    
    template_Teffs, template_lums, def_wave, templates_scaled = prep_scale_templates(def_wave_data, mean_resolution)
    
    print('initializing PyMC fitter')
    c = 2.99792458 * (1e10)
    nu = c*(1e8) / def_wave
    diff = def_wave[1]-def_wave[0]
    wavelength_spacing_model = diff #angstroms
    dnu = tt.extra_ops.diff(nu)
    dnu = tt.concatenate([np.array([dnu[0]]), dnu])
    full_wave = wavelength_spacing_model*np.arange((def_wave_data[0]/wavelength_spacing_model-((def_wave_data[0]-500)//wavelength_spacing_model)),(def_wave_data[0]/wavelength_spacing_model-((def_wave_data[0]-25000)//wavelength_spacing_model)))
    nu_2 = c*(1e8) / full_wave
    wave_cm_2 = (full_wave*(1e-8))
    
    #various constants needed for the model
    Lsun = 3.828*(10**33)
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

    Cns_fb = [152.519, 49.534, -118.858, 92.536, -34.194, 4.982] #valid from 0.125 to 1.6419 um (the photodetachment threshold)

    with pm.Model() as model1:
        #setting priors:
        T = pm.Uniform('T', 5000, 15000)
        n_e_log = pm.Uniform('n_e_log', 10, 16)
        n_e = 10**(n_e_log)
        tau_0 = pm.Uniform('tau_0', 0.01, 5)
        Kslab_1e6 = pm.Uniform('Kslab_1e6', 0, init_params[3]*1e6*1000)
        Kslab = Kslab_1e6/(1e6)
        Kphot_1e6 = pm.Uniform('Kphot_1e6', 0, init_params[4]*1e6*1000)
        Kphot = Kphot_1e6/(1e6)
        Kslab_1e6_log = pm.Deterministic('Kslab_1e6_log', tt.log10(Kslab_1e6))
        Kphot_1e6_log = pm.Deterministic('Kphot_1e6_log', tt.log10(Kphot_1e6))
        Av = pm.Uniform('Av', 0, 10)
        Av_grid_uncert = pm.HalfNormal('Av_grid_uncert', sigma=(0.5)/3) #not an inferred model param
        Teff = pm.Uniform('Teff', 2615, 5550)
        Teff_grid_uncert_dist = pm.Normal.dist(mu=0.0, sigma=100) #not an inferred model param
        Teff_grid_uncert = pm.Truncated("Teff_grid_uncert", Teff_grid_uncert_dist, lower=-200, upper=200)
        Lum_grid_uncert_dist = pm.Normal.dist(mu=0.0, sigma=0.2, shape=(len(template_lums))) #not an inferred model param
        Lum_grid_uncert = pm.Truncated("Lum_grid_uncert", Lum_grid_uncert_dist, lower=-0.5, upper=0.5)
        
        #whether or not to make distance a prior or just a scalar depends on what information you have
        if isinstance(distance_info, np.ndarray) == True or isinstance(distance_info, list) == True:
            d_target, d_l_target, d_u_target = distance_info #parsecs
            d_dist = make_gamma_dist(d_l_target, 0.16, d_u_target, 0.84)
            distance = pm.Interpolated('distance', d_dist[0], d_dist[1])
        elif isinstance(distance_info, float) == True or isinstance(distance_info, int) == True:
            d_target = distance_info
            distance = d_target
        else:
            print('input distance should be in parsecs, as either a float or integer, or a list/array of the format [mean_distance, lower bound , upper bound]')
            break
            
        templates_scaled_shared = tt.as_tensor(np.array(templates_scaled))
        template_Teffs_shared_0 = tt.as_tensor(np.array(template_Teffs))
        template_Teffs_shared = template_Teffs_shared_0
        template_lums_shared_0 =  tt.as_tensor(template_lums)
        template_lums_shared = template_lums_shared_0 + Lum_grid_uncert
        
        template_Teff_right = (tt.switch((template_Teffs_shared > (Teff+Teff_grid_uncert)), template_Teffs_shared, 0))
        template_Teff_right = tt.min(template_Teff_right[template_Teff_right.nonzero()])
        template_right = templates_scaled_shared[(tt.eq(template_Teffs_shared, template_Teff_right).nonzero()[0][0])]
        template_Teff_left = (tt.switch((template_Teffs_shared <= (Teff+Teff_grid_uncert)), template_Teffs_shared, 0))
        template_Teff_left = tt.max(template_Teff_left[template_Teff_left.nonzero()]) 
        template_left = templates_scaled_shared[(tt.eq(template_Teffs_shared, template_Teff_left).nonzero()[0][0])]
        rightweight = (Teff+Teff_grid_uncert-template_Teff_left)/(template_Teff_right-template_Teff_left)
        leftweight = (template_Teff_right-(Teff+Teff_grid_uncert))/(template_Teff_right-template_Teff_left)
        my_template = tt.switch(tt.eq(template_Teff_left,(Teff+Teff_grid_uncert)),template_left,(leftweight*template_left + rightweight*template_right))
        photosphere = my_template
        template_lum_right = template_lums_shared[(tt.eq(template_Teffs_shared, template_Teff_right).nonzero()[0][0])]
        template_lum_left = template_lums_shared[(tt.eq(template_Teffs_shared, template_Teff_left).nonzero()[0][0])]
        my_template_lum = tt.switch(tt.eq(template_Teff_left,(Teff+Teff_grid_uncert)),template_lum_left,(leftweight*template_lum_left + rightweight*template_lum_right))
        
        #the making of the slab model:
        B_out_2 = 2*h*(nu_2**3)*(1/((tt.exp((h*nu_2)/(k_B*T)))-1))/(c**2)
        coeff = (2*h*nu_0*(Z_i**2)) / (k_B*T)
        m_2 = tt.floor((nu_0*(Z_i**2)/nu_2)**(1/2)+1)

        ##this is the new part that I had to change to omit pytensor.scan
        frac = nu_2/(nu_0 * (Z_i**2))
        n_s = np.arange(1,20,1)
        exps = (h*nu_0) / ((n_s**2)*k_B*T)
        additives = tt.zeros((len(n_s), len(nu_2)))
        for n in n_s:
            gn = 1 + (0.1728*(frac**(1/3) - (2*(frac**(-2/3))/(n**2)))) - .0496*(frac**(2/3) - (2*(frac**(-1/3))/((3*(n**2)))) + (2*(frac**(-4/3))/((3*(n**4)))))
            additives = tt.set_subtensor(additives[n-1], (1/(n*n*n))*(tt.exp(exps[n-1]))*gn)
        y_s = tt.zeros(0)
        for start_val in range(1,7): #start_val is m
            additives_masked = tt.switch(tt.eq(m_2, start_val), additives, 0)
            y = tt.sum(additives_masked[start_val-1:], axis=0) #index additives [start_val-1:] and sum those components together
            y = y[y.nonzero()]
            y_s = tt.concatenate((y_s,y))
        g_fb_out_2=y_s*coeff
        
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
        lamb_2_alt = tt.switch(lamb_2 < lamb_0, lamb_2, lamb_0)
        f_out_2 = tt.zeros(len(nu_2))
        for n in range(0,6):
            Cn = Cns_fb[n]
            f_out_2+= tt.switch(lamb_2 < lamb_0, Cn * ((1/lamb_2_alt) - (1/lamb_0))**((n)/2),0)
        sigma_out_2 = tt.switch(lamb_2 < lamb_0, (1e-18)*(lamb_2_alt**3)*(((1/lamb_2_alt) - (1/lamb_0))**(3/2))*(f_out_2),0)
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

        #Cardelli et al 1989 reddening law
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
        A_specific_2  = (Av+Av_grid_uncert)* tt.switch(((wavelength_micron**-1) >= 1.1), z_OPT, z_IR)  
        
        reddened_slab = (slab_shortened * (10 ** (-0.4 * A_specific_2)))
        reddened_photosphere = (photosphere * (10 ** (-0.4 * A_specific)))
        model = reddened_slab*Kslab + reddened_photosphere*Kphot
        y = model
        
        number_of_features = len(YSO_spectrum_features)
        model_spec_features = tt.zeros(number_of_features)
        for f in range(0, len(feature_types)):
            if feature_types[f] == 'point':
                model_spec_features = tt.set_subtensor(model_spec_features[f], (tt.mean(y[(tt.eq(def_wave, feature_bounds[f][0]).nonzero()[0][0]):(tt.eq(def_wave, feature_bounds[f][1]).nonzero()[0][0])])))
            if feature_types[f] == 'slope':
                model_spec_features = tt.set_subtensor(model_spec_features[f],(tt.mean(y[(tt.eq(def_wave, feature_bounds[f][2]).nonzero()[0][0]):(tt.eq(def_wave, feature_bounds[f][3]).nonzero()[0][0])])) - (tt.mean(y[(tt.eq(def_wave, feature_bounds[f][0]).nonzero()[0][0]):(tt.eq(def_wave, feature_bounds[f][1]).nonzero()[0][0])])))
            if feature_types[f] == 'photometry':
                temp_spectrum = 1e-17 *model * 1e29 * (def_wave**2) / (c*(10**8)) #units conversion
                #this line convolves the model spectrum with the photometric filter
                mag = tt.dot((nu/dnu*temp_spectrum), feature_bounds[f]) / tt.sum((nu/dnu*feature_bounds[f]))
                Mag = -2.5 * tt.log10(mag) + 23.9
                model_spec_features = tt.set_subtensor(model_spec_features[f], Mag)
        model_spec_features_traced = pm.Deterministic('model_spec_features_traced', model_spec_features)
        
        integral = tt.sum(generate_slab_out_2*wavelength_spacing_model*Kslab/(1e17)) * 4*math.pi*((distance* 3.08567775815 * (10**18))**2)
        Lacc_log_current = tt.log10(integral/Lsun)
        Lacc_log_traced = pm.Deterministic('Lacc_log_traced',Lacc_log_current)
        
        L_log_current = tt.log10(Kphot * (10**my_template_lum) * (distance**2))
        L_log_traced = pm.Deterministic('L_log_traced',L_log_current)
        
        observation = pm.Normal('observation', mu=model_spec_features, sigma=YSO_spectrum_features_errs, observed = YSO_spectrum_features)
        if isinstance(distance_info, np.ndarray) == True or isinstance(distance_info, list) == True:
            trace0 = pm.sample(length, chains = chains, cores = cores, target_accept = target_accept_set, initvals = {'T': init_params[0], 'n_e_log': np.log10(init_params[1]), 'tau_0': init_params[2], 'Kslab_1e6': init_params[3]*1e6, 'Kphot_1e6': init_params[4]*1e6, 'Av': init_params[5], 'Teff': init_params[6], 'distance': d_target})  #do I really need 'distance' included here? for a starting point yes
        else:
            trace0 = pm.sample(length, chains = chains, cores = cores, target_accept = target_accept_set, initvals = {'T': init_params[0], 'n_e_log': np.log10(init_params[1]), 'tau_0': init_params[2], 'Kslab_1e6': init_params[3]*1e6, 'Kphot_1e6': init_params[4]*1e6, 'Av': init_params[5], 'Teff': init_params[6]})
        
        arviz.to_netcdf(trace0, name+'_tracefile')
        
    return trace0

