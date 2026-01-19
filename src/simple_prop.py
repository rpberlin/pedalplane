from xfoil_wrapper import get_qprop_fit_params
from propeller_layout import Propeller
from pint import Quantity as Q_
import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":
    prop_diam = Q_(2.0,'m')
    hub_diam = Q_(0.2,'m') 
    prop_R = 0.5*prop_diam
    hub_R = 0.5*hub_diam
    N_blades =  2
    chord_ref =  Q_(15,'cm') 
    rho_ref = Q_(1.225,'kg/m^3') 
    U_axial = Q_(0,'m/s')
    mu_ref = Q_(1.18e-5,'Pa*s')
    prop_omega = Q_(1000, 'rpm')
    U_tan_ref = prop_omega*prop_R
    U_mag_ref = (U_axial**2 + U_tan_ref**2)**0.5

    

    radius_ratios = [0, .3,  0.4, .6, .8, 0.98] #0 
    chord_ratios =    [0.7, 0.8, .9 , .9, .6, .4] 
    beta_angles =    [45,  28, 24,  16,  13,  8]
    #Re_ref = (rho_ref*U_mag_ref*chord_ref/mu_ref).to('')
    Re_ref = 1e5
    airfoil_names = ['dae11.dat','dae21.dat','eppler387.dat','dae31.dat','dae41.dat','dae51.dat']


    #qprop_param_objs = []
    #for i, airfoil_name in enumerate(airfoil_names):
    #    qprop_param_dict = get_qprop_fit_params(airfoil_name, Re_ref, False)
    #    qprop_param_objs.append(qprop_param_dict)
    #    print(qprop_param_dict)

    prop = Propeller(diameter_m = prop_diam.m_as('m'), hub_diameter_m = hub_diam.m_as('m'), n_blades=N_blades)



    for rr, cr, beta, name in zip(radius_ratios, chord_ratios, beta_angles, airfoil_names):
        fit = get_qprop_fit_params(name, Re_ref, False)
        prop.add_section(
            r_m=(hub_R+rr*(prop_R-hub_R)).m_as('m'),
            chord_m=(cr * chord_ref).m_as('m'),
            beta_deg=beta,
            af_fit=fit,
            airfoil_name=name,   # optional label
        )

    res = prop.evaluate_qprop(rpm=1000.0, U_axial=0.0,keep_dir=False)
    print(res.T_N, res.Q_Nm, res.P_shaft_W)

    thrust_list = []
    torque_list = []
    power_list = []
    thrust_list_cruise = []
    torque_list_cruise = []
    power_list_cruise = []
    adv_list =[]

    rpm_list = np.linspace(100,600,10)
    for rpm in rpm_list:
        res = prop.evaluate_qprop(rpm=rpm, U_axial=0, keep_dir=False)
        res_cruise = prop.evaluate_qprop(rpm=rpm, U_axial=4, keep_dir=False)
        thrust, torque, power = res.summary['T_N'], res.summary['Q_N_m'], res.summary['Pshaft_W']
        thrust_list.append(thrust)
        torque_list.append(torque)
        power_list.append(power)
        thrust_list_cruise.append(res_cruise.summary['T_N'])
        torque_list_cruise.append(res_cruise.summary['Q_N_m'])
        power_list_cruise.append(res_cruise.summary['Pshaft_W'])

    speed_list = np.linspace(0,5,10)
    Pprop_list = []
    prop_eff_list = []
    for speed in speed_list:
        res = prop.evaluate_qprop(rpm=240,U_axial=speed,keep_dir=False)
        Pprop = res.summary['Pprop']
        prop_eff = res.summary['eff']
        adv_list.append(res.summary['adv'])
        Pprop_list.append(Pprop)
        prop_eff_list.append(prop_eff)

    plt.subplot(2,3,1)
    plt.plot(rpm_list, thrust_list,'b',label='static')
    plt.plot(rpm_list, thrust_list_cruise,'r',label='cruise')
    plt.xlabel('RPM (rev/min)')
    plt.ylabel('Thrust (N)')
    plt.title('Thrust vs RPM')
    plt.legend()
    plt.subplot(2,3,2)
    plt.plot(rpm_list, torque_list,'b',label='static')
    plt.plot(rpm_list, torque_list_cruise,'r',label='cruise')
    plt.xlabel('RPM (rev/min)')
    plt.ylabel('Torque (N-m)')
    plt.title('Torque vs RPM')
    plt.legend()
    plt.subplot(2,3,3)
    plt.plot(rpm_list, power_list,'b',label='static')
    plt.plot(rpm_list, power_list_cruise,'r',label='cruise')
    plt.xlabel('RPM (rev/min)')
    plt.ylabel('Shaft Power (W)')
    plt.title('Shaft Power vs RPM')
    plt.legend()
    plt.subplot(2,3,4)
    plt.plot(speed_list, Pprop_list,'k')
    plt.xlabel('Axial Speed (m/s)')
    plt.ylabel('Propulsive Power (W)')
    plt.title('Propulsive Power vs Speed')
    plt.subplot(2,3,5)
    plt.plot(speed_list, prop_eff_list,'g')
    plt.xlabel('Axial Speed (m/s)')
    plt.ylabel('Propulsive Efficiency (-)')
    plt.title('Propulsive Efficiency vs Speed')
    plt.subplot(2,3,6)
    plt.plot(adv_list, prop_eff_list,'m')
    plt.xlabel('Advance Ratio (-)')
    plt.ylabel('Propulsive Efficiency (-)')
    plt.title('Propulsive Efficiency vs Advance Ratio')
    plt.show()

    radial_plotlist = ['beta','chord','Cl','Cd','effp','Mach']
    color_key = ['r','g','b','k','m', 'c']
    for i, plt_qty in enumerate(radial_plotlist):
        plt.subplot(1,len(radial_plotlist),i+1)
        plt.plot(res.radial['radius'],res.radial[plt_qty],color_key[i])
        if 'eff' in plt_qty:
            plt.ylim([0,1])
        plt.xlabel('Radial Position (m)')
        plt.ylabel(plt_qty)
        plt.title(plt_qty)
    plt.show()
    
    print(thrust_list,torque_list,power_list)


    res0 = prop.evaluate_qprop(rpm=240,U_axial=speed,keep_dir=False)
    delta_beta = 1
    deffp_dbeta_list = []
    dThrust_dbeta = []

    for i in range(0,len(beta_angles)):
        beta_shift = [0]*len(beta_angles)
        beta_shift[i] = delta_beta
        tmp_betas = np.array(beta_angles)+np.array(beta_shift)
        prop.set_beta_angles(tmp_betas)
        res = prop.evaluate_qprop(rpm=240,U_axial=speed,keep_dir=False)
        deffp_dbeta_list.append(res.summary['effprop']-res0.summary['effprop'])
        dThrust_dbeta.append(res.summary['T_N']-res0.summary['T_N'])
    
    for i, section in enumerate(prop.sections):
        section.beta_deg = beta_angles[i]

    effp_linesearch = []
    thrust_linesearch = []
    shaftpower_linesearch = []
    best_effprop = res0.summary['effprop']
    best_i = 0
    best_betashift  = None
    best_res = None
    isweep = range(10,500,10)
    for i in isweep:
        beta_shift = i*np.array(deffp_dbeta_list)
        tmp_betas = np.array(beta_angles)+np.array(beta_shift)
        for j, section in enumerate(prop.sections):
            section.beta_deg = tmp_betas[j]
        res = prop.evaluate_qprop(rpm=240,U_axial=speed,keep_dir=False)
        tmp_effprop = res.summary['effprop']
        tmp_thrust = res.summary['T_N']
        tmp_shaftpower = res.summary['Q_N_m']
        effp_linesearch.append(tmp_effprop)
        thrust_linesearch.append(tmp_thrust)
        shaftpower_linesearch.append(tmp_shaftpower)
        if tmp_effprop > best_effprop:
            best_effprop =  tmp_effprop
            best_i = i
            best_betashift = beta_shift
            best_res = res
            best_thrust = tmp_thrust
            best_shaftpower = tmp_shaftpower


    print(deffp_dbeta_list,dThrust_dbeta)
    #plt.plot(radius_ratios,dThrust_dbeta)
    plt.subplot(1,5,1)
    plt.plot(isweep,effp_linesearch)
    plt.plot(best_i, best_effprop,'ro',label='opt')
    plt.ylabel('Propulsive Efficiency')
    plt.xlabel('Iteration (-)')
    plt.title('Optimizer Progress')
    plt.subplot(1,5,2)
    plt.plot(isweep,thrust_linesearch)
    plt.plot(best_i, best_thrust,'ro',label='opt')
    plt.ylabel('Thrust (N)')
    plt.xlabel('Iteration (-)')
    plt.subplot(1,5,3)
    plt.plot(isweep,shaftpower_linesearch)
    plt.plot(best_i, best_shaftpower,'ro',label='opt')
    plt.ylabel('Shaft Power (W)')
    plt.xlabel('Iteration (-)')
    plt.title('Shaft Power (W)')
    plt.subplot(1,5,4)
    plt.plot(res0.radial['radius'],res0.radial['beta'],label='base')
    plt.plot(best_res.radial['radius'],best_res.radial['beta'],label='opt')
    plt.xlabel('Radial Position (m)')
    plt.ylabel('Local Twist (deg)')
    plt.legend()
    plt.title('Prop Twist Angle')
    plt.subplot(1,5,5)
    plt.plot(res0.radial['radius'],res0.radial['effp'],label='base')
    plt.plot(best_res.radial['radius'],best_res.radial['effp'],label='opt')
    plt.xlabel('Radial Position (m)')
    plt.ylabel('Local Efficiency (-)')
    plt.title('Section Efficiency')
    plt.ylim([0,1])
    plt.legend()
    plt.show()

    for i, plt_qty in enumerate(radial_plotlist):
        plt.subplot(1,len(radial_plotlist),i+1)
        plt.plot(res0.radial['radius'],res0.radial[plt_qty],'r',label='base')
        plt.plot(best_res.radial['radius'],best_res.radial[plt_qty],'b',label='opt')
        if 'eff' in plt_qty:
            plt.ylim([0,1])
        plt.xlabel('Radial Position (m)')
        plt.legend()
        plt.ylabel(plt_qty)
        plt.title(plt_qty)
    plt.show()


    print('this is amazing')











    

        

