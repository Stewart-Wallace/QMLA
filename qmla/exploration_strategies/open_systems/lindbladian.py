import numpy as np
import itertools
import sys
import os

from qmla.exploration_strategies import exploration_strategy
import qmla.shared_functionality.probe_set_generation
from qmla import construct_models

from scipy import linalg, sparse
import math
import random


def tensor_expansion(
        binary_matrix,
        operator
        ):
    I = np.identity(2)
   
    # The binary matrix directs the construction of a matrix:
    #   a 1 corresponds to the matrix passed as 'operator'
    #   a 0 corresponds to an identity matrix
    #   the x axis are tensored in order then summed along the y
    for p in range(0,np.shape(binary_matrix)[0]):
        placeholder = np.array([1])
        # This try is incase a line was sent, not a matrix ie. [0,0,1,0]
        # the try deals with the matix case and the except hands lines
        try:
            for q in range(0,np.shape(binary_matrix)[1]):
                # if statement unpacks line in to matrices then tensor prodicts them
                if binary_matrix[p,q] == 0:
                    placeholder = np.kron(placeholder,I)
                else:
                    placeholder = np.kron(placeholder,operator)
            # each line is then summed into fullmatrix
            if p == 0:
                fullmatrix = placeholder
            else:
                fullmatrix = fullmatrix + placeholder
        except:
            for q in binary_matrix:
                if q == 0:
                    placeholder = np.kron(placeholder,I)
                else:
                    placeholder = np.kron(placeholder,operator)
            # no need to sum here so break    
            fullmatrix = placeholder
            break
    return fullmatrix

   

def liouvillian_evolve_expectation(
    ham,
    t,
    state,
    log_file='QMDLog.log',
    log_identifier='Expecation Value',
    **kwargs
):
    experimental_t_tolerance = 10
   
    try:
        unitary = linalg.expm(ham * t)
        u_psi = np.dot(unitary, state)
    except:
        print(
            #[
                "Failed to build unitary for ham:\n {}".format(ham)
            #],
            #log_file=log_file, log_identifier=log_identifier
        )
        raise
    true_model_string = 'LiouvillianHam_lx_1_d1+LiouvillianHam_la_1_d1+LiouvillianDiss_lb_1_d1+LiouvillianDiss_lc_1_d1'  
    true_components = true_model_string.split('+')
    true_model = np.zeros((ham.shape))
    true_model_terms_params = {
            'LiouvillianHam_lx_1_d1' : 3,
            'LiouvillianHam_la_1_d1' : 1.0034,
            'LiouvillianDiss_lb_1_d1': 0.3523,      
            'LiouvillianDiss_lc_1_d1': 0.6667,
        }
    for ii in true_components:
        true_model = true_model+ qmla.construct_models.compute(ii)*true_model_terms_params[ii]
       
    N = int(np.sqrt(len(state)))
   
    rho1 = np.zeros((N,N),complex)
    rho2 = np.zeros((N,N),complex)
   
    for c,val in enumerate(state):
        rho1[math.floor(c/N),c%N] = val
       
    for c,val in enumerate(u_psi):
        rho2[math.floor(c/N),c%N] = val  
       
       
    if np.array_equal(true_model, ham) :
        #bloch evolution
        yzz = 0.17    #dissipates signal
        Szz = 0.033459  #frequency
        yzz_ = 0.15  #lower level dissipation
        Szz_ = 0.09834  #exponential growth
        Sz0 = -0.0577822   #effects steady state and frequency
        O = 0.35480    #freuqncy and dissipation
        d = 0.04482     #signal growth
        n = np.sqrt(d**2+O**2)
        Gamma = 0.25*(yzz+yzz_)
        kappa = 0.25*(yzz-yzz_)
        lamdaw1 = 0.5*(Szz - Szz_)
        squiggle = 0.5*(Szz+Szz_)
        lamdaw2 = Sz0
        O_dash = O +O*lamdaw1/n
        d_dash = d + lamdaw2
        Times = np.linspace(0,t,5000)
        dt = t/5000
        M = np.array([[-Gamma*(O**2/n**2),-1*d_dash,-Gamma*(d*O**2/n**2)],
                      [d_dash,-Gamma*(O**2/n**2),-1*O_dash],        
                      [0,O,0]])
        b = np.array([[-O*kappa/n],[d*O*(lamdaw2-squiggle)/n**2],[0]])
       
        alpha = np.array([[np.trace(np.dot(rho1,np.array([[0,1],[1,0]])))],[np.trace(np.dot(rho1,np.array([[0,-1j],[1j,0]])))],[np.trace(np.dot(rho1,np.array([[1,0],[0,-1]])))]])
        alpha_end = alpha
        for kk in Times:
            alpha_end = alpha_end+ dt*(np.dot(M,alpha_end)+b)
        expectation_value = np.array([0.5*(1+complex(alpha_end[0]).real)])
        #print(expectation_value)  
    else:
       
        ex_val_tol = 1e-9
       
        if (np.trace(rho1) > 1 + ex_val_tol
            or
            np.trace(rho1) < 0 - ex_val_tol
        ):
            print('rho1 has a trace: ', np.trace(rho1))
           
        if ((np.trace(rho2) > 1 + ex_val_tol
            or
            np.trace(rho2) < 0 - ex_val_tol)
        ):
            print('rho2 is bad rho2,rho2,ham,t: ', rho2, rho1, ham, t)
           
# =============================================================================
#     #Fidelity
#     intermediate_calc_1 = np.dot(linalg.sqrtm(rho1),rho2)
#     intermediate_calc_2 = np.dot(intermediate_calc_1,linalg.sqrtm(rho1))
#     expectation_value_Fid = np.trace(linalg.sqrtm(intermediate_calc_2))**2
#    
#     #Alt Trace Distance
#     intermediate_calc_1 = np.dot((rho1-rho2).conj().T,(rho1-rho2))
#     intermediate_calc_2 = np.trace(linalg.sqrtm(intermediate_calc_1))
#     expectation_value_Tr = 0.5*intermediate_calc_2
#    
# =============================================================================
    #Population -- Operator
        op = tensor_expansion(np.identity(int(np.log2(len(rho2)))), np.array([[0,1],[1,0]]))
        expectation_value_Op = 0.5*(1+np.trace(np.dot(rho2,op)))
        #Expectation Value Choice
        expectation_value = expectation_value_Op

   
# =============================================================================
#     if expectation_value.imag> 0.0001:
#         print('rho2 is bad rho2,rho2,ham,t: ', rho2, rho1, ham, t)
#     print(rho1)
#     if (expectation_value > 1 + ex_val_tol
#         or
#         expectation_value < 0 - ex_val_tol
#     ):
#         print('Fidelity between an initial density matrix and evolved matrix is:  ', expectation_value,'t:', t)
#         #print('rho2:', rho2, 'ham:', ham,'rho1:',rho1)
#        
# =============================================================================

   
       
    #print('Stewart: experimentation result',rho1,rho2, expectation_value)
    return expectation_value


#liouvillian_evolve_expectation(qmla.construct_models.compute('LiouvillianHam_lx_1_d1'),15,np.array([0,0,0,1]))
   
def random_qubit():
    #Created a vector of values that when squared and summed = 1
    #Vector is 2 long thus can be seen as a normalised qubit
   
        #Initially sets first element of vector
    random_num_1 = random.random()
       
        #Calculates second element from first
    random_num_2 = np.sqrt(1-random_num_1**2)
   
    #Tests that qubit is valid
       
    ex_val_tol = 1e-9
    if random_num_1**2+random_num_2**2 < 1 - ex_val_tol:     #Checks qubit is normal within allowed machine inaccuracies
        print('norm of qubit is not maintained', random_num_1**2+random_num_2**2)
           
            #Re-runs if invalid qubit        
        random_qubit()
    else:
           
            #If valid then builds vector and returns.
        mtrx = np.zeros((1,2))
        mtrx[(0,0)] =random_num_1
        mtrx[(0,1)] = random_num_2
        return mtrx
   
def liouv_separable_probe_dict(     #This may be wrong depending on what scheme of vectorisation we are to use.
    max_num_qubits,
    num_probes,
    **kwargs
):
        #Set paramaters for probes creation
    mixed_state = True
    ex_val_tol = 1e-9
    N = int(np.log2(max_num_qubits))
   
        #Initialises Dictionary
    separable_probes = {}
   
    for qq in range(num_probes):                #Iterates to create correct number of probes
           
            #Calls random_qubit() to get a vector of random values that squared and summed = 1
        state = random_qubit()
       
        for pp in range(1,N+1):                 #Iterates to add additional qubits to increase probe system size
            if pp == 1:

                    #Passing single qubit
                state = state
            else:
                    #Increasing the Hilbert space of state with another random qubit.
                state = np.kron(state,random_qubit())
               
            if mixed_state:                      #Applying alteration if probe is to be mixed.
                mtx = np.diag(np.diag(np.dot(state.T,state)))
               
                #Building Density Matrix from state (MAY NEED CHANGED)
            mtx = np.dot(state.T,state)
           
                #Checks that trace is valid
            if (np.trace(mtx) > 1 + ex_val_tol
                or
                np.trace(mtx) < 0 - ex_val_tol
            ):
                print('The following Matrix does not have a valid trace:', mtx)
               
                #Flattens Densiy Matrix into Superket and returns
            separable_probes[qq,pp*2] = mtx.flatten()  
    return separable_probes

def liouville_latex(
    name,
    **kwargs
):
   
    terms = name.split('+')
    ASCII_val = 65
    full_name = ''
    #Need to add sorter to get all ham terms first then Diss.
    for pp in terms:
        #print(pp.split('_')[0])
       
        pp = pp.split('_')
        #print(pp)
        if pp[0] == 'LiouvillianHam':
            pp.remove('LiouvillianHam')
            sites = pp[1:-1]
            if len(pp[0]) == 2:        #single-operator
                op = pp[0].replace('l','')
                op = op.replace('s','-').replace('a','+')  
                term_latex_name = '$' + chr(ASCII_val) + '_{ham} = \sum_{i = '
                for ss in sites:
                    if len(ss) == 3:
                        term_latex_name = term_latex_name + '(' + ss.replace('J',',') + '),'
                    else:
                        term_latex_name = term_latex_name + ss + ','
                term_latex_name = term_latex_name[0:-1] +'}\\sigma_{(i)}^{' + op + '}$'
            else:                   #multi-operator
                op = pp[0].replace('J',', ')
                op = op.replace('s','-').replace('a','+')  
                if len(sites) > 1:
                    term_latex_name = '$' + chr(ASCII_val) + '_{ham} = \sum_{(i,j) = '
                    for ss in sites:
                         term_latex_name = term_latex_name + '(' + ss.replace('J',', ') + '), '
                    term_latex_name = term_latex_name[0:-2] + '}\\sigma_{(i,j)}^{' + op + '}$'
                   
                else:
                    term_latex_name = ('$' + chr(ASCII_val) + '_{ham} = ' +
                                       '\\sigma_{' + sites[0][0] +'}^{' + op[0] + '}' +
                                       '\\sigma_{' + sites[0][-1] +'}^{' + op[-1] + '}$')
        elif pp[0] == 'LiouvillianDiss':
            pp.remove('LiouvillianDiss')
            sites = pp[1:-1]
            if len(pp[0]) == 2:        #single-operator
                op = pp[0].replace('l','')
                op = op.replace('s','-').replace('a','+')  
                term_latex_name = '$' + chr(ASCII_val) + '_{diss} = \sum_{i = '
                for ss in sites:
                    if len(ss) == 3:
                        term_latex_name = term_latex_name + '(' + ss.replace('J',',') + '),'
                    else:
                        term_latex_name = term_latex_name + ss + ','
                term_latex_name = term_latex_name[0:-1] +'}\\sigma_{(i)}^{' + op + '}$'
            else:                   #multi-operator
                op = pp[0].replace('J',', ')
                op = op.replace('s','-').replace('a','+')  
                if len(sites) > 1:
                    term_latex_name = '$' + chr(ASCII_val) + '_{diss} = \sum_{(i,j) = '
                    for ss in sites:
                         term_latex_name = term_latex_name + '(' + ss.replace('J',', ') + '), '
                    term_latex_name = term_latex_name[0:-2] + '}\\sigma_{(i,j)}^{' + op + '}$'
                   
                else:
                    term_latex_name = ('$' + chr(ASCII_val) + '_{diss} = ' +
                                       '\\sigma_{' + sites[0][0] +'}^{' + op[0] + '}' +
                                       '\\sigma_{' + sites[0][-1] +'}^{' + op[-1] + '}$')
        else:
            print(pp[0])
            raise ValueError("Liouvillian Decleration was not a Hamiltonian or a Dissipative term.")
        ASCII_val = ASCII_val + 1
        if full_name != '':
            full_name = full_name[0:-1] +' , ' + term_latex_name[1:]
        else:
            full_name = full_name +  term_latex_name
         
    return full_name

#hamtest = ['LiouvillianHam_lx_1_2_d2' , 'LiouvillianHam_lx_1J2_d2' , 'LiouvillianHam_lx_1J2_2J4_d4'
#           , 'LiouvillianHam_xJy_1J2_d2' , 'LiouvillianHam_xJz_1J2_3J4_d4' , 'LiouvillianHam_lx_1_2_d2+LiouvillianHam_lx_1J2_d2']

#for name in hamtest:
#    print(liouville_latex(name))


def plot_probe(
    max_num_qubits,
    **kwargs
):
        probe_dict = {}
        num_probes = kwargs['num_probes']
        for i in range(num_probes):
            for l in range(1, 1 + max_num_qubits):
                vector_size = np.zeros(l**2)
                vector_size[0] = 1
                probe_dict[(i,l)] = vector_size
        return probe_dict
 
# =============================================================================
# hi = qmla.construct_models.compute('LiouvillianHam_lx_1_d1+LiouvillianDiss_ls_1_d1')
# hey = liouv_separable_probe_dict(2, 10)
# bro = []
# for ii in np.linspace(0,2,100):
#     yo = liouvillian_evolve_expectation(hi, ii, hey[(0,2)])[0]
#     bro.append(liouvillian_evolve_expectation(hi, ii, hey[(0,2)])[1])
#    
# =============================================================================
class Lindbladian(
    exploration_strategy.ExplorationStrategy
):
    def __init__(
        self,
        exploration_rules,
        **kwargs
    ):
        # print("[Exploration Strategies] init nv_spin_experiment_full_tree")
        super().__init__(
            exploration_rules=exploration_rules,
            **kwargs
        )
        self.qinfer_resampler_threshold = 0.5
        self.max_time_to_consider = 100
        # edit time of experimentation
        self.model_heuristic_subroutine = qmla.shared_functionality.experiment_design_heuristics.MixedMultiParticleLinspaceHeuristic
        # self.qinfer_model_class = qmla.shared_functionality.qinfer_model_interface.QInferLiouvilleExperiment
       
       
        self.exclude_evaluation = True
        self.log_print(["initialising new GR"])
        self.latex_string_map_subroutine = liouville_latex
        self.expectation_value_subroutine = liouvillian_evolve_expectation
       
        qmla.construct_models.core_operator_dict['a'] = np.array([[0,0],[0,1]])
        qmla.construct_models.core_operator_dict['b'] = np.array([[1,0.3741966805226849],[0.41421356237309503,-1]],complex)
        qmla.construct_models.core_operator_dict['c'] = np.array([[1,0.3741966805226849],[0.41421356237309503,-1]],complex).T
        self.initial_models = ['LiouvillianHam_lx_1_d1+LiouvillianHam_la_1_d1+LiouvillianDiss_lb_1_d1+LiouvillianDiss_lc_1_d1' ]
        self.tree_completed_initially = True
        #self.qinfer_resampler_a = 0.98
        #self.qinfer_resampler_threshold = 0.5
        self.true_model = 'LiouvillianHam_lx_1_d1+LiouvillianHam_la_1_d1+LiouvillianDiss_lb_1_d1+LiouvillianDiss_lc_1_d1'  
        #print(self.true_model)
        self.plot_probe_generation_function = plot_probe
        self.true_model = construct_models.alph(self.true_model)
        self.true_model_terms_params = {
            'LiouvillianHam_lx_1_d1' : 0.66322,
            'LiouvillianHam_la_1_d1' : 0.73424,
            'LiouvillianDiss_lb_1_d1': 0.01523,      
            'LiouvillianDiss_lc_1_d1': 0.13678,
        }
       
       
        #qmla.construct_models.compute('LiouvillianHam_lx_1J2_d2+LiouvillianDiss_ls_1_2_d2')
        self.probe_generation_function = liouv_separable_probe_dict
       
   
       
# =============================================================================
#         #self.initial_models = ['pauliLikewise_lx_1J2_d3+pauliLikewise_lx_2J3_d3' ]
#         self.latex_model_naming_function = qmla.shared_functionality.latex_model_names.basic_latex_name        
#         self.tree_completed_initially = False
#         self.true_model = 'pauliLikewise_lx_1J2_d3+pauliLikewise_lx_2J3_d3'  
#         self.true_model = construct_models.alph(self.true_model)
#         self.true_model_terms_params = {
#             'pauliLikewise_lx_1J2_d3' : 0.23985,
#             'pauliLikewise_lx_2J3_d3' : 0.42178
#         }
#         self.gaussian_prior_means_and_widths = {
#             'pauliLikewise_lx_1J2_d3' : (0.3,0.2),
#             'pauliLikewise_lx_2J3_d3' : (0.5,0.3)
#
#         }
# =============================================================================
       
# =============================================================================
# mtx0 = np.array([[0.6359,0.4812],[0.4812,0.3641]])
# L = np.array([[0.2975,0+0.4634j,0-0.4634j,0],
# [0+0.4634j,0.1487,0,0-0.4634j],
# [0-0.4634j,0,0.1487,0+0.4634j],
# [-0.2975,0-0.4634j,0+0.4634j,0]])
# times1 = np.linspace(0,200,1000)
# traces = []
# for qq in range(1000):
#     unitary = linalg.expm(L * times1[qq])
#    
#     u_psi = np.dot(unitary,mtx0.flatten())
#     N = int(np.sqrt(len(mtx0.flatten())))
#     rho2 = np.zeros((N,N),complex)
#     for c,val in enumerate(u_psi):
#         rho2[math.floor(c/N),(c%N)] = val
#     traces.append(np.trace(rho2))
# import matplotlib.pyplot as plt
# plt.plot(times1,traces)      
# plt.show()
#
# =============================================================================

