import numpy as np
import scipy as sp
import os
import time
import itertools
import copy

import qinfer as qi
import redis
import pickle

import qmla.redis_settings
import qmla.logging
import qmla.get_growth_rule
import qmla.construct_models

pickle.HIGHEST_PROTOCOL = 4

__all__ = [
    'ModelInstanceForComparison'
]


class ModelInstanceForComparison():
    """
    Model instances used for Bayes factor comparisons.

    When Bayes factors are calculated remotely (ie on RQ workers),
    they require infrastructure to do calculations, e.g. QInfer SMCUpdater instances.
    This class captures the minimum required to enable these calculations.
    After learning, important data from :class:`~qmla.ModelInstanceForLearning`
    is stored on the redis database.
    This class unpickles the useful information and generates new
    instances of the updater etc. to use in the comparison calculations.

    If run locally, `qmla_core_info_database` and `learned_model_info`
    can be passed directly to this class, to save unpickling
    data from the redis database.

    :param int model_id: ID of the model to study
    :param qid: ID of the QMLA instance
    :param dict qmla_core_info_database: essential details about the QMLA
        instance needed to learn/compare models.
        If None, this is retrieved instead from the redis database.
    :param dict learned_model_info: result of learning, generated by
        :meth:`~qmla.ModelInstanceForLearning.learned_info_dict`.
    :param str host_name: name of host server on which redis database exists.
    :param int port_number: port number unique to this QMLA instance on redis database
    :param str log_file: path of QMLA instance's log file.

    """

    def __init__(
        self,
        model_id,
        qid,
        qmla_core_info_database=None,
        learned_model_info=None,
        host_name='localhost',
        port_number=6379,
        log_file='QMD_log.log',
    ):
        self.log_file = log_file
        self.qmla_id = qid
        self.model_id = model_id

        # Get essential data
        if qmla_core_info_database is None:
            redis_databases = qmla.redis_settings.get_redis_databases_by_qmla_id(
                host_name,
                port_number,
                qid
            )
            qmla_core_info_database = redis_databases['qmla_core_info_database']
            qmla_core_info_dict = pickle.loads(
                qmla_core_info_database.get('qmla_settings'))
            self.probes_system = pickle.loads(
                qmla_core_info_database['probes_system'])
            self.probes_simulator = pickle.loads(
                qmla_core_info_database['probes_simulator'])
        else:
            qmla_core_info_dict = qmla_core_info_database.get('qmla_settings')
            self.probes_system = qmla_core_info_database['probes_system']
            self.probes_simulator = qmla_core_info_database['probes_simulator']

        self.plot_probes = pickle.load(
            open(qmla_core_info_dict['probes_plot_file'], 'rb')
        )
        self.plots_directory = qmla_core_info_dict['plots_directory']
        self.debug_mode = qmla_core_info_dict['debug_mode']
        self.plot_level = qmla_core_info_dict['plot_level']

        # Assign attributes based on core data
        self.num_experiments = qmla_core_info_dict['num_experiments']
        self.num_particles = qmla_core_info_dict['num_particles']
        self.probe_number = qmla_core_info_dict['num_probes']
        self.true_model_constituent_operators = qmla_core_info_dict['true_oplist']
        self.true_model_params = qmla_core_info_dict['true_model_terms_params']
        self.true_model_name = qmla_core_info_dict['true_name']
        self.true_param_dict = qmla_core_info_dict['true_param_dict']
        self.experimental_measurements = qmla_core_info_dict['experimental_measurements']
        self.experimental_measurement_times = qmla_core_info_dict['experimental_measurement_times']
        self.results_directory = qmla_core_info_dict['results_directory']

        if learned_model_info is None:
            # Get data specific to this model, learned elsewhere and stored on
            # redis database
            try:
                redis_databases = qmla.redis_settings.get_redis_databases_by_qmla_id(
                    host_name,
                    port_number,
                    qid
                )
                learned_models_info_db = redis_databases['learned_models_info_db']
            except BaseException:
                print("Unable to retrieve redis database.")
                raise

            model_id_str = str(float(model_id))
            try:
                learned_model_info = pickle.loads(
                    learned_models_info_db.get(model_id_str),
                    encoding='latin1'
                )
            except BaseException:
                try:
                    learned_model_info = pickle.loads(
                        learned_models_info_db.get(model_id_str)
                    )
                except:
                    self.log_print([
                        "Failed to unload model data for comparison"
                    ])

        # Assign parameters from model learned info, retrieved from database
        self.model_name = learned_model_info['name']
        self.times_learned_over = learned_model_info['times_learned_over']
        self.final_learned_params = learned_model_info['final_learned_params']
        self.growth_rule_of_this_model = learned_model_info['growth_rule_of_this_model']
        self.posterior_marginal = learned_model_info['posterior_marginal']
        self.model_normalization_record = learned_model_info['model_normalization_record']
        self.log_total_likelihood = learned_model_info['log_total_likelihood']
        self.estimated_mean_params = learned_model_info['estimated_mean_params']
        self.qhl_final_param_estimates = learned_model_info['qhl_final_param_estimates']
        self.qhl_final_param_uncertainties = learned_model_info['qhl_final_param_uncertainties']
        self.covariance_mtx_final = learned_model_info['covariance_mtx_final']
        self.expectation_values = learned_model_info['expectation_values']
        self.learned_hamiltonian = learned_model_info['learned_hamiltonian']
        self.track_experiment_parameters = learned_model_info['track_experiment_parameters']
        self.log_print(["Track exp params eg:", self.track_experiment_parameters[0]])


        # Process data from learned info
        if self.model_name == self.true_model_name:
            self.is_true_model = True
            self.log_print(["This is the true model for comparison."])
        else:
            self.is_true_model = False
        op = qmla.construct_models.Operator(self.model_name)
        self.model_terms_matrices = op.constituents_operators
        self.model_terms_parameters_final = np.array(self.final_learned_params)
        self.growth_class = qmla.get_growth_rule.get_growth_generator_class(
            growth_generation_rule=self.growth_rule_of_this_model,
            log_file=self.log_file,
            qmla_id = self.qmla_id,
        )
        self.model_name_latex = self.growth_class.latex_name(self.model_name)

        # New instances of model and updater used by QInfer
        self.log_print(["Getting QInfer model"])
        self.qinfer_model = self.growth_class.qinfer_model(
            model_name=self.model_name,
            modelparams=self.model_terms_parameters_final,
            oplist=self.model_terms_matrices,
            true_oplist=self.true_model_constituent_operators,
            truename=self.true_model_name,
            trueparams=self.true_model_params,
            true_param_dict=self.true_param_dict,
            num_probes=self.probe_number,
            probe_dict=self.probes_system,
            sim_probe_dict=self.probes_simulator,
            growth_generation_rule=self.growth_rule_of_this_model,
            experimental_measurements=self.experimental_measurements,
            experimental_measurement_times=self.experimental_measurement_times,
            qmla_id=self.qmla_id, 
            log_file=self.log_file,
            debug_mode=self.debug_mode,
        )

        # Reconstruct the updater from results of learning
        self.reconstruct_updater = True  # optionally just load it
        if self.reconstruct_updater:


            try:
                posterior_distribution = qi.MultivariateNormalDistribution(
                    self.estimated_mean_params,
                    self.covariance_mtx_final # TODO this can cause problems - some models have singular cov mt
                )
            except:
                self.log_print([
                    "cov mtx is singular in trying to reconstruct SMC updater.\n",
                    self.covariance_mtx_final
                ])

            num_particles_for_bf = max(
                5,
                int(self.growth_class.fraction_particles_for_bf * self.num_particles)
            )  # this allows the growth rule to use less particles for the comparison stage

            self.qinfer_updater = qi.SMCUpdater(
                model=self.qinfer_model,
                n_particles=num_particles_for_bf,
                prior=posterior_distribution,
                resample_thresh=self.growth_class.qinfer_resampler_threshold,
                resampler=qi.LiuWestResampler(
                    a=self.growth_class.qinfer_resampler_a
                ),
            )
            self.qinfer_updater._normalization_record = self.model_normalization_record
        else:
            # Optionally pickle the entire updater
            # (first include updater in ModelInstanceForLearning.learned_info_dict())
            self.qinfer_updater = pickle.loads(
                learned_model_info['updater']
            )

        # Fresh experiment design heuristic
        self.experiment_design_heuristic = self.growth_class.heuristic(
            model_id=self.model_id,
            updater=self.qinfer_updater,
            oplist=self.model_terms_matrices,
            num_experiments=self.num_experiments,
            num_probes=self.probe_number,
            log_file=self.log_file,
            inv_field=[item[0]
                       for item in self.qinfer_model.expparams_dtype[1:]],
            max_time_to_enforce=self.growth_class.max_time_to_consider,
        )

        # Delete extra data now that everything useful is extracted
        del qmla_core_info_dict, learned_model_info

    ##########
    # Section: update for Bayes factor
    ##########

    def update_log_likelihood(
        self, 
        new_times, 
        new_experimental_params, 
    ):
        r"""

        """

        # Reduced normalization record using only experiments to consider
        experiment_id_to_keep = int(
            len(self.qinfer_updater.normalization_record)
            - (self.growth_class.fraction_own_experiments_for_bf * len(self.qinfer_updater.normalization_record) ) 
        )
        self.qinfer_updater._normalization_record = self.qinfer_updater._normalization_record[experiment_id_to_keep:]
        self.bf_times = self.times_learned_over[experiment_id_to_keep:]
        
        # List of opponent's times, possibly shortened
        experiment_id_to_keep = int(
            len(new_times)
            - (self.growth_class.fraction_opponents_experiments_for_bf * len(new_times) ) 
        )

        epoch_id = len(self.times_learned_over)
        experiments_to_update_with = new_experimental_params[experiment_id_to_keep:]
        self.log_print(["Times to update length:", len(experiments_to_update_with)])

        for experiment in experiments_to_update_with:
            # sample from own updater/heuristic so particle is correct shape
            experiment_for_update = self.experiment_design_heuristic(
                epoch_id = epoch_id
            ) 

            # retrieve probe and time used by opponent
            experiment_for_update['probe_id'] = experiment['probe_id'][0]
            exp_time = experiment['t'][0]
            experiment_for_update['t'] = exp_time
            self.bf_times.append(experiment['t'])

            # run experiment
            params_array = np.array([[
                self.true_model_params[:]
            ]])
            self.log_print_debug([
                "BF update epoch ", epoch_id
            ])
            datum = self.qinfer_model.simulate_experiment(
                params_array,
                experiment_for_update,
                repeat=1
            )

            # update qinfer 
            self.qinfer_updater.update(datum, experiment_for_update)

            epoch_id += 1
        
        self.log_print_debug(["BF times:", self.bf_times])
        self.bf_times = qmla.utilities.flatten(self.bf_times)
        return self.qinfer_updater.log_total_likelihood

    ##########
    # Section: Plotting
    ##########

    def plot_dynamics(self, ax, times):
        r"""
        Plot dynamics of this model after its parameter learning stage. 

        :param ax: matplotlib axis to plot on
        :param list times: times against which to plot
        """

        times_not_yet_computed = list(
            set(times) - set(self.expectation_values.keys())
        )
        # n_qubits = qmla.construct_models.get_num_qubits(self.model_name)
        n_qubits = np.log2( np.shape(self.learned_hamiltonian)[0]) # TODO get model num qubits from learned_info
        plot_probe = self.plot_probes[n_qubits]

        for t in times_not_yet_computed:
            self.expectation_values[t] = self.growth_class.expectation_value(
                ham = self.learned_hamiltonian, #TODO, 
                t = t, 
                state = plot_probe # TODO
            )

        ax.plot(
            times, 
            [self.expectation_values[t] for t in times],
            label = "{}: {}".format(self.model_id, self.model_name_latex), 
        )

    ##########
    # Section: Utilities
    ##########

    def log_print(
        self,
        to_print_list,
        log_identifier=None,
    ):
        r"""Wrapper for :func:`~qmla.print_to_log`"""
        if log_identifier is None: 
            log_identifier="ModelForComparison {}".format(self.model_id)
            
        qmla.logging.print_to_log(
            to_print_list=to_print_list,
            log_file=self.log_file,
            log_identifier=log_identifier
        )
        
    def log_print_debug(
        self, 
        to_print_list
    ):
        r"""Log print if global debug_log_print set to True."""

        if self.debug_mode:
            self.log_print(
                to_print_list = to_print_list,
                log_identifier = 'Debug Comparison Model {}'.format(self.model_id)
            )