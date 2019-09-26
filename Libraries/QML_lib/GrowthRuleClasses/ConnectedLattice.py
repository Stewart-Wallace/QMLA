import numpy as np
import itertools
import sys, os
sys.path.append(os.path.abspath('..'))
import DataBase
import ProbeGeneration
import ModelNames
import ModelGeneration
import SystemTopology
import Heuristics

import SpinProbabilistic
import SuperClassGrowthRule

flatten = lambda l: [item for sublist in l for item in sublist]  # flatten list of lists


class connected_lattice(
    # SpinProbabilistic.spin_probabilistic
    SuperClassGrowthRule.growth_rule_super_class    
):

    def __init__(
        self, 
        growth_generation_rule, 
        **kwargs
    ):
        # print("[Growth Rules] init nv_spin_experiment_full_tree")
        super().__init__(
            growth_generation_rule = growth_generation_rule,
            **kwargs
        )
        self.heuristic_function = Heuristics.one_over_sigma_then_linspace
        self.lattice_dimension = 2
        self.initial_num_sites = 2
        self.lattice_connectivity_max_distance = 1
        self.lattice_connectivity_linear_only = True
        self.lattice_full_connectivity = False

        self.true_operator = 'pauliSet_xJx_1J2_d2PPpauliSet_yJy_1J2_d2'
        self.true_operator = DataBase.alph(self.true_operator)
        self.qhl_models = [self.true_operator]
        self.base_terms = [
            'x', 
            'y', 
            'z'
        ]

        # fitness calculation parameters. fitness calculation inherited.
        self.num_top_models_to_build_on = 1 # 'all' # at each generation Badassness parameter
        self.model_generation_strictness = 0 #1 #-1 
        self.fitness_win_ratio_exponent = 3

        self.generation_DAG = 1
        self.max_num_sites = 4
        self.tree_completed_initially = False
        self.num_processes_to_parallelise_over = 10
        self.max_num_parameter_estimate = 9
        self.max_num_qubits = 4
        self.max_num_models_by_shape = {
            'other' : 10
        }


        self.setup_growth_class()

    def setup_growth_class(self):
        self.max_num_generations = (
            self.max_num_sites - 
            self.initial_num_sites + 
            self.generation_DAG
        )
        self.topology = SystemTopology.topology_grid(
            dimension = self.lattice_dimension,
            num_sites = self.initial_num_sites,
            maximum_connection_distance = self.lattice_connectivity_max_distance, # nearest neighbours only, 
            linear_connections_only = self.lattice_connectivity_linear_only, 
            all_sites_connected = self.lattice_full_connectivity, 
        )
        self.initially_connected_sites = self.topology.get_connected_site_list()

        self.initial_models = self.generate_terms_from_new_site(
            connected_sites = self.initially_connected_sites, 
            base_terms = self.base_terms, 
            num_sites = self.topology.num_sites() 
        )
        self.true_operator = DataBase.alph(self.true_operator)
        self.model_fitness = {}
        self.models_rejected = {
            self.generation_DAG : []
        }
        self.models_accepted = {
            self.generation_DAG : []
        }

        self.spawn_stage = [None]
        self.available_mods_by_generation = {}
        self.available_mods_by_generation[self.generation_DAG] = self.generate_terms_from_new_site(
            connected_sites = self.initially_connected_sites, 
            base_terms = self.base_terms, 
            num_sites = self.topology.num_sites() 
        )
        self.site_connections_considered = self.initially_connected_sites
        self.max_num_sub_generations_per_generation = {
            self.generation_DAG : len(self.available_mods_by_generation[self.generation_DAG])
        }
        self.models_to_build_on = {
            self.generation_DAG : {}
        }
        self.generation_champs = {
            self.generation_DAG : {}
        }
        self.sub_generation_idx = 0 
        self.counter =0




    @property
    def num_sites(self):
        return self.topology.num_sites
    

    def generate_models(
        self, 
        model_list, 
        **kwargs
    ):
        """
        new models are generated for different cases:
            * within dimension: add term from available term list until exhausted (greedy)
            * finalise dimension: return champions of branches within this dimension to determine
                which model(s) to build on for next generation
            * new dimension: add site to topology; get available term list; return model list 
                of previous champ(s) plus each of newest terms
            * finalise QMD: return generation champs 
                (here generation directly corresponds to number of sites)
        cases are indicated by self.spawn_stage
        """
        print("[Connected Lattice] Generate Models fnc")

        # fitness = kwargs['fitness_parameters']
        model_points = kwargs['branch_model_points']
        branch_models = list(model_points.keys())
        # keep track of generation_DAG
        ranked_model_list = sorted(
            model_points, 
            key=model_points.get, 
            reverse=True
        )
        if self.num_top_models_to_build_on == 'all':
            models_to_build_on = ranked_model_list
        else:
            models_to_build_on = ranked_model_list[:self.num_top_models_to_build_on]
        self.sub_generation_idx += 1 
        self.models_to_build_on[self.generation_DAG][self.sub_generation_idx] = models_to_build_on
        # self.generation_champs[self.generation_DAG][self.sub_generation_idx] = models_to_build_on
        self.generation_champs[self.generation_DAG][self.sub_generation_idx] = [
            kwargs['model_names_ids'][models_to_build_on[0]]
        ]

        self.counter+=1
        new_models = []

        if self.spawn_stage[-1] == None:
            # within dimension; just add each term in available terms to 
            # old models (probabilistically). 

            if self.sub_generation_idx == self.max_num_sub_generations_per_generation[self.generation_DAG]:
                # give back champs from this generation and indicate to make new generation
                self.log_print(
                    [
                        "exhausted this generation.",
                        "\ngeneration champs:", 
                        self.generation_champs[self.generation_DAG]
                    ]
                )
                self.spawn_stage.append('make_new_generation')
                new_models = [
                    self.generation_champs[self.generation_DAG][k] for k in 
                    list(self.generation_champs[self.generation_DAG].keys())
                ]
                new_models = flatten(new_models)
                self.log_print(
                    [
                        "new mods:", new_models
                    ]
                )

                if self.generation_DAG == self.max_num_generations:
                    # this was the final generation to learn.
                    # instead of building new generation, skip straight to Complete stage
                    self.spawn_stage.append('Complete')

            else:
                for mod_id in self.models_to_build_on[self.generation_DAG][self.sub_generation_idx]:
                    mod_name = kwargs['model_names_ids'][mod_id]

                    present_terms = DataBase.get_constituent_names_from_name(mod_name)
                    possible_new_terms = list(
                        set(
                            self.available_mods_by_generation[self.generation_DAG]
                        )
                        - set(present_terms)
                    )

                    self.model_fitness_calculation(
                        model_id = mod_id,
                        # fitness_parameters = fitness[mod_id],
                        model_points = model_points
                    )
                    
                    num_sites_this_mod = DataBase.get_num_qubits(mod_name)
                    target_num_sites = num_sites_this_mod
                    p_str = 'P'*target_num_sites
                    # new_num_qubits = num_qubits + 1
                    # mod_name_increased_dim = increase_dimension_pauli_set(mod_name) 
                    for new_term in possible_new_terms: 
                        new_mod = str(
                            mod_name + 
                            p_str +
                            new_term
                        )
                        new_mod = DataBase.alph(new_mod)
                        if self.determine_whether_to_include_model(mod_id) == True:
                            new_models.append(new_mod)
                            self.models_accepted[self.generation_DAG].append(new_mod)
                        else:
                            self.models_rejected[self.generation_DAG].append(new_mod)
        elif self.spawn_stage[-1] == 'make_new_generation':
            self.generation_DAG += 1
            self.sub_generation_idx = 0 

            self.models_to_build_on = {
                self.generation_DAG : {}
            }
            self.generation_champs = {
                self.generation_DAG : {}
            }
            self.models_rejected = {
                self.generation_DAG : []
            }
            self.models_accepted = {
                self.generation_DAG : []
            }
            self.topology.add_site()
            # nearest_neighbours = self.topology.get_nearest_neighbour_list()
            # new_connections = list(
            #     set(nearest_neighbours) - set(self.site_connections_considered)
            # )
            new_connections = self.topology.new_connections[-1]
            self.site_connections_considered.extend(new_connections)
            possible_new_terms = self.generate_terms_from_new_site(
                connected_sites = new_connections, 
                base_terms = self.base_terms,
                num_sites = self.topology.num_sites()
            )
            self.log_print(
                [
                    "Making generation ",  self.generation_DAG,
                    "\nNew connections:", new_connections, 
                    "\nPossible new terms:", possible_new_terms
                ]
            )
            self.available_mods_by_generation[self.generation_DAG] = possible_new_terms
            self.max_num_sub_generations_per_generation[self.generation_DAG] = len(possible_new_terms)

            for mod_id in models_to_build_on:
                new_num_sites = self.topology.num_sites()
                mod_name = kwargs['model_names_ids'][mod_id]
                mod_name = SpinProbabilistic.increase_dimension_pauli_set(
                    mod_name,
                    new_dimension = new_num_sites
                )

                self.model_fitness_calculation(
                    model_id = mod_id,
                    # fitness_parameters = fitness[mod_id],
                    model_points = model_points
                )

                p_str = 'P'*new_num_sites
                for new_term in possible_new_terms: 
                    new_mod = str(
                        mod_name + 
                        p_str +
                        new_term
                    )
                    if self.determine_whether_to_include_model(mod_id) == True:
                        new_models.append(new_mod)
                        self.models_accepted[self.generation_DAG].append(new_mod)
                    else:
                        self.models_rejected[self.generation_DAG].append(new_mod)

            self.spawn_stage.append(None)
            # if self.max_num_sub_generations_per_generation[self.generation_DAG] == 1:
            #     self.spawn_stage.append('make_new_generation')



        elif self.spawn_stage[-1] == 'Complete':
            # return list of generation champs to determine final winner
            champs_all_generations = []
            for gen_idx in list(self.generation_champs.keys()): 
                sub_indices = list(self.generation_champs[gen_idx].keys())
                max_sub_idx = max(sub_indices)
                champ_this_generation =  self.generation_champs[gen_idx][max_sub_idx]
                champs_all_generations.append(champ_this_generation)
                new_models = champ_this_generation
            self.log_print(
                [
                    "Model generation complete.", 
                    "returning list of champions to determine global champion:",
                    new_models
                ]
            )


        elif self.spawn_stage[-1] == 'Complete':
            return model_list
        new_models = list(set(new_models))
        self.log_print(
            ["New models:", new_models]
        )
        return new_models

    def latex_name(
        self, 
        name, 
        **kwargs
    ):
        # print("[latex name fnc] name:", name)
        core_operators = list(sorted(DataBase.core_operator_dict.keys()))
        num_sites = DataBase.get_num_qubits(name)
        p_str = 'P'*num_sites
        separate_terms = name.split(p_str)

        site_connections = {}
        for c in list(itertools.combinations(list(range(num_sites+1)), 2)):
            site_connections[c] = []

        term_type_markers = ['pauliSet', 'transverse']
        for term in separate_terms:
            components = term.split('_')
            if 'pauliSet' in components:
                components.remove('pauliSet')

                for l in components:
                    if l[0] == 'd':
                        dim = int(l.replace('d', ''))
                    elif l[0] in core_operators:
                        operators = l.split('J')
                    else:
                        sites = l.split('J')
                sites = tuple([int(a) for a in sites])
                op = operators[0] # assumes like-like pauli terms like xx, yy, zz
                site_connections[sites].append(op)

        ordered_connections = list(sorted(site_connections.keys()))
        latex_term = ""

        for c in ordered_connections:
            if len(site_connections[c]) > 0:
                this_term = "\sigma_{"
                this_term += str(c)
                this_term += "}"
                this_term += "^{"
                for t in site_connections[c]:
                    this_term += "{}".format(t)
                this_term += "}"
                latex_term += this_term
        latex_term = "${}$".format(latex_term)
        return latex_term

    def generate_terms_from_new_site(
        self, 
        base_terms, 
        connected_sites,
        num_sites 
    ):

        return pauli_like_like_terms_connected_sites(
            connected_sites = connected_sites, 
            base_terms = base_terms, 
            num_sites = num_sites
        )


    def model_fitness_calculation(
        self, 
        model_id, 
        # fitness_parameters, # of this model_id
        model_points, 
        **kwargs
    ):
        # TODO make fitness parameters within QMD 
        # pass 
        # print("model fitness function. fitness params:", fitness_parameters)
        # print("[prob spin] model fitness. model points:", model_points)
        ranked_model_list = sorted(
            model_points, 
            key=model_points.get, 
            reverse=True
        )

        try:
            max_wins_model_points = max(model_points.values())
            win_ratio = model_points[model_id] / max_wins_model_points
        except:
            win_ratio = 1

        if self.model_generation_strictness == 0:
            # keep all models and work out relative fitness
            fitness = (
                win_ratio
                # win_ratio * fitness_parameters['r_squared']
            )**self.fitness_win_ratio_exponent
            # fitness = 1
        elif self.model_generation_strictness == -1:
            fitness = 1
        else:
            # only consider the best model
            # turn off all others
            if model_id == ranked_model_list[0]:
                fitness = 1
            else:
                fitness = 0




        if model_id not in sorted(self.model_fitness.keys()):
            self.model_fitness[model_id] = {}
        # print("Setting fitness for {} to {}".format(model_id, fitness))
        self.model_fitness[model_id][self.generation_DAG] = fitness            


    def determine_whether_to_include_model(
        self, 
        model_id    
    ):
        # biased coin flip
        fitness = self.model_fitness[model_id][self.generation_DAG]
        rand = np.random.rand()
        to_generate = ( rand < fitness ) 
        return to_generate

    def check_tree_completed(
        self,
        spawn_step, 
        **kwargs
    ):
        if self.spawn_stage[-1] == 'Complete':
            return True 
        else:
            return False
        return True

    def name_branch_map(
        self,
        latex_mapping_file, 
        **kwargs
    ):

        import ModelNames
        # TODO get generation idx + sub generation idx

        return ModelNames.branch_is_num_params_and_qubits(
            latex_mapping_file = latex_mapping_file,
            **kwargs
        )






def pauli_like_like_terms_connected_sites(
    connected_sites, 
    base_terms, 
    num_sites
):

    new_terms = []
    for pair in connected_sites:
        site_1 = pair[0]
        site_2 = pair[1]

        acted_on = "{}J{}".format(site_1, site_2)
        for t in base_terms:
            pauli_terms = "{}J{}".format(t, t)
            mod = "pauliSet_{}_{}_d{}".format(acted_on, pauli_terms, num_sites)
            new_terms.append(mod)    
    return new_terms


def possible_pauli_combinations(base_terms, num_sites):
    # possible_terms_tuples = list(itertools.combinations_with_replacement(base_terms, num_sites))
    # possible_terms_tuples = list(itertools.combinations(base_terms, num_sites))
    possible_terms_tuples = [
        (a,)*num_sites for a in base_terms
    ] # only hyerfine type terms; no transverse


    possible_terms = []

    for term in possible_terms_tuples:
        pauli_terms = 'J'.join(list(term))
        acted_on_sites = [str(i) for i in range(1,num_sites+1) ]
        acted_on = 'J'.join(acted_on_sites)
        mod = "pauliSet_{}_{}_d{}".format(pauli_terms, acted_on, num_sites)

        possible_terms.append(mod)
    return possible_terms

def increase_dimension_pauli_set(initial_model, new_dimension=None):
    print("[spin prob incr dim] initial model:", initial_model, "new dim:", new_dimension)
    individual_terms = DataBase.get_constituent_names_from_name(initial_model)
    separate_terms = []
    
    for model in individual_terms:
        components = model.split('_')

        for c in components:
            if c[0] == 'd':
                current_dim = int(c.replace('d', ''))
                components.remove(c)

        if new_dimension == None:
            new_dimension = current_dim + 1
        new_component = "d{}".format(new_dimension)
        components.append(new_component)
        new_mod = '_'.join(components)
        separate_terms.append(new_mod)

    p_str = 'P'*(new_dimension)
    full_model = p_str.join(separate_terms)
    
    return full_model
    
    
    
