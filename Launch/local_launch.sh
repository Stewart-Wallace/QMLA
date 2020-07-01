#!/bin/bash

# redis-server

###############
# QMLA run configuration
###############
num_tests=1
exp=5 # number of experiments
prt=10 # number of particles
qhl_test=0 # perform QHL on known (true) model
multiple_qhl=0 # perform QHL for defined list of models.
do_further_qhl=0 # QHL refinement to best performing models 
q_id=0 # can start from other ID if desired


###############
# QMLA settings
###############
use_rq=0
further_qhl_factor=1
further_qhl_num_runs=$num_tests
plots=0
number_best_models_further_qhl=5

###############
# Choose a growth rule This will determine how QMD proceeds. 
# use_alt_growth_rules=1 # note this is redundant locally, currently
###############

# growth_rule='TestSimulatedNVCentre'
# growth_rule='IsingGeneticTest'
# growth_rule='IsingGeneticSingleLayer'
# growth_rule='NVCentreRevivals'
# growth_rule='NVCentreRevivalsSimulated'

# growth_rule='IsingGenetic'
# growth_rule='SimulatedNVCentre'
# growth_rule='ExperimentNVCentreNQubits'
# growth_rule='NVCentreSimulatedShortDynamicsGenticAlgorithm'
# growth_rule='NVCentreExperimentalShortDynamicsGenticAlgorithm'
growth_rule='NVCentreNQubitBath'

# growth_rule='IsingLatticeSet'
# growth_rule='HeisenbergLatticeSet'
# growth_rule='FermiHubbardLatticeSet'
# growth_rule='NVLargeSpinBath'
# growth_rule='GeneticTest'
# growth_rule='Genetic'
# growth_rule='NVExperimentalData'

alt_growth_rules=(
    # 'GeneticTest'
)

growth_rules_command=""
for item in ${alt_growth_rules[*]}
do
    growth_rules_command+=" -agr $item" 
done

###############
# Parameters from here downwards uses the parameters
# defined above to run QMLA. 
# e.g. to create filepaths to use during QMLA.
###############

let max_qmd_id="$num_tests + $q_id"

running_dir="$(pwd)"
day_time=$(date +%b_%d/%H_%M)
this_run_results_folder="$running_dir/Results/$day_time/"
mkdir -p $this_run_results_folder

bayes_csv="$this_run_results_folder/cumulative.csv"
true_expec_path="$this_run_results_folder/system_measurements.p"
prior_pickle_file="$this_run_results_folder/prior.p"
system_parameters_file="$this_run_results_folder/system_parameters.p"
plot_probe_file="$this_run_results_folder/plot_probes.p"
latex_mapping_file="$this_run_results_folder/latex_mapping.txt"
analysis_script="$this_run_results_folder/analyse.sh"
this_log="$this_run_results_folder/qmla.log"
further_qhl_log="$this_run_results_folder/qhl_further.log"
cp $(pwd)/local_launch.sh "$this_run_results_folder/launched_script.txt"
git_commit=$(git rev-parse HEAD)

###############
# First set up parameters/data to be used by all instances of QMD for this run. 
###############

python3 ../scripts/set_qmla_params.py \
    -prt=$prt \
    -true=$system_parameters_file \
    -prior=$prior_pickle_file \
    -probe=$plot_probe_file \
    -ggr=$growth_rule \
    -dir=$this_run_results_folder \
    -log=$this_log \
    -true_expec_path=$true_expec_path \
    $growth_rules_command 

echo "Generated configuration."

###############
# Write analysis script 
# before launch in case run stopped before some instances complete.
###############

echo "
cd $this_run_results_folder
python3 ../../../../scripts/analyse_qmla.py \
    -dir=$this_run_results_folder \
    --bayes_csv=$bayes_csv \
    -log=$this_log \
    -top=$number_best_models_further_qhl \
    -qhl=$qhl_test \
    -fqhl=0 \
    -true_expec=$true_expec_path \
    -ggr=$growth_rule \
    -plot_probes=$plot_probe_file \
    -params=$system_parameters_file \
    -latex=$latex_mapping_file \
    -gs=1

python3 ../../../../scripts/generate_results_pdf.py \
    -t=$num_tests \
    -dir=$this_run_results_folder \
    -p=$prt \
    -e=$exp \
    -log=$this_log \
    -ggr=$growth_rule \
    -run_desc=\"localdevelopemt\" \
    -git_commit=$git_commit \
    -qhl=$qhl_test \
    -mqhl=$multiple_qhl \
    -cb=$bayes_csv \

" > $analysis_script

chmod a+x $analysis_script

###############
# Run instances
###############
for i in `seq 1 $max_qmd_id`;
do
    redis-cli flushall
    let q_id="$q_id+1"
    # python3 -m cProfile -s time \
    python3 \
        ../scripts/implement_qmla.py \
        -qhl=$qhl_test \
        -mqhl=$multiple_qhl \
        -rq=$use_rq \
        -p=$prt \
        -e=$exp \
        -qid=$q_id \
        -log=$this_log \
        -dir=$this_run_results_folder \
        -pt=$plots \
        -pkl=1 \
        -cb=$bayes_csv \
        -prior_path=$prior_pickle_file \
        -true_params_path=$system_parameters_file \
        -true_expec_path=$true_expec_path \
        -plot_probes=$plot_probe_file \
        -latex=$latex_mapping_file \
        -ggr=$growth_rule \
        $growth_rules_command \
        > $this_run_results_folder/output.txt
done

echo "
------ QMLA completed ------
"

###############
# Furhter QHL, optionally
###############

if (( $do_further_qhl == 1 )) 
then
    sh $analysis_script

    further_analyse_filename='analyse_further_qhl.sh'
    further_analysis_script="$this_run_results_folder$further_analyse_filename"
    let particles="$further_qhl_factor * $prt"
    let experiments="$further_qhl_factor * $exp"
    echo "------ Launching further QHL instance(s) ------"
    let max_qmd_id="$num_tests + 1"

    # write to a script so we can recall analysis later.
    cd $this_run_results_folder
    cd ../../../

    for i in \`seq 1 $max_qmd_id\`;
        do
        redis-cli flushall 
        let q_id="$q_id + 1"
        echo "QID: $q_id"
        python3 /scripts/implement_qmla.py \
            -fq=1 \
            -p=$particles \
            -e=$experiments \
            -rq=$use_rq \
            -qhl=0 \
            -dir=$this_run_results_folder \
            -qid=$q_id \
            -pt=$plots \
            -pkl=1 \
            -log=$this_log \
            -cb=$bayes_csv \
            -prior_path=$prior_pickle_file \
            -true_params_path=$system_parameters_file \
            -true_expec_path=$true_expec_path \
            -plot_probes=$plot_probe_file \
            -latex=$latex_mapping_file \
            -ggr=$growth_rule \
            -ggr=$growth_rule \
            $growth_rules_command 
    done
    echo "
    cd $this_run_results_folder
    python3 ../../../../scripts/AnalyseMultipleQMD.py \
        -dir=$this_run_results_folder \
        --bayes_csv=$bayes_csv \
        -log=$this_log \
        -top=$number_best_models_further_qhl \
        -qhl=0 \
        -fqhl=1 \
        -true_expec=$true_expec_path \
        -ggr=$growth_rule \
        -plot_probes=$plot_probe_file \
        -params=$system_parameters_file \
        -latex=$latex_mapping_file
    " > $further_analysis_script

    chmod a+x $further_analysis_script
    echo "------ Launching analyse further QHL ------"
    # sh $further_analysis_script
fi


# redis-cli shutdown
