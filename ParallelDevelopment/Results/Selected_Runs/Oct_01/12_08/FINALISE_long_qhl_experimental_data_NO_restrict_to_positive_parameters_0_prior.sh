
#!/bin/bash 
cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_08 --bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_08/multiQMD.csv -top=2 -qhl=1 -fqhl=0 -data=NVB_dataset.p -exp=1 -params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_08/true_params.p -true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_08/true_expec_vals.p

"""
qmd_id=100
cd /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment
for i in `seq 0 99`;
do
	let qmd_id=1+$qmd_id
	qsub -v QMD_ID=$qmd_id,OP=xTiPPyTiPPzTiPPxTxPPyTyPPzTz,QHL=0,FURTHER_QHL=1,EXP_DATA=1,GLOBAL_SERVER=newblue1,RESULTS_DIR=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_08,DATETIME=Oct_01/12_08,NUM_PARTICLES=10000,NUM_EXP=2500,NUM_BAYES=2499,RESAMPLE_A=0.8,RESAMPLE_T=0.5,RESAMPLE_PGH=2.0,PLOTS=0,PICKLE_QMD=0,BAYES_CSV=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_08/multiQMD.csv,CUSTOM_PRIOR=1,DATASET=NVB_dataset.p,DATA_MAX_TIME=5000,DATA_TIME_OFFSET=205,GROWTH=two_qubit_ising_rotation_hyperfine_transverse,TRUE_PARAMS_FILE=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_08/true_params.p,PRIOR_FILE=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_08/prior.p,TRUE_EXPEC_PATH=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_08/true_expec_vals.p -N finalise_long_qhl_experimental_data_NO_restrict_to_positive_parameters_0_prior\_$qmd_id -l walltime=10:00:00,nodes=1:ppn=2 -o /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_08/output_and_error_logs//finalise_output.txt -e /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_01/12_08/output_and_error_logs//finalise_error.txt run_qmd_instance.sh 
done 
"""
	
