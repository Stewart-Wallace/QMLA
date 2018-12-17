
#!/bin/bash 
"""
cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15 --bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15/multiQMD.csv -top=2 -qhl=0 -fqhl=0 -data=NVB_dataset.p -exp=1 -params=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15/true_params.p -true_expec=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15/true_expec_vals.p
"""

qmd_id=100
cd /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment
for i in `seq 0 99`;
do
	let qmd_id=1+$qmd_id
	qsub -v QMD_ID=$qmd_id,OP=xTiPPyTiPPzTiPPxTxPPyTyPPzTz,QHL=0,FURTHER_QHL=1,EXP_DATA=1,MEAS=hahn,GLOBAL_SERVER=newblue4,RESULTS_DIR=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15,DATETIME=Oct_15/20_15,NUM_PARTICLES=6000,NUM_EXP=2000,NUM_BAYES=799,RESAMPLE_A=0.8,RESAMPLE_T=0.5,RESAMPLE_PGH=1.2,PLOTS=0,PICKLE_QMD=0,BAYES_CSV=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15/multiQMD.csv,CUSTOM_PRIOR=1,DATASET=NVB_dataset.p,DATA_MAX_TIME=5000,DATA_TIME_OFFSET=205,GROWTH=two_qubit_ising_rotation_hyperfine,TRUE_PARAMS_FILE=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15/true_params.p,PRIOR_FILE=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15/prior.p,TRUE_EXPEC_PATH=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15/true_expec_vals.p -N finalise_optimised_params_long_qmd_exp_data\_$qmd_id -l walltime=20:00:00,nodes=1:ppn=2 -o /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15/output_and_error_logs//finalise_output.txt -e /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Oct_15/20_15/output_and_error_logs//finalise_error.txt run_qmd_instance.sh 
done 
	
