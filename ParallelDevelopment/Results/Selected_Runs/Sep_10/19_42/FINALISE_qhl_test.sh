
#!/bin/bash 
cd /panfs/panasas01/phys/bf16951/QMD/Libraries/QML_lib
python3 AnalyseMultipleQMD.py -dir=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_10/19_42 --bayes_csv=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_10/19_42/multiQMD.csv -top=2


cd /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment

qsub -v QMD_ID=112,OP=xTiPPyTiPPzTiPPxTxPPyTyPPzTz,QHL=0,FURTHER_QHL=1,EXP_DATA=1,GLOBAL_SERVER=newblue1,RESULTS_DIR=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_10/19_42,DATETIME=Sep_10/19_42,NUM_PARTICLES=3000,NUM_EXP=1500,NUM_BAYES=1999,RESAMPLE_A=0.8,RESAMPLE_T=0.5,RESAMPLE_PGH=0.3,PLOTS=1,PICKLE_QMD=0,BAYES_CSV=/panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_10/19_42/multiQMD.csv,CUSTOM_PRIOR=1,DATASET=NVB_HahnPeaks_Newdata,DATA_MAX_TIME=5000,DATA_TIME_OFFSET=205 -N qhl_test_110 -l walltime=10:00:00,nodes=1:ppn=2 -o /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_10/19_42/OUTPUT_AND_ERROR_FILES//output_file_110.txt -e /panfs/panasas01/phys/bf16951/QMD/ParallelDevelopment/Results/Sep_10/19_42/OUTPUT_AND_ERROR_FILES//error_file_110.txt run_qmd_instance.sh
