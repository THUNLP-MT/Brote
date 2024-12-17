export CUDA_VISIBLE_DEVICES=$1

DATAPATH=$2
SAVEDIR=$3
SAVENAME=$4

#make sure your conda env is correct and supports Brote
python -u get_conditions.py \
	--input_data ${DATAPATH} \
	--output_dir ${SAVEDIR} \
	--output_name ${SAVENAME} \
	--model_name MMICL \
	--model_path BleachNick/MMICL-Instructblip-T5-xl \
	--processor_path Salesforce/instructblip-flan-t5-xl \
	--batch_size 16 \
	--model_size xl \
