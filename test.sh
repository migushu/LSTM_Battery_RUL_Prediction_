#python lstm_predict.py --predict_measure 1 --sequence_length 11 --batch_size 16 --epochs 100
#python lstm_predict.py --predict_measure 1 --sequence_length 20 --batch_size 16 --epochs 50
#python lstm_multi_variable.py --predict_measure 0 --sequence_length 11 --epochs 200 --dropout 0.2

#python run.py --epochs 50 --feature_num 1 --get_model_measure 0
#python run.py --epochs 50 --feature_num 1 --usecols 9 10 --get_model_measure 0
#python run.py --epochs 50 --feature_num 7 --usecols 3 4 5 6 7 8 9 10 --get_model_measure 0
#python run.py --sequence_length 50 --epochs 50 --feature_num 7 --usecols 3 4 5 6 7 8 9 10 --get_model_measure 0
python ./sks/sks_run.py --timestep 5 --epochs 50 --batch_size 8