
# AlterRec

Official code for the NeurIPS'23 paper "Enhancing ID and Text Fusion via Alternative Training in Session-based Recommendation".

# Download Data
Please download the amazon-m2 dataset through this [link] (https://drive.google.com/file/d/180K8PtOi0Rgsi4MEYHYQD0pE1yhRDOiB/view?usp=sharing), put it in the `dataset` folder and unzip it.

# Reproduce
AlterRec without augmentation (sess_FR_text_FR: Amazon-French, sess_ES_text_ES: spanish, sess_IT_text_IT: Italian):
```
python -u main.py  --data_name amazonm2/sess_FR_text_FR  --id_module transformer --lr_dc_step_id 2 --lr_dc_step_text 5 --random_epoch 4 --max_len 50  
```

AlterRec with augmentation:
```
python -u main.py  --data_name amazonm2/sess_FR_text_FR --aug --id_module transformer --lr_dc_step_id 2 --lr_dc_step_text 5 --random_epoch 4 --max_len 50  
```