# tumor-classification

Steps to run :
1. Set up directory structure as

<code>

    Root —— DIPG — dicoms
		|
        
		—— EP — dicoms
        
		|
        
		—— MB — dicoms
<code>  
        
    
2. Set DATA_DIR in training_config.py

3. Run utils/preprocess_dataset.py

4. Run mri_dataset/split_data.py

5. Run tumor_classification.py 
