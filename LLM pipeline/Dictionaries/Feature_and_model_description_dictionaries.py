# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:38:04 2025

@author: arnab
"""
Model_description_dict = {
    "DT": ''' This is a Decision Tree regressor that uses the squared error criterion to measure the quality of splits. 
              It has no limit on tree depth or the number of leaf nodes, allowing it to grow fully until stopping conditions are met. 
              The tree uses the 'best' splitter strategy to choose the most optimal feature at each node. 
              It requires at least 2 samples to attempt a split and at least 1 sample per leaf. 
              No pruning is applied (ccp_alpha=0.0), and no feature or sample constraints are enforced during training. 
              The model is initialized with a fixed random state (random_state=42) for reproducibility and is well-suited for capturing non-linear relationships in tabular regression tasks."
                 ''',
    
    "RF": '''This is a Random Forest regressor consisting of 100 decision trees, using the squared error criterion to measure split quality. It employs bootstrap sampling (bootstrap=True) and uses all features when determining splits (max_features=1.0). 
            The model has no specified maximum tree depth, allowing trees to grow until all leaves are pure or contain fewer than the minimum required samples. 
            Each tree requires at least 2 samples to split and at least 1 sample per leaf. The model does not use out-of-bag evaluation (oob_score=False) and is initialized with a fixed random state (random_state=42) to ensure reproducibility. 
            It is well-suited for regression tasks on structured data with non-linear relationships.''',
    
    "MLP": '''The model named Multi-Layer Perceptron is a multi-layer perceptron designed for tabular data with 6 input features. 
               It consists of two hidden layers: the first with 64 neurons using ReLU activation, and the second with 32 neurons also using ReLU. 
               The output layer has 2 neurons with linear activation, suitable for regression tasks. All layers are trainable, and the model uses float32 precision with Glorot Uniform weight initialization. 
               This structure makes it well-suited for predicting two continuous values from structured input.''',
    
    "Transformer": '''The model named Feature Transformer is a Transformer encoder for tabular data with 8 input features. 
                  Each feature is projected to a 128-dimensional embedding with a learnable positional embedding to 
                  encode feature identity. The encoder stack has 2 layers, each with 4 self-attention heads and a 
                  256-unit feed-forward network. Encoded tokens are pooled, passed through a 64-unit ReLU layer, and 
                  mapped to 2 linear outputs for regression. All layers are trainable, using float32 precision, Glorot 
                  Uniform initialization, and the Adam optimizer. This design enables modeling of complex dependencies 
                  between input features.''',
}

Primary_features_dict = {
    'Healthcare': {

        "Demographics & Admission Information": 
            '''ADMISSION_TYPE: Category describing how the patient entered the hospital 
                               (e.g., EMERGENCY, ELECTIVE, URGENT). Strong predictor of acuity and LOS.

               GENDER: Biological sex recorded at admission. Affects disease prevalence (e.g., anemia, renal injury).

               ETHNICITY: Patient’s self-reported ethnicity. In MIMIC, often coarse categories (WHITE/BLACK/ASIAN); 
                          sometimes correlates with comorbidity patterns or access-to-care factors.

               AGE: Patient age at admission. One of the strongest predictors of mortality and expected length of stay.

               AGE_GROUP: Binned version of AGE (18–40, 41–65, 66–90). Helps models learn non-linear patterns more easily.
            ''',

        "Comorbidity Indicators": 
            '''NUM_DIAGNOSES: Number of unique ICD-9 codes assigned during the hospital stay.
                               Higher values capture disease burden and multimorbidity.

               AGE_X_NUM_DIAGNOSES: Interaction term combining aging and comorbidity load.
                                    Older patients with many diagnoses typically have the highest LOS.

               LOG_NUM_DIAGNOSES: Log-scaled comorbidity measure to reduce skewness and 
                                   emphasize differences among low–moderate disease burden.

               LABEL_MORTALITY: Binary indicator (0/1) of in-hospital mortality. 
                                 Included *only as a descriptive feature*; avoid during prediction to prevent leakage.
            ''',

        "Cardio-Respiratory Vital Signs (First 6h of ICU/Hospital Admission)":
            '''HR (Heart Rate): Beats per minute. High values may indicate sepsis, fever, hypovolemia, 
                                or acute stress response; low values may indicate conduction issues.

               MAP (Mean Arterial Pressure): Key marker of perfusion. Low MAP suggests shock or organ under-perfusion.

               RESP_RATE (Respiratory Rate): Elevated in infection, respiratory distress, metabolic acidosis; 
                                             suppressed in neurological depression.

               SPO2 (Oxygen Saturation): Non-invasive indicator of oxygenation efficiency. 
                                        Lower values signal respiratory failure or hypoxia.

               TEMP_C (Temperature): Body temperature in °C. High fever indicates infection; 
                                     low temperature may signal sepsis severity.
            ''',

        "Laboratory Measurements (First 24h Labs)": 
            '''FIRST_HEMOGLOBIN: First hemoglobin value (g/dL). Reflects anemia, bleeding, and oxygen-carrying capacity.

               FIRST_CREATININE: Indicator of kidney function. Early AKI strongly increases LOS.

               FIRST_GLUCOSE: Blood glucose level. Both hyperglycemia and hypoglycemia correlate with poor outcomes.

               FIRST_SODIUM: Important for fluid balance, osmolarity, and neurological function.

               FIRST_POTASSIUM: Key electrolyte regulating cardiac conduction; abnormal values increase risk.

               FIRST_BICARBONATE: Reflects metabolic acidosis/alkalosis; important for identifying sepsis or renal failure.

               LOG_FIRST_*: Log-transformed versions of laboratory values to reduce skewness 
                             and improve model learnability.
            '''
    },
        
    'Networking':{
                    "Radio and signal features": '''RSRQ (Reference Signal Received Quality): Indicates the quality of the received LTE signal by combining signal strength and interference; lower values suggest poor quality.
                                                    RSSI (Received Signal Strength Indicator): Measures the total received power of the signal, including noise and interference; used to assess overall signal strength.
                                 ''',
                    
                    "Location Features": '''Longitude: Geographic longitude (horizontal coordinate) of the user equipment or network event.
                                            Latitude: Geographic latitude (vertical coordinate) of the user equipment or network event.''',
                    
                    "Channel Quality and Mobility": '''CQI (Channel Quality Indicator): Integer representing channel conditions as reported by the device; higher values imply better downlink channel quality.
                              Speed: The velocity of the user equipment, typically in km/h or m/s; may affect signal stability and throughput.''',
                    
                    "Throughput Features": '''UL_bitrate (Uplink Bitrate): Actual uplink data transmission rate in bits per second.
            
                                              DL_bitrate (Downlink Bitrate): Actual downlink data transmission rate in bits per second.
            
                                              ('UL_bitrate_predicted',): Predicted uplink bitrate by the machine learning model, used to compare against actual performance.
            
                                              ('DL_bitrate_predicted',): Predicted downlink bitrate by the machine learning model.'''},
    
    'AQ': {
                    "Nitrogen Oxides (Traffic-related gases)": '''NO (Nitric Oxide): A primary pollutant mainly emitted from vehicle exhaust and combustion sources. 
                                                                 Short-lived, often elevated near traffic sites.
                                                                 
                                                                 NO₂ (Nitrogen Dioxide): Secondary pollutant formed from NO oxidation; 
                                                                 contributes to ozone formation and respiratory irritation.''',
                
                    "Particulate Matter": '''PM₁₀ (Particulate Matter ≤10 µm): Coarse particles from dust, construction, and road resuspension; 
                                              inhalable but less deeply penetrating into lungs.
                                              
                                              PM₂.₅ (Particulate Matter ≤2.5 µm): Fine particles from combustion and secondary aerosols; 
                                              penetrate deeply into the lungs and are strongly linked to health effects.''',
                
                    "Photochemical Pollutant": '''O₃ (Ozone): Secondary pollutant formed by photochemical reactions of NOₓ and volatile organic compounds (VOCs) 
                                                   under sunlight; can irritate airways and damage ecosystems.''',
                
                    "Carbonaceous Aerosol": '''BC (Black Carbon): Light-absorbing carbonaceous aerosol from incomplete combustion (diesel, biomass burning). 
                                               Strongly linked to both climate warming (via absorption and snow albedo effects) 
                                               and health risks (cardiovascular, respiratory).''',
                
                    "Meteorological Variables": '''WD (Wind Direction): Indicates the origin of airflow (in degrees). Important for pollution transport and source attribution.
                                                   
                                                   WS (Wind Speed): Governs pollutant dispersion and dilution; low wind speeds often coincide with higher pollution accumulation.
                                                   
                                                   T (Temperature): Influences atmospheric chemistry (e.g., ozone formation) and vertical stability; 
                                                   also directly affects human thermal comfort.
                                                   
                                                   RH (Relative Humidity): Modulates aerosol hygroscopic growth, visibility, and secondary particle formation.
                                                   
                                                   P (Pressure): Surface atmospheric pressure (hPa). Slow changes relate to synoptic weather systems, 
                                                   while short-term variations can indicate local meteorological influences.'''
}
                
}


Secondary_features_dict = {
    "DT": '''DT_conditions: A string-encoded logical expression summarizing the combination of signal quality, geographic, and mobility conditions (e.g., RSRQ ≤ -0.62 AND Latitude > 0.37) used by the RF model to predict.
                 ''',
    
    "RF": '''RF_conditions: A string-encoded logical expression summarizing the combination of signal quality, geographic, and mobility conditions (e.g., RSRQ ≤ -0.62 AND Latitude > 0.37) used by the RF model to predict.''',
    
    "MLP": ''''Secondary features: epoch 1 to epoch 20: Model loss or error value after each training epoch (iteration over training data) for a row or batch of data; helps track learning progress.
               Cumulative_epoch_loss: The total or average loss accumulated across all epochs for a row or batch of data, used as a summary metric of model training quality.''',
               
    "Transformer": ''''Secondary features: epoch 1 to epoch 20: Model loss or error value after each training epoch (iteration over training data) for a row or batch of data; helps track learning progress.
               Cumulative_epoch_loss: The total or average loss accumulated across all epochs for a row or batch of data, used as a summary metric of model training quality.'''
}

