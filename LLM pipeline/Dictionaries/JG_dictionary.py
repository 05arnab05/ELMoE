# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:38:04 2025

@author: arnab
"""

JG_dict = {
    "Accuracy": '''Judgment guidance (not an algorithm):

                    - Cluster context: If the query’s cluster probability is high, treat that cluster as the main context. Prefer high-confidence gradient directions; treat low-confidence directions as weaker evidence.

                    - Neighbor consistency: Prefer neighbors that closely match the query features and show consistent predicted targets. Consider internal quality (e.g., lower loss) as supportive evidence.
                    
                    - Variability matters: Use the provided LocalRange (feature range among similar points) and UniqueVals (number of distinct values among similar points). Features with very low LocalRange or UniqueVals≈1 are weak local evidence, even if their gradient is large. Favor features that both vary locally and have informative gradients for the given target.
                    
                    - Target-specific: Re-evaluate evidence separately for each target; a feature can matter for one target and not the others.
                    
                    - Transparency (implicit): Internally consider the flow cluster context → neighbor consistency → local variability → gradient confidence. Use this to guide your final percentages, but output JSON only.
                                     ''',
    
    "Counterfactuals": '''Judgment guidance (not an algorithm):

                    - Cluster context: If the query’s cluster probability is high, treat that cluster as the main context. Prefer high-confidence gradient directions; treat low-confidence directions as weaker evidence.

                    - Neighbor consistency: Prefer neighbors that closely match the query features and show consistent predicted targets. Consider internal quality (e.g., lower loss) as supportive evidence.
                    
                    - Variability matters: Use the provided LocalRange (feature range among similar points) and UniqueVals (number of distinct values among similar points). Features with very low LocalRange or UniqueVals≈1 are weak local evidence, even if their gradient is large. Favor features that both vary locally and have informative gradients for the given target.
                    
                    - Target-specific: Re-evaluate evidence separately for each target; a feature can matter for one target and not the others.
                    
                    - Transparency (implicit): Internally consider the flow cluster context → neighbor consistency → local variability → gradient confidence. Use this to guide your final percentages, but output JSON only.
                                     ''',
    
    "Feature_importance": ''''You will receive a dataframe of network features and predicted CQI labels. 
                        Your role is to quantify the importance of each feature in determining CQI using real numbers. 
                        Each feature should have a non-zero value. 
                        Use your own reasoning and do not include the target column CQI.''',
}
