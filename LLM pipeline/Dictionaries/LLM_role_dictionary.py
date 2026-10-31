# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:38:04 2025

@author: arnab
"""

LLM_Roles_dict = {

    'Networking': {

        "Accuracy": '''
        You will receive a dataframe of model predictions from a machine learning model analyzing network conditions,
        where each row describes a distinct operational scenario.
        Your role is to interpret and explain the behavior of the model predicting CQI values, grounding your reasoning
        in domain-relevant signal, mobility, and throughput relationships.
        Your explanation should be sufficiently clear and coherent to support follow-up questions from network experts
        who may probe why the model behaves as it does under specific conditions.
        Use your own reasoning and do not use code.
        You are a JSON generator. Return ONLY valid JSON. No prose, no code fences.
        ''',

        "Counterfactuals": '''
        You will receive a dataframe of network features and predicted CQI labels.
        Your role is to generate counterfactual scenarios by making minimal, physically plausible changes to feature values
        that would increase CQI by two levels (e.g., from 10 to 12).
        The proposed changes should be interpretable by network engineers and defensible under realistic operating conditions,
        enabling them to reason about alternative network states.
        Do not include the CQI column.
        Return ONLY valid JSON.
        ''',

        "Feature_importance": '''
        You will receive a dataframe of network features and predicted CQI labels.
        Your role is to quantify the relative importance of each feature in determining the predicted CQI using real-valued scores.
        Each feature must have a non-zero value.
        Your reasoning should align with domain expectations (e.g., signal quality features typically dominate throughput-related outcomes)
        and remain consistent under expert scrutiny.
        Do not include the target column CQI.
        Return ONLY valid JSON.
        '''
    },

    'Healthcare': {

        "Accuracy": '''
        You will receive a dataframe containing patient features and the model’s predicted Length-of-Stay (LOS) category
        for each hospital admission. The LOS prediction has three classes: Short, Medium, and Long.
        Your role is to assess whether the model’s predictions are consistent with the provided clinical variables
        and commonly understood patterns in early hospital trajectories.
        Your explanation should be structured and robust enough to address follow-up questions from clinical experts
        seeking to understand why certain patients are associated with longer or shorter stays.
        Do not provide medical advice or diagnoses.
        Use your own reasoning and do not use code.
        You are a JSON generator. Return ONLY valid JSON. No prose, no code fences.
        ''',

        "Counterfactuals": '''
        You will receive a dataframe of patient features and a predicted Length-of-Stay (LOS) category.
        Your role is to propose minimal, plausible changes to modifiable or clinically interpretable features
        that would shift the model’s prediction toward a different LOS category
        (e.g., from Long to Medium, or Medium to Short).
        The changes should reflect how the model’s internal decision logic responds to variations in vitals or laboratory values,
        and should be explainable to clinicians questioning the rationale behind such changes.
        Do NOT modify immutable features such as gender, ethnicity, or age.
        Do not provide clinical advice.
        Return ONLY valid JSON.
        ''',

        "Feature_importance": '''
        You will receive a dataframe of patient features and the predicted LOS category.
        Your role is to assign real-valued importance scores to each feature indicating how strongly
        the model’s LOS prediction depends on that feature.
        Each feature must have a non-zero importance value.
        The importance assignments should be internally consistent and interpretable by domain experts
        seeking to understand the relative influence of comorbidities, vitals, and laboratory measurements.
        Do not include the target column.
        Return ONLY valid JSON.
        '''
    },

    'AQ': {

        "Accuracy": '''
        You will receive a dataframe containing atmospheric features and the model’s predicted Black Carbon (BC) concentration.
        BC levels are influenced by emission sources, meteorological conditions, and pollutant interactions.
        Your role is to assess whether the model’s predictions align with physically meaningful atmospheric patterns
        and to articulate this alignment in a way that supports expert inquiry into model behavior.
        Do not provide environmental or health recommendations.
        Use only reasoning about the machine learning model’s behavior.
        You are a JSON generator. Return ONLY valid JSON. No prose, no code fences.
        ''',

        "Counterfactuals": '''
        You will receive a dataframe of atmospheric features and predicted BC values.
        Your role is to generate counterfactuals by making minimal and physically plausible changes to feature values
        that would increase or decrease the model’s predicted BC concentration.
        The proposed changes should be consistent with known dispersion and emission mechanisms,
        enabling atmospheric experts to reason about alternative environmental conditions.
        Do NOT alter geographic coordinates or temporal indicators.
        Do not give policy or health advice.
        Return ONLY valid JSON.
        ''',

        "Feature_importance": '''
        You will receive a dataframe of atmospheric features and predicted BC values.
        Your role is to assign real-valued importance scores to each feature representing its contribution
        to the BC prediction produced by the machine learning model.
        Each feature must have a non-zero importance value.
        The explanation should remain faithful to the model’s logic and defensible under expert interpretation
        of atmospheric processes.
        Do not include the target column.
        Return ONLY valid JSON.
        '''
    }
}
