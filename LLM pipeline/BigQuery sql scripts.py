# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 13:30:35 2025

@author: arnab
"""
#SQL queries

""" chartevents query

WITH icu AS (
  SELECT
    ICUSTAY_ID,
    HADM_ID,
    SUBJECT_ID,
    INTIME AS ICU_INTIME
  FROM `physionet-data.mimiciii_clinical.icustays`
),
ce AS (
  SELECT
    ce.SUBJECT_ID,
    ce.HADM_ID,
    ce.ICUSTAY_ID,
    ce.CHARTTIME,
    ce.ITEMID,
    ce.VALUENUM,
    ce.VALUEUOM
  FROM `physionet-data.mimiciii_clinical.chartevents` ce
  JOIN icu
    USING (ICUSTAY_ID, HADM_ID, SUBJECT_ID)
  WHERE
    -- keep only rows within 6 hours of ICU admission
    TIMESTAMP_DIFF(ce.CHARTTIME, icu.ICU_INTIME, HOUR) BETWEEN 0 AND 6
    -- filter to vital-sign ITEMIDs (example list, adjust as needed)
    AND ce.ITEMID IN (
      211,    -- Heart Rate
      220045, -- Heart Rate (Metavision)
      456,    -- Mean BP
      220052, -- Mean BP (Art)
      618,    -- Resp Rate
      220210, -- Resp Rate (Metavision)
      646,    -- SpO2
      220277, -- SpO2 (Metavision)
      678,    -- Temp C
      223761  -- Temp C (Metavision)
    )
)
SELECT * FROM ce
LIMIT 1000000;
"""

"""
chartevents extended

-- chartevents query including vital signs + Glasgow Coma Scale (GCS)
WITH icu AS (
  SELECT
    ICUSTAY_ID,
    HADM_ID,
    SUBJECT_ID,
    INTIME AS ICU_INTIME
  FROM `physionet-data.mimiciii_clinical.icustays`
),

ce AS (
  SELECT
    ce.SUBJECT_ID,
    ce.HADM_ID,
    ce.ICUSTAY_ID,
    ce.CHARTTIME,
    ce.ITEMID,
    ce.VALUENUM,
    ce.VALUEUOM
  FROM `physionet-data.mimiciii_clinical.chartevents` ce
  JOIN icu
    USING (ICUSTAY_ID, HADM_ID, SUBJECT_ID)
  WHERE
    -- within first 6 hours of ICU admission
    TIMESTAMP_DIFF(ce.CHARTTIME, icu.ICU_INTIME, HOUR) BETWEEN 0 AND 6
    AND ce.VALUENUM IS NOT NULL
    AND ce.ITEMID IN (
      -- Vitals
      211,       -- Heart Rate (CareVue)
      220045,    -- Heart Rate (MetaVision)
      456,       -- Mean BP (CareVue)
      220052,    -- Mean BP (Art)
      618,       -- Resp Rate (CareVue)
      220210,    -- Resp Rate (MetaVision)
      646,       -- SpO2 (CareVue)
      220277,    -- SpO2 (MetaVision)
      678,       -- Temp C (CareVue)
      223761,    -- Temp C (MetaVision)

      -- GCS: Eye, Verbal, Motor responses
      184, 220739,   -- Eye Opening
      723, 223900,   -- Verbal Response
      454, 223901    -- Motor Response
    )
),

first_vals AS (
  -- keep the first recorded value per (HADM_ID, ITEMID)
  SELECT
    SUBJECT_ID,
    HADM_ID,
    ICUSTAY_ID,
    ITEMID,
    VALUENUM,
    ROW_NUMBER() OVER (
      PARTITION BY HADM_ID, ITEMID
      ORDER BY CHARTTIME
    ) AS rn
  FROM ce
)

SELECT
  SUBJECT_ID,
  HADM_ID,
  ICUSTAY_ID,

  -- vitals
  MAX(CASE WHEN ITEMID IN (211, 220045) THEN VALUENUM END) AS FIRST_HR,
  MAX(CASE WHEN ITEMID IN (456, 220052) THEN VALUENUM END) AS FIRST_MAP,
  MAX(CASE WHEN ITEMID IN (618, 220210) THEN VALUENUM END) AS FIRST_RESP_RATE,
  MAX(CASE WHEN ITEMID IN (646, 220277) THEN VALUENUM END) AS FIRST_SPO2,
  MAX(CASE WHEN ITEMID IN (678, 223761) THEN VALUENUM END) AS FIRST_TEMP_C,

  -- GCS components
  MAX(CASE WHEN ITEMID IN (184, 220739) THEN VALUENUM END) AS FIRST_GCS_EYE,
  MAX(CASE WHEN ITEMID IN (723, 223900) THEN VALUENUM END) AS FIRST_GCS_VERBAL,
  MAX(CASE WHEN ITEMID IN (454, 223901) THEN VALUENUM END) AS FIRST_GCS_MOTOR,

  -- total GCS score
  SAFE_CAST(
    MAX(CASE WHEN ITEMID IN (184, 220739) THEN VALUENUM END) +
    MAX(CASE WHEN ITEMID IN (723, 223900) THEN VALUENUM END) +
    MAX(CASE WHEN ITEMID IN (454, 223901) THEN VALUENUM END)
    AS FLOAT64
  ) AS FIRST_GCS_TOTAL

FROM first_vals
WHERE rn = 1
GROUP BY SUBJECT_ID, HADM_ID, ICUSTAY_ID
ORDER BY HADM_ID
LIMIT 1000000;

"""


"""
labevents query

-- First-24h labs per hospital admission (HADM_ID)
-- Extended to include more core labs (Chloride, BUN, Hematocrit, WBC)
-- using MIMIC-III clinical dataset on BigQuery

WITH lab_items AS (
  SELECT
    le.SUBJECT_ID,
    le.HADM_ID,
    le.ITEMID,
    le.CHARTTIME,
    le.VALUENUM
  FROM `physionet-data.mimiciii_clinical.labevents` AS le
  JOIN `physionet-data.mimiciii_clinical.admissions` AS adm
    USING (HADM_ID)
  WHERE
    -- Lab ITEMIDs of interest
    le.ITEMID IN (
      -- Core chemistry / CBC
      50811,        -- Hemoglobin (blood gas)
      50912,        -- Creatinine
      50931,        -- Glucose (chemistry)
      50983,        -- Sodium
      50971,        -- Potassium
      50882,        -- Bicarbonate (HCO3-)
      50902,        -- Chloride
      51006,        -- Urea Nitrogen (BUN)
      51221,        -- Hematocrit
      51300, 51301  -- WBC Count / White Blood Cells
    )
    AND le.VALUENUM IS NOT NULL
    -- first 24 hours after hospital admission
    AND DATETIME_DIFF(le.CHARTTIME, adm.ADMITTIME, HOUR) BETWEEN 0 AND 24
),

first_labs AS (
  -- take the FIRST measurement per HADM_ID + ITEMID (earliest CHARTTIME)
  SELECT
    SUBJECT_ID,
    HADM_ID,
    ITEMID,
    VALUENUM,
    ROW_NUMBER() OVER (
      PARTITION BY HADM_ID, ITEMID
      ORDER BY CHARTTIME
    ) AS rn
  FROM lab_items
)

SELECT
  SUBJECT_ID,
  HADM_ID,

  -- first value for each lab, pivoted to columns
  MAX(CASE WHEN ITEMID = 50811 THEN VALUENUM END) AS FIRST_HEMOGLOBIN,
  MAX(CASE WHEN ITEMID = 50912 THEN VALUENUM END) AS FIRST_CREATININE,
  MAX(CASE WHEN ITEMID = 50931 THEN VALUENUM END) AS FIRST_GLUCOSE,
  MAX(CASE WHEN ITEMID = 50983 THEN VALUENUM END) AS FIRST_SODIUM,
  MAX(CASE WHEN ITEMID = 50971 THEN VALUENUM END) AS FIRST_POTASSIUM,
  MAX(CASE WHEN ITEMID = 50882 THEN VALUENUM END) AS FIRST_BICARBONATE,

  -- NEW: additional labs to better match the 17-variable set
  MAX(CASE WHEN ITEMID = 50902 THEN VALUENUM END) AS FIRST_CHLORIDE,
  MAX(CASE WHEN ITEMID = 51006 THEN VALUENUM END) AS FIRST_BUN,
  MAX(CASE WHEN ITEMID = 51221 THEN VALUENUM END) AS FIRST_HEMATOCRIT,
  MAX(CASE WHEN ITEMID IN (51300, 51301) THEN VALUENUM END) AS FIRST_WBC

FROM first_labs
WHERE rn = 1       -- keep only earliest measurement per lab
GROUP BY
  SUBJECT_ID,
  HADM_ID
ORDER BY
  HADM_ID
;


"""