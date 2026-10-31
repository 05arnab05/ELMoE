# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:20:55 2025

@author: arnab
"""
from docx import Document

def Write_to_Doc(CQI_model,Metric,Prompt_sent,result_table): #result_table: result table for 1 model  1 metric
    dataframe=result_table
    try:
        file_name='abc'
        #Open the existing document
        document = Document(file_name)
        
        # Add a new title and body
        document.add_heading(CQI_model+' '+Metric, level=1)
        document.add_paragraph(Prompt_sent)
        
        # Add a table with a header row
        table = document.add_table(rows=1, cols=len(dataframe.columns))
        table.style = "Table Grid"

        # Add column headers
        hdr_cells = table.rows[0].cells
        for i, column in enumerate(dataframe.columns):
            hdr_cells[i].text = str(column)

        # Add rows from the DataFrame
        for _, row in dataframe.iterrows():
            row_cells = table.add_row().cells
            for i, value in enumerate(row):
                row_cells[i].text = str(value)

        
        # Save the changes to the same file
        document.save(file_name)
        print(f"Content appended to {file_name}")
    except Exception as e:
        print(f"An error occurred: {e}")
    