
# Ensure latest version of gradio installed to get dropdowns to work
import os
os.system("pip install gradio -U")


import gradio as gr
from datetime import datetime
import pandas as pd
import numpy as np
from transformers import pipeline
from keybert import KeyBERT


today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")
# -

def extract_kwords(text, text_df, length_slider, in_colnames, diversity_slider, candidate_keywords):
             
        if not text_df:
            in_colnames="text"
            in_colnames_list_first = in_colnames

            in_text_df = pd.DataFrame({in_colnames_list_first:[text]})
            
        else: 
            in_text_df = pd.read_csv(text_df[0].name, delimiter = ",", low_memory=False, encoding='cp1252')
            in_colnames_list_first = in_colnames[0]
        
        if not candidate_keywords:
            keywords_text = KeyBERT().extract_keywords(list(in_text_df[in_colnames_list_first].str.lower()), stop_words='english', top_n=length_slider, 
                                                   keyphrase_ngram_range=(1, 1), use_mmr=True, diversity=diversity_slider)
            
        # Do this if you have pre-assigned keywords
        else:
        
            candidates_list = pd.read_csv(candidate_keywords.name, delimiter = ",", low_memory=False, encoding='cp1252').iloc[:,0].tolist()
            candidates_list_lower = [x.lower() for x in candidates_list]
  
            print(candidates_list_lower)
        
            keywords_text = KeyBERT().extract_keywords(list(in_text_df[in_colnames_list_first].str.lower()), stop_words='english', top_n=length_slider, 
                                                   keyphrase_ngram_range=(1, 1), use_mmr=True, diversity=diversity_slider, candidates=candidates_list_lower)


        if not keywords_text:
            return "No keywords found, original file returned", text_df[0].name


        if text_df == None:
            keywords_text_labels = [i[0] for i in keywords_text]
            keywords_text_scores = [i[1] for i in keywords_text]
            keywords_text_out = str(keywords_text_labels) #keywords_text[0].values()
            keywords_scores_out = str(keywords_text_scores)
           
        else: 
            print(keywords_text)
            
            keywords_text_out = []
            keywords_scores_out = []
            
            for x in keywords_text:
                keywords_text_labels = [i[0] for i in x]
                keywords_text_scores = [i[1] for i in x]
                
                keywords_text_out.append(keywords_text_labels) #[d['keyword_text'] for d in keywords_text_labels] #keywords_text[0].values()
                keywords_scores_out.append(keywords_text_scores)
                
        
        #print(keywords_text_out)
        
        output_name = "keywords_output_" + today_rev + ".csv"
        output_df = pd.DataFrame({"Original text":in_text_df[in_colnames_list_first],
                                  "Keywords":keywords_text_out,
                                  "Scores":keywords_scores_out})
        
        # Expand keywords out to columns
        ## Find the longest keyword list length to know how many columns to add

        if (len(output_df['Keywords']) > 1):
            list_len = [len(i) for i in output_df["Keywords"]]
            max_list_length = max(list_len)
            print(list_len)
            print(max_list_length)
            
            keyword_colname_list = ['kw' + str(x) for x in range(1,max_list_length+1)]
        else:
            print(len(eval(output_df["Keywords"][0])))
            keyword_colname_list = len(eval(output_df["Keywords"][0]))
        
        
        output_df[keyword_colname_list] = pd.DataFrame(output_df['Keywords'].tolist(), index= output_df.index)
 
        output_df["Keywords"] = output_df["Keywords"].astype(str).str.replace("[", "").str.replace("]", "")
        output_df["Scores"] = output_df["Scores"].astype(str).str.replace("[", "").str.replace("]", "")
               
        keywords_text_out_str = str(output_df["Keywords"][0])#.str.replace("dict_values([","").str.replace("])",""))
        keywords_scores_out_str = str(output_df["Scores"][0])#.str.replace("dict_values([","").str.replace("])",""))
        
        output_text = "Words: " + keywords_text_out_str + "\n\nScores: " + keywords_scores_out_str
        
        output_df.to_csv(output_name, index = None)
        
        return output_text, output_name

def detect_file_type(filename):
    """Detect the file type based on its extension."""
    if (filename.endswith('.csv')) | (filename.endswith('.csv.gz')) | (filename.endswith('.zip')):
        return 'csv'
    elif filename.endswith('.xlsx'):
        return 'xlsx'
    elif filename.endswith('.parquet'):
        return 'parquet'
    else:
        raise ValueError("Unsupported file type.")

def read_file(filename):
    """Read the file based on its detected type."""
    file_type = detect_file_type(filename)
    
    if file_type == 'csv':
        return pd.read_csv(filename, low_memory=False)
    elif file_type == 'xlsx':
        return pd.read_excel(filename)
    elif file_type == 'parquet':
        return pd.read_parquet(filename)

def put_columns_in_df(in_file):
    new_choices = []
    concat_choices = []
    
    for file in in_file:
        df = read_file(file.name)
        #print(df.columns)
        new_choices = list(df.columns)

        concat_choices.extend(new_choices)
        #concat_choices = list(set(concat_choices))
        
        print(concat_choices)
        
    return gr.Dropdown(choices=concat_choices)

def dummy_function(in_colnames):
    """
    A dummy function that exists just so that dropdown updates work correctly.
    """
    return None

# ## Gradio app - extract keywords

block = gr.Blocks(theme = gr.themes.Base())

with block:
 
    gr.Markdown(
    """
    # Extract keywords from text
    Enter open text below to get keywords. You can copy and paste text directly, or upload a file and specify the column that you want to keywords.
    """)    
   
    with gr.Accordion("I will copy and paste my open text", open = False):
        in_text = gr.Textbox(label="Copy and paste your open text here", lines = 5)
        
    with gr.Accordion("I have a file", open = False):
        in_text_df = gr.File(label="Input text from file", file_count="multiple")
        in_colnames = gr.Dropdown(choices=["Choose a column"], multiselect = True, label="Select column to find keywords (first will be chosen if multiple selected).")


    with gr.Accordion("I have my own list of keywords. Keywords will be taken from the leftmost column of the file.", open = False):
        candidate_keywords = gr.File(label="Input keywords from file (csv)")
        
    with gr.Row():
        length_slider = gr.Slider(minimum = 1, maximum = 100, value = 5, step = 1, label = "Maximum number of keywords")
        diversity_slider = gr.Slider(minimum = 0, maximum = 1, value = 0, step = 0.1, label = "Keyword diversity: 0 - keywords are based purely on score, 1 - keywords are ranked by diversity and less on score")

    with gr.Row():
        keywords_btn = gr.Button("Extract keywords")
        
    
    with gr.Row():
        output_single_text = gr.Textbox(label="Output example (first example in dataset)")
        output_file = gr.File(label="Output file")

    # Update column names dropdown when file uploaded
    in_text_df.upload(fn=put_columns_in_df, inputs=[in_text_df], outputs=[in_colnames])    
    in_colnames.change(dummy_function, in_colnames, None)

    keywords_btn.click(fn=extract_kwords, inputs=[in_text, in_text_df, length_slider, in_colnames, diversity_slider, candidate_keywords],
                    outputs=[output_single_text, output_file], api_name="keywords")

block.queue(concurrency_count=1).launch()

