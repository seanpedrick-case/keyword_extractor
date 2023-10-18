# +
import gradio as gr
from datetime import datetime
import pandas as pd
import numpy as np
from transformers import pipeline
from keybert import KeyBERT


today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")
# -

# ## Gradio app - extract keywords

# +

block = gr.Blocks(css=".gradio-container {background-color: black}")

with block:
    
    #default_colnames = np.array("text")
    #in_colnames=default_colnames
    
    def extract_kwords(text, text_df, length_slider, in_colnames, diversity_slider, candidate_keywords):
             
        if text_df == None:
            in_colnames="text"
            in_colnames_list_first = in_colnames

            in_text_df = pd.DataFrame({in_colnames_list_first:[text]})
            
        else: 
            in_text_df = pd.read_csv(text_df.name, delimiter = ",", low_memory=False, encoding='cp1252')
            in_colnames_list_first = in_colnames.tolist()[0][0]
        
        if candidate_keywords == None:
            keywords_text = KeyBERT().extract_keywords(list(in_text_df[in_colnames_list_first]), stop_words='english', top_n=length_slider, 
                                                   keyphrase_ngram_range=(1, 1), use_mmr=True, diversity=diversity_slider)
            
        # Do this if you have pre-assigned keywords
        else:
        
            candidates_list = pd.read_csv(candidate_keywords.name, delimiter = ",", low_memory=False, encoding='cp1252').iloc[:,0].tolist()
            candidates_list_lower = [x.lower() for x in candidates_list]
            
            #print(candidates_list)
        
            keywords_text = KeyBERT().extract_keywords(list(in_text_df[in_colnames_list_first]), stop_words='english', top_n=length_slider, 
                                                   keyphrase_ngram_range=(1, 1), use_mmr=True, diversity=diversity_slider, candidates=candidates_list_lower)

        if text_df == None:
            keywords_text_labels = [i[0] for i in keywords_text]
            keywords_text_scores = [i[1] for i in keywords_text]
            keywords_text_out = str(keywords_text_labels) #keywords_text[0].values()
            keywords_scores_out = str(keywords_text_scores)
           
        else: 
            #print(keywords_text_labels)
            
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
      
    gr.Markdown(
    """
    # Extract keywords from text
    Enter open text below to get keywords. You can copy and paste text directly, or upload a file and specify the column that you want to keywords.
    """)    
   
    with gr.Accordion("I will copy and paste my open text", open = False):
        in_text = gr.Textbox(label="Copy and paste your open text here", lines = 5)
        
    with gr.Accordion("I have a file", open = False):
        in_text_df = gr.File(label="Input text from file")
        in_colnames = gr.Dataframe(label="Write the column name for the open text to keywords",
                                   type="numpy", row_count=(1,"fixed"), col_count = (1,"fixed"),
                               headers=["Open text column name"])#, "Address column name 2", "Address column name 3", "Address column name 4"])
        
    with gr.Accordion("I have my own list of keywords. Upload a csv file with one column only - column title 'keywords'", open = False):
        candidate_keywords = gr.File(label="Input keywords from file (csv)")
        
    with gr.Row():
        length_slider = gr.Slider(minimum = 1, maximum = 100, value = 5, step = 1, label = "Maximum number of keywords")
        diversity_slider = gr.Slider(minimum = 0, maximum = 1, value = 0, step = 0.1, label = "Keyword diversity: 0 - keywords are based purely on score, 1 - keywords are ranked by diversity and less on score")

    with gr.Row():
        keywords_btn = gr.Button("Extract keywords")
        
    
    with gr.Row():
        output_single_text = gr.Textbox(label="Output example (first example in dataset)")
        output_file = gr.File(label="Output file")

    keywords_btn.click(fn=extract_kwords, inputs=[in_text, in_text_df, length_slider, in_colnames, diversity_slider, candidate_keywords],
                    outputs=[output_single_text, output_file], api_name="keywords")

block.queue(concurrency_count=1).launch()

