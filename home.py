import os
import logging
import time

import numpy as np
import pandas as pd

import utils.custom_utils as cu
from utils.custom_utils import save_var, load_var, save_json, load_json

import streamlit as st
from PIL import Image

from paper_collector import paper_collector, paper_summarizer, email_sender

# ============================================
# User Functions
# ============================================
    
def get_pref(json_path='./preferences/'):
    return load_json(json_path + "pref.json")

def check_key(key, src):
    # Check if the key exists
    if key not in src:
        # If it doesn't exist, create a new key with an empty list as its value
        src[key] = []
    return src

def is_in_session_state(state_name):
    return True if state_name in st.session_state else False

def save_in_session_state(session_state):
    st.session_state = session_state

if is_in_session_state("pref"):
    pref = st.session_state.pref
else:
    pref = get_pref()
    
# ============================================
# Initialize
# ============================================

# --------------------------------------------
# markdown pages

st.markdown("# Accio Paper! 🪄📜")

# --------------------------------------------
# tabs

tab_home, tab_coll, tab_sele, tab_summ, tab_mail = st.tabs(
    ['Home', 
     'Collect', 
     'Select',
     'Summarize',
     'Email',
    ])

# ============================================
# tab 1 - home
# ============================================


col_img, col_msg = tab_home.columns(2)

with col_img:

    img_snape = Image.open('./images/severus_snape.webp')
    col_img.image(img_snape, use_column_width="auto")

# --------------------------------------------
# initialize - paper collector

@st.cache(allow_output_mutation=True)
def get_papercollector():
    return paper_collector.PaperCollector()

with st.spinner('Initializing...'):
    
    pc = get_papercollector()
    summ = paper_summarizer.Summarizer()
    es = email_sender.EmailSender()

with col_msg:
    col_msg.success('Good to go!', icon='🧹')
    










# ============================================
# tab 2 - Collect
# ============================================

# --------------------------------------------
# button - collect

tab_coll.markdown("# Query")

def set_papers():
    papers = load_var("papers")
    pc.set_papers(papers)


class StQuery():
    def __init__(self, tab):
        self.pref = self.update_pref()
        self.update_df()
        self.tab = tab
        self.pref_name = "filter_words_history"   # pref. key name for filter keyword 
        
    def save_pref(self, pref, path='./preferences/'):
        save_json(pref, path + "pref.json")
        
    def update_pref(self, path='./preferences/'):
        self.pref = load_json(path + "pref.json")
        
    def update_df(self):
        self.df_code = pd.read_csv("./preferences/arxiv_category.csv")
        
    def get_category_name(self):
        return [f"{item[0]} - {item[1]}" for item in zip(
            self.df_code["code"], self.df_code["name"])]
    
    def get_selected_category(self, selected_categories):
        category = [item.split(" - ")[0] for item in selected_categories]
        category = list(self.df_code[self.df_code["code"].isin(category)]["code"])
        category = [f"cat:{item}" for item in category]
        category = ' OR '.join(category)
        return category
    
    def get_selected_keywords(self, selected_keywords):
        return ','.join(selected_keywords)
    
    def remove_query_keywords(self, keywords):
        return [item.strip() for item in keywords.split(",")]
    
    def make_query_keywords(self, selected_keywords):
        keywords = [f"abs:{item}" for item in selected_keywords]
        keywords = ' OR '.join(keywords)
        return keywords
    
    def update_category(self):
        
        # category
        category_name = self.get_category_name()

        # category - multiselect
        self.category_ms = self.tab.multiselect(
            "Paper Category", category_name)

        # category - query making
        selected_code = self.get_selected_category(self.category_ms)

        # category - text input (display)
        self.category_ti = self.tab.text_input(
            "Category to Collect (empty: all)", selected_code)
        
           
    def update_keywords(self, pref):

        # keyword - multiselect
        self.keywords_ms = self.tab.multiselect(
            "Keywords History", pref[self.pref_name], pref[self.pref_name][-1])
        
        # keyword - query making
        selected_keywords = self.get_selected_keywords(self.keywords_ms)

        # keyword - text input (display)
        self.keywords_ti = self.tab.text_input(
            "Keyword to Collect (empty: all)", selected_keywords)
        
        
    def display_made_query(self, query, container):
        self.display_query_ti = container.text_input("Query:", query)

    def is_nonexistent(self, item, data):
        if (item != "") and (item not in data):
            return True
        else:
            return False

    def add_list(self, item, data):
        data.append(item)
        return data

    def make_query(self, category, keyword):
        return f"({category}) AND ({keyword})" if category and keyword else f"{category} {keyword}".strip()

    def append_unique_items(self, target_list, source_list):
        """
        Append items from source_list to target_list if they don't already exist.

        Parameters:
        target_list (list): The list to which items will be appended.
        source_list (list): The list containing items to be appended.

        Returns:
        list: The updated target_list.
        """
        target_list.extend(item for item in source_list if item not in target_list)
        return target_list        

    
    
stq = StQuery(tab_coll)
stq.update_category()
stq.update_keywords(pref)



# --------------------------------------------
# button - query

btn_query = tab_coll.button("Query")


# --------------------------------------------
# container - collect

disp_coll = tab_coll.container()


# --------------------------------------------
# button - query - processing



if btn_query:
    # save category as history
    if stq.is_nonexistent(stq.category_ti, pref["query_category"]):
        pref["query_category"].append(stq.category_ti)
        stq.save_pref(pref)

    # save keywords as history
    keywords = stq.remove_query_keywords(stq.keywords_ti)
    
    # using filter keywords history instead of query keywords
    pref[stq.pref_name] = stq.append_unique_items(pref[stq.pref_name], keywords)
    stq.save_pref(pref)

    # making query
    query_keywords = stq.make_query_keywords(keywords)
    query = stq.make_query(stq.category_ti, query_keywords)

    pc.query = query
    stq.display_made_query(pc.query, disp_coll)
    
    
tab_coll.markdown("---")

# --------------------------------------------
# button - collect

tab_coll.markdown("# Collect")

btn_collect = tab_coll.button("Collect")

tab_coll.markdown("---")

# --------------------------------------------
# container - paper table

disp_coll_papers = tab_coll.container()

display_columns = [
    "title", 
    "primary_author",
    "published", 
    "primary_category", 
    "entry_id", 
    "pdf_url",
]

display_columns_ext = [
    "downloaded_path",
    "summarized_path",    
]

def display_papers_collected():
    if len(pc.df_papers) != 0:
        disp_coll_papers.dataframe(
            pc.df_papers[display_columns])

tab_coll.write("Load collected papers from the saved file:")
btn_load_papers = tab_coll.button("Load papers")

tab_coll.write("Load the saved dataframe:")
btn_load_df = tab_coll.button("Load dataframe")
        
if btn_load_papers:
    set_papers()
    display_papers_collected()

if btn_load_df:
    df_papers = cu.load_var("df_papers")
    if len(df_papers) != 0:
        pc.df_papers = df_papers

    st.session_state.is_loaded_df = True

if is_in_session_state("is_loaded_df"):
    if st.session_state.is_loaded_df:
        tab_coll.dataframe(pc.df_papers[display_columns + display_columns_ext])

# --------------------------------------------


if btn_collect:
    
    with st.spinner('Collecting...'):

        # set the completed query
        stq.display_made_query(pc.query, disp_coll)
        pc.query = stq.display_query_ti
        disp_coll.write(pc.query)

        # normal mode
        pc.collect(query=pc.query)
        save_var(pc.papers, "papers")

        display_papers_collected()
        tab_coll.success('Collected!', icon='🪄')
        tab_coll.balloons()












# ============================================
# tab 2 - Select
# ============================================

# --------------------------------------------
# container - show tabel

import ast

def str2type(str_value):
    # Using ast.literal_eval to safely convert the string to a dictionary
    return ast.literal_eval(str_value)

disp_sele = tab_sele.container()

def get_filter_conditions_from_pref():
    key_name = "filter_conditions"
    stq.update_pref()
    pref = check_key(key_name, stq.pref)
    fc = pref[key_name]
    if len(fc) != 0:
        return fc[-1]
    else:
        return pc.get_default_conditions()

filter_ti = tab_sele.text_input(
    "Filter:",
    get_filter_conditions_from_pref()
)

btn_filtered = tab_sele.button("Filtered papers")

def display_papers_filtered(filter_conditions):

    # Check if the key exists
    pref_key = "filter_conditions"
    if pref_key not in pref:
        # If it doesn't exist, create a new key with an empty list as its value
        pref[pref_key] = []
    pref[pref_key].append(filter_conditions)
    stq.save_pref(pref)

    # filter the collected results    
    pc.df_filtered = pc.filter_dataframe(pc.df_papers, **filter_conditions)
    disp_sele.write("Filtered papers:")
    if len(pc.df_filtered) != 0:
        disp_sele.write(pc.filter_words)
        disp_sele.dataframe(pc.df_filtered[display_columns + display_columns_ext])

if is_in_session_state("is_paper_filtered"):
    if st.session_state.is_paper_filtered:
        if len(pc.df_filtered) != 0:
            disp_sele.dataframe(pc.df_filtered[display_columns + display_columns_ext])

if btn_filtered:
    filter_conditions = str2type(filter_ti)
    tab_sele.write(filter_conditions)
    display_papers_filtered(str2type(filter_ti))  # str to dict
    st.session_state.is_paper_filtered = True
    
tab_sele.markdown("---")

        
# --------------------------------------------
# multiselect


if len(pc.df_filtered) != 0:
    paper_list = list(pc.df_filtered["title"])
else:
    paper_list = []

selected_papers = tab_sele.multiselect(
    "Selected papers", paper_list)

def get_multiselect_values():
    selected_index = pc.df_filtered[pc.df_filtered["title"].isin(selected_papers)].index
    pc.select_dataframe(selected_index)
    tab_sele.dataframe(pc.df_selected[display_columns + display_columns_ext])
    st.session_state.is_paper_selected = True
    
# --------------------------------------------
# button - download papers


btn_download = tab_sele.button("Download papers")


if btn_download:

    with st.spinner('Downloading papers and preparing documents...'):
        
        get_multiselect_values()

        pc.download_papers()   # download pdf file of the selected papers
        pc.make_documents()   # make documents from pdf for summarization

        tab_sele.success("Prepared to summarize!", icon="📜")
        









        
# ============================================
# tab 3 - summarisation
# ============================================

disp_summ = tab_summ.container()

if is_in_session_state("is_paper_selected"):
    if st.session_state.is_paper_selected:

        disp_summ.dataframe(pc.df_selected[display_columns + display_columns_ext])
    
        btn_summ = tab_summ.button(
            "Summarize",
            key="summarize"
        )

def verify_documents(docs):
    return ' '.join(docs)

def get_documents(index):
    return pc.df_selected.iloc[index].processed_chunks

def summarize(docs):
    summ.summarize(docs)
    return summ.summarization_results

def disp_contents(contents, title):
    tab_summ.text_area(
        title,
        value=contents,
        height=400,
    )

def save_summary_to_file(index, text):
    file_path = pc.df_papers.iloc[index].downloaded_path.split('/')[-1].replace('pdf', 'txt')
    file_path = './summarized_papers/' + file_path
    cu.ensure_directory_exists('summarized_papers')
    
    try:
        with open(file_path, 'w') as f:
            f.write(text)
    
        file_size = os.path.getsize(file_path) / (1024 ** 2)  # Get size in MB
        logging.info(f'Saved: {file_path} (Size: {file_size:.2f} MB)')

    except Exception as e:
        logging.error(f"Error saving variable to {file_path}: {e}")

    return file_path

if is_in_session_state("is_paper_selected"):
    if st.session_state.is_paper_selected:

        # summarize button
        if btn_summ:

            tab_summ.write("Total progress:")
            papers_bar = tab_summ.progress(0)
            
            # with st.spinner("Summarizing..."):
            papers_number = len(pc.index_selected)
            for loop_index, paper_index in enumerate(pc.index_selected):

                tab_summ.write(pc.df_papers.iloc[paper_index].title)
                
                papers_progress = int(((loop_index + 1) / papers_number * 100))
                papers_bar.progress(papers_progress)

                documents_bar = tab_summ.progress(0)
                documents = pc.df_papers.iloc[paper_index].processed_chunks
                documents_number = len(documents)
                documents_summary = []
                for loop_index, doc in enumerate(documents):

                    # documents_summary.append(summarize([doc]))   # org.
                    documents_summary.append(f"{doc[:100]}")   # dev.
                    time.sleep(0.2)
                    documents_progress = int(((loop_index + 1) / documents_number * 100))
                    documents_bar.progress(documents_progress)
                    # break  # dev.

                documents_bar.progress(100)
                documents = pc.df_papers.iloc[paper_index]
                file_path = save_summary_to_file(paper_index, '\n\n'.join(documents_summary))
                pc.df_papers.loc[paper_index, 'summarized_path'] = file_path

            # update selected dataframe
            pc.make_selected_dataframe()

            papers_bar.progress(100)
            st.snow()
            tab_summ.success("Summarization Complete!", icon='🪄')
        
        # summrized contents information 

        tab_summ.markdown("---")
        
        selected_paper_summ = tab_summ.selectbox(
            "Selected papers:",
            pc.df_selected['title'])
        
        df_sel_summ = pc.df_selected[pc.df_selected.title == selected_paper_summ]
        tab_summ.dataframe(df_sel_summ[display_columns + display_columns_ext])

        # abstract
        disp_contents(df_sel_summ.iloc[0].summary, "Paper abstract:")

        # paper contents which is processed chunks
        paper_contents = '\n\n'.join(df_sel_summ.iloc[0].processed_chunks)
        disp_contents(paper_contents, "Paper contents:")

        # load summary from the file
        file_path = df_sel_summ.iloc[0].summarized_path
        summary = cu.load_text(file_path)
        disp_contents(summary, "Summary:")

        tab_summ.markdown("---")

        btn_save_df = tab_summ.button(
            "Save the dataframe",
        )
        
        if btn_save_df:
            cu.save_var(pc.df_papers, "df_papers")
            tab_summ.write("Saved!")

else:
    tab_summ.write('Not ready to summarize. \nPlease select papers!')









# ============================================
# tab 4 - email
# ============================================

if is_in_session_state("is_paper_selected"):
    if st.session_state.is_paper_selected:

        title = 'Contents:'
        paper_contents = []
        
        papers_number = len(pc.index_selected)
        for loop_index, paper_index in enumerate(pc.index_selected):

            # title
            title = pc.df_papers.iloc[paper_index].title
            # authors
            authors = [str(item) for item in pc.df_papers.iloc[paper_index].authors]
            authors = ', '.join(authors)
            # published
            published = pc.df_papers.iloc[paper_index].published
            # link
            entry_id = pc.df_papers.iloc[paper_index].entry_id
            # categories
            categories = ', '.join(pc.df_papers.iloc[paper_index].categories)
            # abstract
            abstract = pc.df_papers.iloc[paper_index].summary
            # summary
            file_path = pc.df_papers.iloc[paper_index].summarized_path
            summary = cu.load_text(file_path)
            # make contents
            tmp_contents = f"Title: {title}\nAuthors: {authors}\nPublished: {published}\nCategories: {categories}\nLink: {entry_id}\n\nAbstract:\n{abstract}\n\nSummary:\n{summary}"
            paper_contents.append(tmp_contents)

        contents = f"\n\n\n{'=' * 80}\n\n\n".join(paper_contents)
        tab_mail.text_area(
            title,
            value=contents,
            height=400,
        )
    
        btn_mail = tab_mail.button(
            "Send"
        )
        
        if btn_mail:
        
            es.send_msg(contents)
            tab_mail.success("Sent!", icon='🦉')

else:

    tab_mail.write('Not ready to send. \nPlease select and summarize papers!')