import os
import pandas as pd
import arxiv as arx
import utils.custom_utils as cu
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

class PaperCollector:
    
    
    def __init__(self):
        
        self.file_paths = []                   # of downloaded papers
        self.paper_documents = []              # 
        self.papers = []                       # collected results
        self.filter_words = []                 # for collecting
        self.category = []                     # for collecting
        self.keywords = []                     # for collecting
        self.query = ""                        # for collecting
        self.filter_conditions = {}            # for filtering the collected results
        self.df_papers = pd.DataFrame([])      # dataframe - collected results
        self.df_filtered = pd.DataFrame([])    # dataframe - filtered in collected results
        self.df_selected = pd.DataFrame([])    # dataframe - selected in filtered results

        cu.ensure_directory_exists('papers')
        cu.ensure_directory_exists('summarized_papers')
    

    # --------------------------------------------------------
    # arXiv API
    # --------------------------------------------------------
    
    def set_arxiv_generator(self, query, page_size):
        
        self.arx_search = arx.Search(
          query=query,
          id_list=[],
        #   sort_by=arx.SortCriterion.Relevance,
          sort_by=arx.SortCriterion.SubmittedDate,
          sort_order=arx.SortOrder.Descending,
        )

        self.arx_client = arx.Client(
          page_size=page_size,
          delay_seconds=3,
          num_retries=3
        )

        self.results_generator = self.arx_client.results(self.arx_search)
        
    def make_arxiv_results(self, results_number):
        count = 0
        papers = []
        for paper in self.results_generator:
            papers.append(paper)
            count += 1
            if count >= results_number:
                break
        return papers
    
    
    # --------------------------------------------------------
    # Collect
    # --------------------------------------------------------    
        
    def collect(self, query="", page_size=1000, results_number=1000, filter_conditions={}):
        
        # collect papers
        self.set_arxiv_generator(query=query, page_size=page_size)
        self.papers = self.make_arxiv_results(results_number=results_number)
        
        # check the result
        if len(self.papers) != 0:
            self.set_papers(self.papers)
            
        # check the filter conditions
        # if it is empty, then set the default conditions
        if len(filter_conditions) == 0:
            self.filter_conditions = self.get_default_conditions()
            
        # filter the collected results by the filter conditions
#         self.filter_dataframe(self.papers, **self.filter_conditions)
        
        
        
    # --------------------------------------------------------
    # dataframe
    # --------------------------------------------------------
    
    def get_default_conditions(self):
        
        filter_conditions = {
            "title": ["", ""],                           # e.g ["language", "model"]
            "summary": ["", ""],                         # e.g ["language", "model"]
            "primary_category": ["cs.AI", "cs.LG"],      # e.g ["cs.AI", "cs.LG"]
            # other columns and values...
        }
        return filter_conditions
        
    
    def filter_dataframe(self, df, **kwargs):
        """
        Filter DataFrame based on multiple conditions (case-insensitive).

        Parameters:
        df (pd.DataFrame): Input DataFrame.
        **kwargs: Keyword arguments representing conditions.

        Returns:
        pd.DataFrame: Filtered DataFrame.
        """
        mask = pd.Series(True, index=df.index)

        for column, values in kwargs.items():
            if values is not None:
                if isinstance(values, list):
                    # If values is a list, use str.contains for substring matching
                    mask &= df[column].str.lower().str.contains('|'.join(map(str.lower, values)))
                elif isinstance(values, dict):
                    # If values is a dictionary, use exact match
                    mask &= df[column].isin(values.values())
                else:
                    # Handle other types of conditions as needed
                    pass

        return df[mask]
    
    def set_dataframe_filtered(self, df):
        self.df_filtered = df
    
    def make_dataframe(self):
        self.df_papers = pd.DataFrame([vars(paper) for paper in self.papers])
        self.df_papers['_result'] = self.papers
        self.df_papers['primary_author'] = self.df_papers['authors'].apply(lambda x: str(x[0]) if x else None)
        self.df_papers['paper_id'] = self.df_papers['entry_id'].apply(lambda x: x.split('/')[-1])
        
        self.df_papers['downloaded_path'] = ''
        self.df_papers['summarized_path'] = ''
        self.df_papers['processed_chunks'] = ''

        # Get the column names
        columns = self.df_papers.columns.tolist()
        
        # Move the 'paper_id' column to the first position
        columns.insert(0, columns.pop(columns.index('paper_id')))
        
        # Move the 'primary_author' column to the fifth position
        columns.insert(5, columns.pop(columns.index('primary_author')))

        # Move the 'entry_id' column to the fifth position from the last.
        columns.insert(-5, columns.pop(columns.index('entry_id')))
        
        # Reorder the DataFrame columns
        self.df_papers = self.df_papers[columns]
        
#     def filter_dataframe(self):
#         self.df_filtered = self.df_papers[self.df_papers["primary_category"] == 'cs.LG']
#         self.df_filtered = self.df_filtered[self.df_filtered['title'].str.contains(
#             '|'.join(self.filter_words), case=False)]
        
    def select_dataframe(self, index):
        self.index_selected = index
        self.make_selected_dataframe()

    def make_selected_dataframe(self):
        self.df_selected = self.df_papers.iloc[self.index_selected]
                
    def set_papers(self, papers):
        self.papers = papers
        self.make_dataframe()
        self.check_downloaded_papers()
        self.check_summarized_papers()
        
    def download_papers(self):
        if len(self.df_selected) != 0:
            for index in self.index_selected:
                if self.df_papers.iloc[index].downloaded_path == '':
                    file_path = self.df_papers.iloc[index]._result.download_pdf(
                            dirpath="./papers",)
                    # self.file_paths.append(file_path)
                    self.df_papers.loc[index, 'downloaded_path'] = file_path
                    print(f'downloaded. {file_path}')
            self.make_selected_dataframe()

    # --------------------------------------------------------
    # documents
    # --------------------------------------------------------
    
    def make_documents(self):
        if len(self.df_selected) != 0:
            for index in self.index_selected:
                self.df_papers.at[index, 'processed_chunks'] = self.get_document(self.df_papers.iloc[index]['downloaded_path'])
            self.make_selected_dataframe()
                
    def load_pdf(self, file):
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        return pages

    def split_documents(self, doc, chunk_size=200, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(doc)

    def file_preprocessing(self, file, chunk_size=200, chunk_overlap=50):
        pages = self.load_pdf(file)
        texts = self.split_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return texts

    def merge_content(self, texts, chunk_size=4096, ratio=0.5):
        final_texts = []
        tmp_texts = ''
        for text in texts:
            text_length = len(text)
            #print('> text_length: ', text_length)
            if len(tmp_texts) > 0:
                tmp_texts += ' ' + text        
                if len(tmp_texts) > chunk_size:
                    final_texts.append(tmp_texts)
                    tmp_texts = ''
            elif len(text) > chunk_size:
                final_texts.append(tmp_texts)
            else:
                tmp_texts = text
            
        return final_texts

    def get_document(self, file_path, verbose=False):
        processed_content = self.file_preprocessing(
            file_path, chunk_size=512, chunk_overlap=64)

        page_content = []
        for item in processed_content:
            page_content.append(item.page_content)

        merged_content = self.merge_content(
            page_content, chunk_size=4096, ratio=0.9)

        if verbose:
            for index, content in enumerate(merged_content):
                print(index, len(content))
                
        return merged_content

    # --------------------------------------------------------
    # files
    # --------------------------------------------------------
    
    def get_file_names(self, file_path="./papers"):
        file_names = os.listdir(file_path)
        return file_names

    def get_file_paths(self):
        return list(self.df_selected['downloaded_path'])
    
    def set_file_paths(self, paths):
        self.file_paths = paths

    # deprecated
    def get_downloaded_file_names(self, file_path="./papers/", file_extention=".pdf"):
        file_names = []
        for item in self.file_paths:
            file_names.append(item.split(file_path)[1].split(file_extention)[0].replace("_", " "))
        return file_names
    
    def make_query(self):
        self.query = " AND ".join([self.category, self.keywords])    

    def check_downloaded_papers(self):
        file_names = self.get_file_names()
        paper_ids = ['.'.join(x.split('.')[:2]) for x in file_names]
        paper_list = zip(paper_ids, file_names)
        for paper_id, file_name in paper_list:
            self.df_papers.loc[self.df_papers['paper_id'] == paper_id, 'downloaded_path'] = f"./papers/{file_name}"

    def check_summarized_papers(self):
        file_names = self.get_file_names(file_path = "./summarized_papers")
        paper_ids = ['.'.join(x.split('.')[:2]) for x in file_names]
        paper_list = zip(paper_ids, file_names)
        for paper_id, file_name in paper_list:
            self.df_papers.loc[self.df_papers['paper_id'] == paper_id, 'summarized_path'] = f"./summarized_papers/{file_name}"


