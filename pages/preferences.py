import streamlit as st
import pandas as pd

st.markdown("# Preferences")


# ============================================
# json
# ============================================

# --------------------------------------------
# load preferences

json_path = './preferences/'

import json

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(file_path):
    with open(file_path) as f:
        tmp_json = json.load(f)
        return tmp_json

def get_pref():
    return load_json(json_path + "pref.json")

pref = get_pref()
# st.json(pref)

# --------------------------------------------
# load preferences


def is_nonexistent_list(part, whole):
    is_nonexistent = [(item not in whole) for item in part]
    return True if True in is_nonexistent else False

def add_nonexistent_items(part, whole):
    return list(set(whole) | set(part))

filter_history = pref["filter_words_history"]
filter_words = pref["filter_words"]

if is_nonexistent_list(
    part=filter_words, 
    whole=filter_history):
    filter_history = add_nonexistent_items(
        part=filter_words,
        whole=filter_history)


    
# --------------------------------------------
# button - add

filter_add = st.text_input(
    "Filter to add:", "")

btn_add = st.button("Add")

pref_tmp = load_json(json_path + "pref_tmp.json")
filter_added = pref_tmp["filter_added"]

if btn_add:
    if filter_add != "":
        filter_added.append(filter_add)        
        pref_tmp["filter_added"] = filter_added
        save_json(pref_tmp, "pref_tmp.json")
        
filter_history.extend(filter_added)
filter_words.extend(filter_added)

selected_filter_words = st.multiselect(
    "Select filter words", 
    filter_history,
    default=filter_words,
)

df_filter_selected = pd.DataFrame(
    selected_filter_words, 
    columns=["Filter Words"])
st.dataframe(df_filter_selected)



# --------------------------------------------
# button - save

btn_save = st.button("Save")

if btn_save:
    pref["filter_words_history"] = filter_history
    pref["filter_words"] = selected_filter_words
    save_json(pref, json_path + "pref.json")
    
    pref_tmp["filter_added"] = []
    save_json(pref_tmp, json_path + "pref_tmp.json")
