import os
import shutil
import time
import string as st
from typing import Iterable
import pandas as pd
import numpy as np
import streamlit as stm
import warnings
import spacy
from multiprocessing import Pool

# from skillNer.skill_extractor_class import SkillExtractor
# from skillNer.general_params import SKILL_DB


from modules.parser import extract_text, format_segment
from modules.utils import extract_hard_soft_skills
from modules.preprocessing import preprocess_text
from modules.evaluation import calculate_rouge
warnings.filterwarnings('ignore')

# load spacy small model
nlp = spacy.load("en_core_web_sm")

# initialize some global variables
jd_file_type = ".pdf"
cv_file_type = ".pdf"
jd_skills = ""
uploaded_files_dir = "./uploaded_files/"

def save_uploaded_file(uploaded_file, dir, file_name):
    """
    save uploaded files by user in the passed directory with passed filenames

    Arguments:
    uploaded_file: file uploaded by user
    dir: path of directory to store uploaded file
    file_name: file_name to store uploaded file 
    """
    with open(dir + file_name, 'wb+') as f:
            f.write(uploaded_file.getbuffer())

def make_directory(path):
    """
    create directory on the given path
    """
    os.mkdir(path)

def delete_directory(path):
    """
    delete files/dir using the path passed as an argument
    """
    if os.path.exists(path):
        # remove if exists
        shutil.rmtree(path)

def process_pdf_generate_score(uploaded_file):
    """
    Extract name from uploaded cv files using streamlit fileuploader, then segment cv into experience and skills section, then extract 
    skills using SKILLNER from segmented cv and generate recall via extracted skills from CV and JD.

    Arguments:
    uploaded_file: File uploaded using streamlit fileuploader

    Returns:
    record_skills_score: list with file_name, extracted skills, and recall score as value
    """
    # initialize list to record file_name, skills, and similarity score
    record_skills_score = []

    # get file name via uploaded pdf
    file_name = uploaded_file.name

    # buffer uploaded files
    save_uploaded_file(uploaded_file, uploaded_files_dir, file_name)

    # extract from CV
    cv = extract_text(uploaded_files_dir + file_name, cv_file_type)
    # cv  = extract_text(uploaded_file, cv_file_type)

    # cv segmentation and preprocessing
    segmented_cv_info = format_segment(cv)
    experience = " ".join(preprocess_text(segmented_cv_info['experiences']))
    skills = " ".join(preprocess_text(segmented_cv_info['skills']))

    # extract hard soft skills
    combined_info = experience + " " + skills
    hard_soft_skills = extract_hard_soft_skills(combined_info)
    cv_hard_skills, cv_soft_skills = hard_soft_skills['Hard Skill'], hard_soft_skills['Soft Skill']

    # concatenate extracted cv hard and soft skills list
    cv_skills = cv_hard_skills + cv_soft_skills

    # calculate rouge recall
    rouge_recall = calculate_rouge(jd_skills, cv_skills)['rouge1'][1]

    # add info to initialized list
    record_skills_score.append(file_name)
    record_skills_score.append(cv_skills)
    record_skills_score.append(rouge_recall)

    return record_skills_score

try:
    stm.title("CV scoring")
    stm.write("")
    stm.write("")
  
    # input JD pdf file
    jd_file = stm.file_uploader("Select JD pdf file")
    if jd_file is not None:
        # make directory
        make_directory(uploaded_files_dir)
        
        # save uploaded file
        save_uploaded_file(jd_file, uploaded_files_dir, "jd.pdf")
        
        # extract jd text
        jd = extract_text(uploaded_files_dir + 'jd.pdf', jd_file_type)
        # jd = extract_text(jd_file, jd_file_type)
        # preprocess extracted jd text
        clean_jd = " ".join(preprocess_text(jd))

        # extract hard and soft skills using SkillNER
        hard_soft_skills = extract_hard_soft_skills(clean_jd)
        # display extracted jd hard and soft skills to user
        jd_hard_skills, jd_soft_skills = hard_soft_skills['Hard Skill'], hard_soft_skills['Soft Skill']
        # concatenate jd_hard_skills and jd_soft_skills that represent jd skills 
        jd_skills = jd_hard_skills + jd_soft_skills

        # create radio button to grab user choice whether to hard and soft skills
        selected = stm.radio("Do you want to see extracted JD's Hard and Soft Skills?", ("yes", "no"), index = 1)
        if selected == "yes":
            stm.markdown("### Displaying JD Hard and Soft Skills!!!")
            col1, col2 = stm.columns(2)
            col1.write(pd.Series(jd_hard_skills, name='Hard Skills').to_frame())
            col2.write(pd.Series(jd_soft_skills, name="Soft Skills").to_frame())
    
    # input multiple CV pdf file
    uploaded_cv_files = stm.file_uploader("Select CV pdf file", accept_multiple_files = True)
    if len(uploaded_cv_files) > 0:
        # Do not use multiprocessing if user uploads a single file
        if (len(uploaded_cv_files) == 1):
            # Notify user
            stm.write("### Score for uploaded CV!!!")

            # process uploaded cv and generate recall score based on jd skills
            # start_time = time.time()
            cv_score_with_skills = [process_pdf_generate_score(uploaded_cv_files[0])]

            # end_time = time.time()
            # print(end_time - start_time, "sec")

        else:
            # Notify user
            stm.write("### Generating Score for uploaded CV's!!!")

            # Pool multiprocessing for faster execution, 
            # process all of the uploaded pdf using multiprocessing
            p = Pool(processes=6)
            start_time = time.time()
            cv_score_with_skills = p.map(process_pdf_generate_score, iterable=uploaded_cv_files)
            p.close()
            p.join()
            
            # accumulate results of multiprocessing in a list
            cv_score_with_skills = list(cv_score_with_skills)
            # end_time = time.time()
            # print((end_time - start_time), "sec")
            # start_time = time.time()
            # cv_score_with_skills = [process_pdf_generate_score(item) for item in uploaded_cv_files]
            # end_time = time.time()
            # print((end_time - start_time), "sec")

        # create dataframe via obtained 2d lists
        # columns = Name, SKills, Score
        cv_score_with_skills = pd.DataFrame(cv_score_with_skills, columns=["Name", "Skills", "Score"])
        # display dataframe
        stm.write(cv_score_with_skills)

    # delete uploaded files directory
    delete_directory(uploaded_files_dir)

except Exception as e:
    # print(e)
    # stm.write(e)
    stm.exception("SOME PROBLEM OCCURED")