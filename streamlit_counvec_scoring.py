import os 
import string as st
import pandas as pd
import numpy as np
import streamlit as stm
import warnings
import spacy
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from spacy.matcher import PhraseMatcher

from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB

from nltk.stem.snowball import SnowballStemmer
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from modules.utils import main


warnings.filterwarnings('ignore')
# load spacy small model
nlp = spacy.load("en_core_web_sm")
# education_path = "./data/education.csv"
position = 'ML Engineer (CV / NLP)'
job_description = '''
    Must have at least a bachelor's degree in Computer Science or similar. 
    Min. 2 years of relevant work experience. Proficient in Basic Machine Learning concepts: algorithms, evaluation procedures, etc, and 
    dealing with Common failure modes. Experienced in at least one area of application (E.g. CV, NLP, etc). 
    Has a sound knowledge of mathematical concepts like Linear Algebra, Probability and Statistics, Calculus. 
    Proficient in framework & libraries such as Numpy, Pandas, Matplotlib, Scikit-learn and a good grasp of at least one of Tensorflow or Pytorch.
    Familiar with Flask, FastAPI or Django, and some domain-specific tools (e.g: opencv, spacy, etc). 
    Good Grasp on programming language and concepts such as Python + OOP + SOLID, Data Structures and Algorithms, RESTful APIs, 
    and familiar with Architecture Design. Good Grasp of software tools and platforms such as git, conda, pip, jupyter, Docker, 
    and at least one cloud platform like AWS/GCP. Good grasp of a database such as SQL/NoSQL. Has a good grasp of 
    agile processes like Sprint and Kanban. Good Team Management, Communication, and Problem-Solving Skills


    Develop AI applications to adhere to designs that support business requirements for internal and external clients. Research and develop machine learning 
    models and work on the whole ML pipeline: data collection, wrangling, pre-processing, model building, evaluation, and deployment. Perform data analysis 
    to uncover insights that can be immediately actionable or can inform decisions around the ML process. Take initiative and ownership in writing requirement 
    specifications and design documents for a variety of development tasks including feature development, database design, and system integrations. Preparation, 
    drafting, and review of software documentation and project reports to meet internal and client requirements. Orchestrate deployment, monitoring, and 
    maintenance of ML applications as per requirement. Lead one or more projects in different capacities (if required). Guide other developers and help them 
    (as required) to do their work and look for ways to improve overall team output. Take on Leadership roles (e.g: Supervisorial) as required.
    '''
try:
    stm.title("CV scoring")
    stm.write("")
    stm.write("")
    # input job descriptions from users
    # job_description = stm.text_input("Enter Job Descriptions:", value=job_description)
    job_description = stm.text_area("Enter Job Descriptions:", value=job_description)

    # generate similarity using user Job Descriptions
    cv_df_with_score, job_df = main(job_description, position)
    # sort cv_df_with_score dataframe in descending order as per similarity scores
    cv_df_with_score.sort_values(by = ['Similarity'], ascending = False, inplace=True)

    # display extracted jd hard and soft skills to user
    jd_hard_skills, jd_soft_skills = job_df['hard_skills'], job_df['soft_skills']
    selected = stm.radio("Witness extracted JD's Hard and Soft Skills?", ("yes", "no"), index = 1)
    if selected == "yes":
        stm.markdown("### Displaying JD Hard and Soft Skills!!!")
        col1, col2 = stm.columns(2)
        col1.write(jd_hard_skills)
        col2.write(jd_soft_skills)
        # stm.write("**Hard Skills (JD):**", jd_hard_skills)
        # stm.write("**Soft Skills (JD):**", jd_soft_skills)

    # stm.subheader("Displaying computed Similarity score between JD and manually extracted CV data")
    stm.markdown("### Similarity score between passed JD and manually extracted 20 CV data!!!")
    stm.dataframe(cv_df_with_score)

except Exception as e:
    print(e)
    stm.exception("SOME PROBLEM OCCURED")
