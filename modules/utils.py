import os
import re
import string as st
import numpy as np
import pandas as pd 

import warnings
warnings.filterwarnings('ignore')
import spacy
import nltk

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from spacy.matcher import PhraseMatcher

from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB



nlp = spacy.load("en_core_web_sm")
# education_path = './education.csv'

skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

def extract_hard_soft_skills(text):
    ''' 
       Extract hard skills, soft skills and certification from given job text 
       using pre-trained model SkillNER
    '''
    annotations = skill_extractor.annotate(text)
    hard_soft_skills = {'Hard Skill' : [],
                   'Soft Skill' : [],
                   'Certification': []}

    # Distinguish skills based on skills type i.e. Hard or Soft skills
    for _ , arr_skills in annotations["results"].items():
        for skill in arr_skills:
            skill_name = skill["doc_node_value"]
            skill_type = SKILL_DB[skill["skill_id"]]["skill_type"]
            hard_soft_skills[skill_type].append(skill_name)
            
    # extract unique hard and soft skills
    hard_soft_skills['Hard Skill'], hard_soft_skills['Soft Skill'] = list(set(hard_soft_skills['Hard Skill'])), list(set(hard_soft_skills['Soft Skill']))
    return hard_soft_skills

def create_dataframe(job_description, position = ''):
    hard_skills = ''
    soft_skills = ''
    # education = extract_education(job_description, education_path)
    education = ''
    df = pd.DataFrame(columns = ['position', 'job_description', 'hard_skills', 'soft_skills', 'education'])
#     print(df.columnsumns)
    hard_soft_skills = extract_hard_soft_skills(job_description)
    # education = extract_education(job_description, education_path)
    education = ''
    for h_skill in hard_soft_skills['Hard Skill']:
        hard_skills += h_skill + ', '
    
    for s_skill in hard_soft_skills['Soft Skill']:
        soft_skills += s_skill + ', '
        
    jd = {'position': [position],
        'job_description' : [job_description],
     'hard_skills': [hard_skills],
     'soft_skills': [soft_skills],
     'education': ' '.join(education)}
    
    df = pd.DataFrame(jd)
    
    return df


def load_cv_data(path):
    df = pd.read_csv(path)
    return  df
    
def convert_text_vec(manual_data, df):
    '''
      manual data -> dataframe 20+ different CV
      df -> dataframe of jd
    '''
    count_vectorize = CountVectorizer(analyzer='word', ngram_range=(1, 2),  
                                      stop_words='english', lowercase=True)
    
    count_vectorize.fit(manual_data['Hard Skills'] + 
                        manual_data['Soft Skills'] + manual_data['Education'])
    
    count_vectorize.fit(df['position'] + df['hard_skills'] + df['hard_skills'] + df['education'])
    
    person_vector = count_vectorize.transform(manual_data['Hard Skills'] + manual_data['Soft Skills'] + manual_data['Education'])
    
    # job_df = df[df['position'] == position].copy()
    job_df = df
    job_vector = count_vectorize.transform(job_df['hard_skills'] + job_df['soft_skills'] + job_df['position'] + job_df['education'])
    
    return job_vector, person_vector

def get_similarity(vec_a, vec_b):
    similarity = cosine_similarity(vec_a, vec_b)[0] * 100
    return similarity

def final_result(position, cv_data, job_vector, person_vector):
#     job_vector, person_vector =  convert_text_vec(manual_data, df)
    position = 'Machine Learning Tech Lead'
    similarity = get_similarity(job_vector, person_vector)
    cv_data['Similarity'] = similarity
    cv_data.sort_values('Similarity', ascending=False).head()
    return cv_data

def main(job_description, position):
    '''
    
    '''
    path = './data/cv_data.csv'
    # hard_soft_skills =  extract_hard_soft_skills(job_description)
    # education = extract_education(job_description, education_path)
    
    # education = ''
    job_df = create_dataframe(job_description)
    cv_df = load_cv_data(path)

    # replace null values with ' '
    cv_df.fillna(' ', inplace=True)
    job_vector, person_vector =  convert_text_vec(cv_df, job_df)
    # print(np.array(person_vector.todense().shape))
    # print(np.array(job_vector.todense().shape))
    # print(person_vector)
    # similarity = get_similarity(job_vector, person_vector)
    # print(similarity)
    cv_data = final_result(position, cv_df, job_vector, person_vector)
    return cv_data[['Name', 'Hard Skills', 'Soft Skills', 'Similarity']], job_df

if __name__ == "__main__":
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

    position = 'ML Engineer (CV / NLP)'
    result = main(job_description, position)
    print(result.columns)
    print(result)