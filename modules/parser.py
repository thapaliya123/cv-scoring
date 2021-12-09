'''
parse JD and CV data 
'''
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import docx2txt
import pickle
import fitz

# set path
model_path = './models/segment_identifier.pkl'

# def extract_text_from_pdf(pdf_path):
#     '''
#     Helper function to extract the plain text from .pdf files
#     :param pdf_path: path to PDF file to be extracted
#     :return: iterator of string of extracted text
#     '''
#     if str(type(pdf_path)) == "<class 'str'>":
#         with open(pdf_path, 'rb') as fh:
#             for page in PDFPage.get_pages(fh, 
#                                         caching=True,
#                                         check_extractable=True):
#                 resource_manager = PDFResourceManager()
#                 fake_file_handle = io.StringIO()
#                 converter = TextConverter(resource_manager, fake_file_handle, codec='utf-8', laparams=LAParams())
#                 page_interpreter = PDFPageInterpreter(resource_manager, converter)
#                 page_interpreter.process_page(page)
    
#                 text = fake_file_handle.getvalue()
#                 yield text
    
#                 # close open handles
#                 converter.close()
#                 fake_file_handle.close()
#     else:
#         for page in PDFPage.get_pages(pdf_path, 
#                                         caching=True,
#                                         check_extractable=True):
#                 resource_manager = PDFResourceManager()
#                 fake_file_handle = io.StringIO()
#                 converter = TextConverter(resource_manager, fake_file_handle, codec='utf-8', laparams=LAParams())
#                 page_interpreter = PDFPageInterpreter(resource_manager, converter)
#                 page_interpreter.process_page(page)
    
#                 text = fake_file_handle.getvalue()
#                 yield text
    
#                 # close open handles
#                 converter.close()
#                 fake_file_handle.close()

# Fitz Parser
def extract_text_from_pdf(pdf_path):
    '''
    Helper function to extract the plain text from .pdf files using fitz parser
    :param pdf_path: path to PDF file to be extracted
    :return: iterator of string of extracted text
    '''
    doc = fitz.open(pdf_path) 
    
    for page in doc:
        yield (page.get_text("text", sort=True))

def extract_text_from_doc(doc_path):
    '''
    Helper function to extract plain text from .doc or .docx files
    :param doc_path: path to .doc or .docx file to be extracted
    :return: string of extracted text
    '''
    temp = docx2txt.process(doc_path)
    text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
    return ' '.join(text)

def extract_text(file_path, extension):
    '''
    Wrapper function to detect the file extension and call text extraction function accordingly
    :param file_path: path of file of which text is to be extracted
    :param extension: extension of file `file_name`
    '''
    text = ''
    if extension == '.pdf':
        for page in extract_text_from_pdf(file_path):
            text += ' ' + page
    elif extension == '.docx' or extension == '.doc':
        text = extract_text_from_doc(file_path)
    return text

def load_model(pickle_path):
    with open(pickle_path,'rb') as pkl:
        model = pickle.load(pkl)
    return model

def resume_creator():
    """
    Create list of words for each of the sections such as Experience, projects, skills, etc in the resume. Created list
    of words may help to segment different sections in the resume. Also returns possible title keywords dictionary
    by concatinating all the words which is used as a feature for later prediction using segment_identifier.pkl
    """
    personal_info = set("profile information personal information additional information information: informations: introduction introduction:".split(' '))
    objective = set("summary career summary summary: Summary: summary career objective objective: motivation motivation:".split(' '))
    skills = set("technical skills key mainfiles professional practical skill"
                          " skill: skills skills skills:"
                          " competencies competencies:".split(' '))
    experiences = set("work job projects summary jobs practical"
                               " responsibilities responsibilities: employment "
                               "professional career experience experience: "
                               "experiences experiences: profile Work Experience"
                               " profile: profiles profiles:".split(' '))
    projects = set("training training: trainings trainings: college projects "
                            "projects: training trainings: attended attended:".split(' '))
    academics = set("EDUCATION education education: educational academic qualification"
                             " qualification: qualifications qualifications:".split(' '))
    rewards = set("certification certification: certifications "
                           "certifications rewards rewards:License License"
                           " honours awards Licenses Licenses:".split(' '))
    languages = set("language language: languages languages:".split(' '))
    references = set("reference reference: references references:".split(' '))
    links = set('links links: LINK LINK:'.split(' '))
    
    possible_title_keywords = personal_info | objective | experiences | skills | projects | academics | rewards | languages | references | links
    
    return personal_info, objective, skills, experiences, projects, academics, rewards, languages, references, links,  possible_title_keywords

def unique_index_headings(text):
    """
    Returns: unique_idx_headings (dict)
        A dictionary with key = integer representing sentence in pdf, value = possible word in that sentence to be heading 
        predicted by model named gaussian.
    """
    # load model 
    model = load_model(model_path)
    personal_info, objective, skills, experiences, projects, academics, rewards, languages, references, links, possible_title_keywords = resume_creator()      
    sent_lines = text.splitlines()
    list_of_headings_with_repeated_index = []
    # get sentence and its sentence index in resume document.
    for index, sent in enumerate(sent_lines):
        sent = sent.split(" ")
        sent = [x.strip() for x in sent if x.strip()]
        if len(sent) < 4:
            for word in sent:        
                features = [word.istitle() | word.isupper(), word.islower(), word.isupper(), word.endswith(":"),len(word) <= 3, word.lower() in possible_title_keywords]
                features = [[int(elem) for elem in features]]
                #                 predict if word with above features is heading
                if model.predict(features) == 1:
                    list_of_headings_with_repeated_index.append({index: word})
                        
    uniquekeys = set()
    unique_indx_headings = dict()
    # titles_with_repeated_indices = predict_repeating_index_headings(input_text) 
    for t in list_of_headings_with_repeated_index:
        for key, value in t.items():
            if key not in uniquekeys:
                unique_indx_headings[key] = value
                uniquekeys.add(key)
            else:
                unique_indx_headings[key] = unique_indx_headings[key] + ' ' + value
    return unique_indx_headings

def sliced_resume_text(text = None):
        resume_text = text
    
        sent_lines = resume_text.splitlines()
       
        # index of last line of the resume
        end_index = len(sent_lines)
        sliced_text = {}
        unique_index_heading_title = unique_index_headings(text)
        #list of indices of heading titles, will be used for slicing of resume text
        list_of_title_indices = list(unique_index_heading_title)
       
        # add last index of sentence splited resume to slice the last section of resume
        list_of_title_indices.append(end_index)
        
        # i=0 initialization is required to slice resume text from index-0 to first heading index of resume
        # which is in most cases, the personal information of the candidate
        i = 0
        for key, value in unique_index_heading_title.items():
            #for i in range(len(list_of_title_indices)-1):
            sliced_text["profile information"] = sent_lines[0:list_of_title_indices[0]]
            sliced = sent_lines[list_of_title_indices[i] + 1:list_of_title_indices[i + 1]]
            sliced_text[value] = sliced
            #sliced_text.append(value: sliced)
            i += 1
        return sliced_text

def format_segment(text):
        Profile = []
        Objectives = []
        Experiences = []
        Skills = []
        Projects = []
        Educations = []
        Rewards = []
        Languages = []
        References = []
        Links = []

        pharsed_info = sliced_resume_text(text)
        personal_info, objective, skills, experiences, projects, academics, rewards, languages, references, links,  possible_title_keywords = resume_creator()
        
        for k, v in pharsed_info.items():
            if len(set(k.lower().split())) == len(personal_info.intersection(set(k.lower().split()))):
                Profile.extend(v)
            elif len(set(k.lower().split())) == len(objective.intersection(set(k.lower().split()))):
                Objectives.extend(v)
            elif len(set(k.lower().split())) == len(skills.intersection(set(k.lower().split()))):
                Skills.extend(v)
            elif len(set(k.lower().split())) == len(experiences.intersection(set(k.lower().split()))):
                Experiences.extend(v)
            elif len(set(k.lower().split())) == len(languages.intersection(set(k.lower().split()))):
                Languages.extend(v)
            elif len(set(k.lower().split())) == len(projects.intersection(set(k.lower().split()))):
                Projects.extend(v)
            elif len(set(k.lower().split())) == len(academics.intersection(set(k.lower().split()))):
                Educations.extend(v)
            elif len(set(k.lower().split())) == len(rewards.intersection(set(k.lower().split()))):
                Rewards.extend(v)
            elif len(set(k.lower().split())) == len(references.intersection(set(k.lower().split()))):
                References.extend(v)
            elif len(set(k.lower().split())) == len(links.intersection(set(k.lower().split()))):
                Links.extend(v)
            else:
                pass

        resume_info_extracted = {'profile':' '.join(Profile),
                                 'objectives':' '.join( Objectives),
                                 'experiences':'EXPERIENCE\n'+'\n'.join( Experiences),
                                 'skills':' '.join(Skills),
                                 'projects':' '.join(Projects),
                                 'academics':'EDUCATION\n'+'\n'.join( Educations),
                                 'rewards': ' '.join(Rewards),
                                 'languages':' '.join(Languages),
                                 'references':' '.join( References),
                                 'links':' '.join(Links),
                                 }
        return resume_info_extracted
# print(extract_text("./data/jd.pdf", ".pdf"))