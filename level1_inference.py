import json
from bs4 import BeautifulSoup
import time

def get_data():
    with open("./sim_questions.json","r") as f:
        suggested_questions = json.load(f)["resources"][0]['values']
    return suggested_questions

def cleanTextDedup(text1):
    text1 = text1.replace('$$','').replace('\(', '').replace('\)', '').replace('\mathrm', '').replace('&nbsp;', '').replace('Question', '')
    text1 = BeautifulSoup(text1, "lxml").text
    text1 = [s for s in text1 if s.isalnum() or s.isspace()]
    text1 = "".join(text1)
    return text1

def solve(inference_engine, question_text, suggested_questions = None, mpnet_threshold=0.85):
    question_text = cleanTextDedup(question_text)
    if suggested_questions is None:
        suggested_questions = get_data()
    internal_duplicate_questions = []
    mpnet_duplicates = []
    final_questions = []
    temp_questions = []

    t1 = time.time()
    sentences = [cleanTextDedup(question['question']) for question in suggested_questions]
    similarity = inference_engine.compute_similarity_many_to_many(sentences)

    for i in range(0, len(suggested_questions)):
        for j in range(0, len(suggested_questions)):
            if(j<=i):
                continue
            if(j in internal_duplicate_questions):
                continue
            if similarity[i][j] > (float(mpnet_threshold) if mpnet_threshold else 0.85):
                internal_duplicate_questions.append(j)
                mpnet_duplicates.append(suggested_questions[j])
        if(i not in internal_duplicate_questions):
            temp_questions.append(suggested_questions[i])
    
    print("t1", time.time()-t1)
    t2 = time.time()
    temp_questions_text = [cleanTextDedup(question['question']) for question in temp_questions]
    similarity_query_to_suggested =  inference_engine.compute_similarity_one_to_many(
        question_text, temp_questions_text)
    for ix,question in enumerate(temp_questions):
        if( similarity_query_to_suggested[ix]  > (float(mpnet_threshold) if mpnet_threshold else 0.85)):
            mpnet_duplicates.append(question)
        else:
            final_questions.append(question)
    print("t2", time.time()-t2)
    final_questions = [q['question'] for q in final_questions]
    return final_questions


if __name__ == "__main__":
    start = time.time()
    res = solve("What is photosynthesis?")
    print(time.time()-start)