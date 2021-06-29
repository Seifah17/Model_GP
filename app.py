import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pymongo 
import dns
from bson.json_util import dumps, loads
import json
from flask import Flask, request

app = Flask(__name__)


@app.route('/')
def index():
    return "Flask server"

@app.route('/postdata', methods=['GET', 'POST'])
def postdata():
    user = request.get_json()
    print(user)
    connection_url = 'mongodb+srv://Seif:00774400@jobsultant.nnpaa.mongodb.net/jobsultant?retryWrites=true&w=majority'
    client = pymongo.MongoClient(connection_url)
    db = client['jobsultant']
    collection = db['TestJobs']
    jobs = collection.find()
    list_cur = list(jobs)
    json_data = dumps(list_cur, indent = 2) 
    jobs = json.loads(json_data)
    jobs_df = pd.json_normalize(jobs)
    jobs_skills = jobs_df[['_id.$oid','Key_Skills']]
    jobs_skills = jobs_skills.append(user,ignore_index=True)
    tfidf = TfidfVectorizer(stop_words='english')
    jobs_skills['Key_Skills'] = jobs_skills['Key_Skills'].fillna('')
    skills_matrix = tfidf.fit_transform(jobs_skills['Key_Skills'])
    similarity_matrix = linear_kernel(skills_matrix,skills_matrix)
    mapping = pd.Series(jobs_skills.index,index = jobs_skills['_id.$oid'])
    def recommend_jobs(job):
        job_index = mapping[job]
        similarity_score = list(enumerate(similarity_matrix[job_index]))
        similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        similarity_score = similarity_score[1:15]
        job_indicies = [i[0] for i in similarity_score]
        result = (jobs_skills['_id.$oid'].iloc[job_indicies])
        api = result.to_json()
        return api
    return json.dumps(recommend_jobs(user['_id.$oid']))

if __name__ == "__main__":
    app.run(env.process.port)
