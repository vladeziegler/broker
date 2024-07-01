from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import logging
from main import MyCrew
import os
from dotenv import load_dotenv
from threading import Thread
from uuid import uuid4
from datetime import datetime
import json
from threading import Lock

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load environment variables
load_dotenv()

# Job management
jobs = {}
jobs_lock = Lock()

# N8N webhook URL
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/bab8957f-3d34-4461-97bc-9d254cefc4e2"

# Configure logging
logging.basicConfig(level=logging.DEBUG,  # Set level to DEBUG for more verbosity
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Event:
    def __init__(self, timestamp, data):
        self.timestamp = timestamp
        self.data = data

class Job:
    def __init__(self):
        self.status = 'PENDING'
        self.result = None
        self.events = []

def append_event(job_id, data):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].events.append(Event(timestamp=datetime.now(), data=data))
            logger.debug(f"Event appended to job {job_id}: {data}")

def run_analysis(job_id, company, url):
    logger.debug(f"Starting analysis for job {job_id} with company {company} and url {url}")
    try:
        my_crew = MyCrew(company=company, job_id=job_id, url=url)
        result = my_crew.run()
        
        with jobs_lock:
            jobs[job_id].status = 'COMPLETE'
            jobs[job_id].result = result
            logger.debug(f"Analysis complete for job {job_id}. Result: {result}")
        
        # Send results to N8N webhook
        send_to_n8n(job_id, company, url, result)
    except Exception as e:
        with jobs_lock:
            jobs[job_id].status = 'ERROR'
            jobs[job_id].result = str(e)
        append_event(job_id, f"An error occurred: {str(e)}")
        logger.error(f"An error occurred during analysis for job {job_id}: {e}")

def send_to_n8n(job_id, company, url, result):
    # Implementation remains the same
    logger.debug(f"Sending results to N8N for job {job_id}")
    pass

@app.route('/api/analyze', methods=['POST'])
def analyze_company():
    data = request.json
    company = data.get('company')
    url = data.get('url')
    
    logger.debug(f"Received analyze request with data: {data}")

    if not company:
        logger.warning("Company name is required but missing in the request")
        return jsonify({"error": "Company name is required"}), 400
    
    job_id = str(uuid4())
    with jobs_lock:
        jobs[job_id] = Job()
        logger.debug(f"Job created with ID {job_id} for company {company} and url {url}")
    
    thread = Thread(target=run_analysis, args=(job_id, company, url))
    thread.start()
    
    logger.debug(f"Analysis thread started for job {job_id}")
    return jsonify({"job_id": job_id}), 202

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    logger.debug(f"Status request received for job {job_id}")

    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            logger.warning(f"Job with ID {job_id} not found")
            abort(404, description="Job not found")
    
    try:
        result_json = json.loads(job.result) if job.result else None
    except json.JSONDecodeError:
        result_json = job.result
    
    logger.debug(f"Returning status for job {job_id}: {job.status}")

    return jsonify({
        "job_id": job_id,
        "status": job.status,
        "result": result_json,
        "events": [{"timestamp": event.timestamp.isoformat(), "data": event.data} for event in job.events]
    })

if __name__ == '__main__':
    app.run(debug=True, port=3001)
