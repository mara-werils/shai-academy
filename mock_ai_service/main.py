from flask import Flask, jsonify, request
import uuid

app = Flask(__name__)


@app.post("/generate")
def generate():
    _ = request.get_json(force=True, silent=True) or {}
    response = {"task_id": str(uuid.uuid4()), "status": "queued"}
    return jsonify(response), 202

