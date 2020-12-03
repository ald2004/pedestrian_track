import flask
from flask import jsonify
import json
from flask import Flask, request, Response

app = Flask(__name__)
HOST = "192.168.8.121"


# Testing URL
@app.route('/api/v1/config/', methods=['GET'])
def hello_world():
    ret = {"code":200,"data":{"X":1100,"Y":1100},"success":True}
    return Response(json.dumps(ret, ensure_ascii=False),
                    mimetype='application/json')


app.run(debug=True, port=8080, host=HOST)
