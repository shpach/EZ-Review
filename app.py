import requests
import re
import os
import time
from flask import Flask, render_template, request
import ez_review

app = Flask(__name__)
now = 0
prev_item = ""

@app.route('/', methods=['POST','GET'])
def index():

	result = {"answer":0, "time":0}
	if request.method=='POST':
		search_item = {"search": request.form["search"]}
		result = search(search_item)
	return render_template('index.html', results=result)

def search(item):
	result = ez_review.main(item['search'])

	return { 
		"titles": result[0],
		"datasets": result[2],
		"methods": result[3]
		}

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8080)
