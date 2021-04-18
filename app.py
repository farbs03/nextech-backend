from werkzeug.utils import cached_property
from flask import Flask, request, jsonify, make_response
from flask_restful import Api, Resource
import numpy as np
import pandas as pd
import sys
import model


app = Flask(__name__)
api = Api(app)

class GenerateInsights(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response
	
	def post(self):
		try: 
			formData = request.json
			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": model.run(formData)
			})
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})

api.add_resource(GenerateInsights, '/')


if __name__ == "__main__":
	app.run(debug=True)