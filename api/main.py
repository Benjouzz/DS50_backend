import werkzeug # pip install werkzeug==2.0.3
werkzeug.cached_property = werkzeug.utils.cached_property
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func

# main.py
from flask import Flask, request
from request_data import *
from flask_restplus import Api, Resource
from flask_cors import CORS, cross_origin
import json

app = Flask(__name__)
CORS(app, support_credentials=True)
api = Api(
    app, 
    version='v1', 
    title='DS50 Project API', 
    description='DS50 Project API',
    license='UTBM DS50',
    contact='Elian Belmonte',
    contact_email='elian.belmonte@utbm.fr',
    terms_url='https://www.utbm.fr/',
    default="DS50 Project API",
    default_label="",
    endpoint="DS50 Project API/swagger.json"
)

@api.route('/apis/DS50/Book/First1000')
class Book(Resource):
    def get(self):
        return getFirst1000Books()

@api.route('/apis/DS50/Author/First1000')
class Author(Resource):
    def get(self):
        return getFirst1000Authors()

@api.route('/apis/DS50/Review/First1000')
class Review(Resource):
    def get(self):
        return getFirst1000Reviews()

@api.route('/apis/DS50/Work/First1000')
class Work(Resource):
    def get(self):
        return getFirst1000Works()

@api.route('/apis/DS50/Interaction/First1000')
class Interaction(Resource):
    def get(self):
        return getFirst1000Interactions()

@api.route('/apis/DS50/Wrote/First1000')
class Wrote(Resource):
    def get(self):
        return getFirst1000Wrotes()

app.run()