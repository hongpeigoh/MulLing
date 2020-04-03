from mulling import MulLingVectorsAnnoy
from src import get, processing, query

import os.path
import math
from csv import writer

import json
from flask import Flask, request, jsonify, url_for, render_template, redirect
from flask_cors import CORS, cross_origin
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length, Email
app = Flask(__name__)
cors = CORS(app)
app.config['JSON_AS_ASCII'] = False
app.config['CORS_HEADERS'] = 'Content-Type'
secret_key = 'cs1tmu11ing'
app.config['SECRET_KEY'] = secret_key 


''' APP '''
class FeedbackForm(FlaskForm):
    name = StringField('Name',[DataRequired()])
    email = StringField('Email',[Email(message=('Not a valid email address.')), DataRequired()])
    data_type = SelectField('Feedback Type',[DataRequired()],
         choices=[('comments','Comments'),
                  ('questions', 'Questions'),
                  ('bugs', 'Bug Reports'),
                  ('features', 'Feature Requests'),
                  ('suggestions', 'Suggestions for Improvement')])
    feedback = TextAreaField('Feedback',[DataRequired()])
    submit = SubmitField('Submit')


@app.route('/')
def approot():
    return r"<strong>Hello World!</strong>"

@app.route('/feedback', methods=['GET','POST'])
@cross_origin()
def feedbackform():
    form = FeedbackForm()
    if form.validate_on_submit():
        row_content = [str(form.name.data),str(form.email.data),str(form.data_type.data),str(form.feedback.data)]
        print('Feedback received')
        if not os.path.isfile('dump/sheets/feedback.csv'):
            with open('dump/sheets/feedback.csv', 'a+', newline='') as f:
                writer(f).writerow(['Name','E-mail','Feedback Type', 'Text'])
                writer(f).writerow(row_content)
                f.close()
        else:
            with open('dump/sheets/feedback.csv', 'a+', newline='') as f:
                writer(f).writerow(row_content)
                f.close()
        
        return render_template('success.html')

    return render_template('index.html', form=form, key=secret_key)

@app.route('/query_mono')
@cross_origin()
def appquery():
    q = str(request.args.get('q'))
    model = str(request.args.get('model'))
    lang = str(request.args.get('lang'))
    k = int(request.args.get('k'))
    print(
        '    Query:          %s\n' %q,
        '   Model:          %s\n' %model,
        '   Language:       %s\n' %lang,
        '   No. of Results: %i' %k)
    
    results = query.monolingual_annoy_query(app_object, q, model, lang, k, clustering=True)

    return jsonify(
        allresults= list(get.json(app_object, results, model))
    )

@app.route('/query_multi')
@cross_origin()
def multiappquery():
    q = str(request.args.get('q'))
    model = str(request.args.get('model'))
    lang = str(request.args.get('lang'))
    k = int(request.args.get('k'))
    normalize = bool(request.args.get('normalize'))
    olangs = list()
    for lang_ in ['en', 'zh', 'ms', 'ta']:
        o = str(request.args.get('o%s' % lang_))
        if o == 'true':
            olangs.append(lang_)

    print(
        '    Query:           : %s\n' %q,
        '   Model:           : %s\n' %model,
        '   Language:        : %s\n' %lang,
        '   Output Languages : %s\n' %olangs,
        '   No. of Results   : %i\n' %k,
        '   Normalize Top L: : %s' %str(normalize))

    L = k if normalize else math.ceil(k/math.log(len(olangs)))

    if lang=='null':
        results = query.mulling_annoy_query(app_object, q, model, k, L=L, normalize_top_L=normalize, multilingual=True, olangs=olangs, clustering=True)
    else:
        results = query.mulling_annoy_query(app_object, q, model, k, L=L, normalize_top_L=normalize, multilingual=False, lang_= lang, olangs=olangs, clustering=True)

    return jsonify(
        allresults= list(get.json(app_object, results, model))
    )

if __name__ == "__main__":
    models = ['baa','bai','meta','laser','metalaser','senlaser','senbai']
    path = './dump'
    langs = ['en','zh','ms','ta']

    app_object = MulLingVectorsAnnoy(models=models, path=path, langs=langs)
    app.run(port=5050, host='0.0.0.0', debug=False)
