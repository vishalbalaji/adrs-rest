from flask import Flask, render_template, request, session, redirect, jsonify, flash
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
import os
import cv2
import numpy
import base64
import time
import random
import string
from bin.defectmarkingcall import defmarking
from bin.defectanalyzernew import load_graph, load_labels, predict
import json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.secret_key = os.urandom(12)
db = SQLAlchemy(app)
bcrypt = Bcrypt()
app.config['UPLOAD_FOLDER'] = './static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.mkdir(app.config['UPLOAD_FOLDER'])

graph1 = load_graph('bin/tmp1/output_graph.pb')
labels1 = load_labels("bin/tmp1/output_labels.txt")
graph2 = load_graph('bin/tmp/output_graph.pb')
labels2 = load_labels("bin/tmp/output_labels.txt")

for dirpath, dirnames, filenames in os.walk(app.config['UPLOAD_FOLDER']):
    for file in filenames:
        if os.path.isfile(os.path.join(dirpath, file)):
            os.remove(os.path.join(dirpath, file))

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)


def __repr__(self):
    return f'User("{self.username}", {self.password})'

def get_results(user_folder, file, pene_type, weld_type, printval):
    filename = os.path.join(user_folder, file.filename)
    results = defmarking(filename, pene_type, weld_type)

    new_filename = ''.join(random.choice(string.ascii_lowercase) for i in range(50)) + '.jpg'
    cv2.imwrite(os.path.join(user_folder, new_filename), results[1])
    direction = results[2].lstrip().strip('()')
    faults = []
    for fault in results[3]:
        if fault[2] == 'R':
            fault_length = f'{fault[1]:.2f}' + ' (Ø)'
        else:
            fault_length = f'{fault[1]:.2f}'
        new_fault = {'fault_length': fault_length}
        faults.append(new_fault)
    final_result = {'img_path': os.path.join(user_folder, new_filename),
            'success': True,
            'direction': direction,
            'linear': f'{results[4]}',
            'rounded': f'{results[5]}',
            'faulty': True,
            'printval': printval,
            'faults': faults}
    return jsonify(final_result)



@app.route('/', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user:
            if bcrypt.check_password_hash(user.password, password):
                session['logged_in'] = True
                session['user'] = username
                return redirect('/main')
            else:
                flash('Invalid password. Please try again')
                return redirect('/')
        else:
            flash('No such user. Please try again')
            return redirect('/')
    return render_template('login.html')


@app.route('/register', methods = ['GET', 'POST'])
def register():

    if request.method == 'POST':
        print('REGISTER POST')
        username = request.form['username']
        password = request.form['password']
        conf_pass = request.form['confirm-password']

        if password != '':
            if password != conf_pass:
                flash('Password does not match. Please try again')
            else:
                flash('Congratulations, user created!')
        else:
            flash('Password field cannot be empty')
    return redirect('/')

@app.route('/main', methods=['GET', 'POST'])
def main():
    if not session.get('logged_in'):

        return render_template('login.html')

    if request.method == 'POST':
        print('--------> POST <---------')
        if request.files['file']:
            file = request.files['file']
            user_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"./{session['user']}")
            if os.path.exists(user_folder):
                for name in os.listdir(user_folder):
                    os.remove(os.path.join(user_folder, name))
                print('Folder cleared')
            else:
                os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], f"./{session['user']}"))

            orig_img_path = os.path.join(user_folder, file.filename)
            file.save(orig_img_path)
            pene_type = request.form['pene-type']
            weld_type = request.form['weld-type']
            weld_type = weld_type.lower()
            print(pene_type, weld_type)
            current_time = time.time()
            try:
                printval = 'Processed RT film is ' + predict(orig_img_path,
                        graph1, labels1, graph2, labels2)
                final_result = {}
                if printval != 'Processed RT film is Non discontinuity':
                    final_result = get_results(user_folder, file, pene_type, weld_type, printval)
                else:
                    final_result.update({'faulty': False})
                    final_result.update({'printval': printval})
                    final_result.update({'success': True})

            except:
                final_result.update({'success': False})
            print(f'Elapsed time: {time.time() - current_time:.2f}s')

            return final_result

        else:
            return jsonify({'message': 'failure'})
    return render_template('main.html', username=session['user'].capitalize())


@app.route('/logout')
def logout():
    session['logged_in'] = False
    return redirect('/')



if __name__ == '__main__':

    app.config['TEMPLATES_AUTO_RELOAD'] = True

    app.run(debug=True, use_reloader=True, host='0.0.0.0', port=6200)
