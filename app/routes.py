from flask import Blueprint, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from .process import process_excel

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('main.index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('main.index'))
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('app/uploads', filename)
        file.save(filepath)

        # Process the Excel file with your script
        result = process_excel(filepath)

        return render_template('result.html', result=result)
