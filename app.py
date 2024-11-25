from flask import Flask, render_template, request, redirect, url_for
from client_dataset import ProductPredictionPipeline

app = Flask(__name__)

# Route to process the dataset
@app.route('/process_dataset')
def process_dataset():
    # Call the function from client_dataset.py
    dataset_path = ProductPredictionPipeline()
    return f"Dataset processed and saved at {dataset_path}. <a href='/'>Go to Main Site</a>"

# Main route to serve the website
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/page1')
def page1():
    return render_template('page1.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form.get('number')
    return render_template('home.html', message=f'You entered: {user_input}')

if __name__ == '__main__':
    app.run(debug=True)
