from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/bitcoin')
def bitcoin_page():
    return render_template('bitcoin.html')

@app.route('/ethereum')
def ethereum_page():
    return render_template('ethereum.html')

@app.route('/litecoin')
def litecoin_page():
    return render_template('litecoin.html')