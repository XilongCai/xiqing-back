# backend/server.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

app = Flask(__name__)
CORS(app)

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    data = request.json
    V = float(data['x'])
    T = float(data['y'])

    def obj2():     
        nonlocal V
        nonlocal T        
        x = np.array([i / 10 for i in range(40)])
        K= -572.872341576*V - 0.89982819062*T + 801.704037275 + 354.67789388*V**2
        B= -0.33631875971*V - 0.0559766169332*T + 4.9467224951
        Q = 4000
        nu = 0.3
        y2 = K/(1 + Q * np.exp(-B*x))**nu
        return y2


    x = np.array([i / 10 for i in range(40)])
    plt.figure(figsize=(8,8), dpi= 100)
    plt.plot(x, obj2(), '-', color = 'blue', label = 'predict curve', linewidth = 3.0)
    plt.legend(fontsize = 24)
    plt.title(f'V={V}L/min T={T}℃', fontsize = 30)
    plt.xlabel('Pressure/MPa', fontsize = 30)
    plt.ylabel('Capacity/L', fontsize = 30)

    plt.tick_params(labelsize = 15)



    ax = plt.gca()
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    # Close plot to prevent resource leaks
    plt.close()
    
    # Return the base64-encoded image
    return jsonify({'plotBase64': image_base64})

@app.route('/generate_plot_release', methods=['POST'])
def generate_plot_release():
    data = request.json
    V = float(data['x'])
    T = float(data['y'])

    def obj2():     
        nonlocal V
        nonlocal T        
        x = np.array([i / 10 for i in range(40)])
        A = -467.543856173*V - 2.94691441564*T + 897.98773058 + 153.048114948*V**2
        K = 0.159775370844*V - 0.0970233190479*T + 7.72009208192
        C = 0.178672195081*V - 0.0550430835279*T + 4.66345107942
        y = A/(1+np.exp(-K*(x-C)))
        return y


    x = np.array([i / 10 for i in range(40)])
    plt.figure(figsize=(8,8), dpi= 100)
    plt.plot(x, obj2(), '-', color = 'blue', label = 'predict curve', linewidth = 3.0)
    plt.legend(fontsize = 24)
    plt.title(f'V={V}L/min T={T}℃', fontsize = 30)
    plt.xlabel('Pressure/MPa', fontsize = 30)
    plt.ylabel('Capacity/L', fontsize = 30)

    plt.tick_params(labelsize = 15)



    ax = plt.gca()
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    # Close plot to prevent resource leaks
    plt.close()
    
    # Return the base64-encoded image
    return jsonify({'plotBase64': image_base64})

if __name__ == '__main__':
    app.run(debug=True)
