from flask import Flask
from predict_4 import predict_product
app = Flask(__name__)

@app.route('/')
def home():
    return {
        'status': 200,
        'message': 'Hello, this is the XAI XDL homepage!'
    }


@app.route('/predict/<reactant_smiles>')
def predict(reactant_smiles):
    product_smiles = predict_product(reactant_smiles)
    return product_smiles


    
@app.route('/health')
def health():
    return {
        'status': 'healthy',
        'message': 'The XAI XDL API is up and running!'
    }
 

    
if __name__ == "__main__":
    app.run(debug=True)