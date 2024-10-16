from flask import Flask, request, jsonify
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster

app = Flask(__name__)

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Perform hierarchical clustering
    Z = sch.linkage(scaled_data, method='ward')
    clusters = fcluster(Z, t=3, criterion='maxclust')
    
    # Add clusters to the data
    df['Cluster'] = clusters
    
    # Return the results as JSON
    return jsonify(df.to_dict(orient='records'))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
