from flask import Flask, request, jsonify
from flask_cors import CORS
from pop import FinancialExpertAgent

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize FinancialExpertAgent
expert_agent = FinancialExpertAgent()

@app.route('/get_expert_advice', methods=['POST'])
def get_expert_advice():
    data = request.get_json()
    if not data or 'user_query' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    user_query = data['user_query']
    try:
        advice_response = expert_agent.get_expert_advice(user_query)
        
        # Format the response properly before sending it back
        formatted_advice = f"""
### Query:
{advice_response['query']}

### Advice:
{advice_response['advice']}
"""
        
        return jsonify({
            "query": advice_response["query"],
            "advice": formatted_advice.strip()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)