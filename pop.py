from typing import List, Dict
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from datetime import datetime
import groq


# API Configuration
GROQ_API_KEY = "Input Your Groq API key here"  # Replace with your actual Groq API key
import requests

def list_groq_models():
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        models = response.json()
        return models
    else:
        print(f"Error: Unable to fetch models. Status code: {response.status_code}")
        print(response.json())
        return None

# Fetch and print available models
models = list_groq_models()
if models:
    print("Available Groq Models:")
    for model in models.get("data", []):
        print(f"- {model['id']}")
class FinancialExpertAgent:
    def __init__(self):
        self.groq_client = groq.Groq(api_key=GROQ_API_KEY)
        self.expert_context = """
        You are an expert in Financial Analysis , Financial Planning , Financial Advice.
        Analyse the Question Provided.
        Generate a Tailored answer for the question
        The tailored answer has to be elaborate
        Also recommend some investing tips in relation to the question
        Format the answer to a proper format so that it is acceptable in HTML formmat
        """
        
    def get_expert_advice(self, user_query: str, risk_tolerance: str = "medium", 
                        investment_horizon: str = "5-10 years") -> Dict:
        """
        Provide expert financial guidance based on natural language queries
        Args:
            user_query: Financial question/request from user
            risk_tolerance: low/medium/high
            investment_horizon: short-term (<3y), medium-term (3-10y), long-term (>10y)
        """
        prompt = f"""
        {self.expert_context}
        
        User Query: {user_query}
        
        Provide detailed advice covering:
        1. Current financial best practices
        2. Regulatory considerations (RBI guidelines, SEBI regulations)
        3. Tax implications
        4. Risk management strategies
        5. Recommended asset allocation
        6. Elaborate Answer Pertaining to the query

        Options 1 through 5 can be short and fitted into a small paragraph called advice and the 6th option should be elaborate with lot of key details
        """
        
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": self.expert_context},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        return {
            "query": user_query,
            "advice": response.choices[0].message.content
        }

class FinanceAgent:
    def __init__(self):
        self.groq_client = groq.Groq(api_key=GROQ_API_KEY)
        self.fraud_detector = IsolationForest(contamination=0.1, random_state=42)
        self.expense_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def predict_expenses(self, historical_data: List[Dict]) -> float:
        """Predict future expenses using ML and AI insights"""
        # Prepare features for ML model
        X = np.array([[
            tx['amount'],
            tx['day_of_month'],
            tx['category_code']
        ] for tx in historical_data])
        
        y = np.array([tx['amount'] for tx in historical_data])
        
        # Train and predict
        self.expense_predictor.fit(X[:-1], y[:-1])
        next_prediction = self.expense_predictor.predict(X[-1].reshape(1, -1))[0]
        
        # Get AI insights
        prompt = f"""
        Based on the following spending pattern:
        - Average spend: ${np.mean(y):.2f}
        - Recent spend: ${y[-1]:.2f}
        - Predicted next spend: ${next_prediction:.2f}
        
        Provide a brief analysis of the spending trend and what it indicates for future expenses.
        """
        
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "predicted_amount": float(next_prediction),
            "analysis": response.choices[0].message.content
        }
    
    def detect_fraud(self, transaction: Dict, historical_transactions: List[Dict]) -> Dict:
        """Detect potential fraud using anomaly detection"""
        # Prepare features for anomaly detection
        features = np.array([[
            tx['amount'],
            tx['hour_of_day'],
            tx['distance_from_usual_location'],
            tx['merchant_category_code']
        ] for tx in historical_transactions])
        
        # Train the model and predict
        self.fraud_detector.fit(features)
        
        current_features = np.array([[
            transaction['amount'],
            transaction['hour_of_day'],
            transaction['distance_from_usual_location'],
            transaction['merchant_category_code']
        ]])
        
        is_fraud = self.fraud_detector.predict(current_features)[0] == -1
        fraud_score = float(self.fraud_detector.score_samples(current_features)[0])
        
        # Get AI analysis
        prompt = f"""
        Analyze this transaction for potential fraud:
        Amount: ${transaction['amount']}
        Time: {transaction['hour_of_day']}:00
        Distance from usual location: {transaction['distance_from_usual_location']}km
        Merchant category: {transaction['merchant_category_code']}
        Anomaly score: {fraud_score}
        
        Provide a brief explanation of why this might or might not be fraudulent.
        """
        
        response = self.groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # Updated from llama2-70b-4096
    messages=[{"role": "user", "content": prompt}]
)

        
        return {
            "is_fraudulent": is_fraud,
            "fraud_score": fraud_score,
            "analysis": response.choices[0].message.content
        }
    
    def get_financial_advice(self, financial_data: Dict) -> Dict:
        """Generate personalized financial advice"""
        # Calculate key financial metrics
        monthly_income = financial_data['monthly_income']
        monthly_expenses = financial_data['monthly_expenses']
        savings = financial_data['savings']
        debt = financial_data.get('debt', 0)
        
        # Prepare financial health indicators
        savings_ratio = savings / monthly_income if monthly_income > 0 else 0
        expense_ratio = monthly_expenses / monthly_income if monthly_income > 0 else 1
        debt_ratio = debt / monthly_income if monthly_income > 0 else 0
        
        prompt = prompt = f"""
        Based on the following financial metrics:
        - Monthly Income: ${monthly_income}
        - Monthly Expenses: ${monthly_expenses} ({expense_ratio*100:.1f}% of income)
        - Savings: ${savings} ({savings_ratio*100:.1f}% of income)
        - Debt: ${debt} ({debt_ratio*100:.1f}% of income)
        
        Provide personalized financial advice covering:
        1. Spending patterns and recommendations
        2. Savings strategy
        3. Debt management (if applicable)
        4. Investment suggestions
        5. Specific actionable steps for improvement
        6. Give Question Specific Answers 
        
        Keep the advice concise and actionable.
        """
        
        response = self.groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # Updated from llama2-70b-4096
    messages=[{"role": "user", "content": prompt}]
)

        
        return {
            "metrics": {
                "savings_ratio": savings_ratio,
                "expense_ratio": expense_ratio,
                "debt_ratio": debt_ratio
            },
            "advice": response.choices[0].message.content
        }

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = FinanceAgent()
    # Initialize expert agent
    expert = FinancialExpertAgent()
    
    # Example expert consultation
    user_question = input("Enter Your Financial Query")
    expert_advice = expert.get_expert_advice(
        user_question
    )
    
    financial_data = {
        "monthly_income": int(input("Enter Monthly Income")),
        "monthly_expenses": int(input("Monthly Expenses")),
        "savings": int(input("Current Savings")),
        "debt": int(input("Current Debt"))
    }
    
    
    
    
    # Test financial advice
    advice_result = agent.get_financial_advice(financial_data)
    print("\nFinancial Advice:")
    print(f"Savings Ratio: {advice_result['metrics']['savings_ratio']:.2%}")
    print(f"Expense Ratio: {advice_result['metrics']['expense_ratio']:.2%}")
    print(f"Advice: {advice_result['advice']}")
