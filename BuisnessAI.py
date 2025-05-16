import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load secret from .env
load_dotenv()
API_KEY = os.getenv("apikey")

# ------------------ Business Intelligence Assistant ------------------ #
class BIAssistant:
    def __init__(self):
        self.data = None
        self.column_map = {
            'date': None,
            'product': None,
            'region': None,
            'sales': None,
            'customer_age': None
        }
        self.stats_summary = ""

    def load_data(self, file):
        try:
            self.data = pd.read_csv(file)
            self._detect_columns()

            if self.column_map['date']:
                self.data[self.column_map['date']] = pd.to_datetime(self.data[self.column_map['date']], errors='coerce')

            return "Data loaded successfully!", self._get_data_preview()
        except Exception as e:
            return f"Error loading data: {e}", None

    def _detect_columns(self):
        for col in self.data.columns:
            col_lower = col.lower()
            for expected in self.column_map.keys():
                if expected in col_lower:
                    self.column_map[expected] = col

    def _get_data_preview(self):
        return f"""
        <h3>Data Preview</h3>
        <p>Rows: {len(self.data)}, Columns: {len(self.data.columns)}</p>
        {self.data.head().to_html()}
        """

    def _generate_statistical_summaries(self):
        summary = "Business Data Summary:\n"
        product_col = self.column_map['product']
        sales_col = self.column_map['sales']
        date_col = self.column_map['date']

        if product_col and sales_col:
            try:
                top_products = self.data.groupby(product_col)[sales_col].sum().nlargest(5)
                summary += f"\nTop 5 Products by Sales:\n{top_products.to_string()}\n"
            except:
                summary += "\nProduct analysis unavailable\n"

        if date_col and sales_col:
            try:
                monthly_sales = self.data.groupby(pd.Grouper(key=date_col, freq='M'))[sales_col].sum()
                summary += f"\nLast 3 Months Sales:\n{monthly_sales.tail(3).to_string()}\n"
            except:
                summary += "\n Sales trend analysis unavailable\n"

        return summary

    def ask_question(self, question):
        load_dotenv()
        API_KEY = os.getenv("apikey")

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=API_KEY,
        )

        response = client.chat.completions.create(
            model="nvidia/llama-3.3-nemotron-super-49b-v1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing summarised answer for the content. With the use of the following " + self.stats_summary},
                {"role": "user", "content": question}
            ],
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,
            frequency_penalty=0,
            presence_penalty=0,
            stream=False
        )
        return response.choices[0].message.content.strip().replace("*", "")

    def generate_forecast_plot(self):
        try:
            date_col = self.column_map['date']
            sales_col = self.column_map['sales']
            product_col = self.column_map['product']
            if not date_col or not sales_col:
                raise ValueError("Date or Sales column not found.")

            df = self.data[[date_col, sales_col]].dropna()
            df = df.rename(columns={date_col: "ds", sales_col: "y"})
            df = df.groupby("ds").sum().reset_index()
            df = df.sort_values("ds")

            df['ds_ordinal'] = df['ds'].map(datetime.toordinal)
            X = df[['ds_ordinal']]
            y = df['y']

            model = LinearRegression()
            model.fit(X, y)

            future_dates = pd.date_range(start=df['ds'].max(), periods=4, freq='M')
            future_ordinal = future_dates.map(datetime.toordinal).to_numpy().reshape(-1, 1)
            predictions = model.predict(future_ordinal)

            # Main Forecast Plot
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(df['ds'], df['y'], label="Historical Sales")
            ax1.plot(future_dates, predictions, 'r--', label="Forecast")
            ax1.set_title("Monthly Sales Forecast")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Sales")
            ax1.legend()
            ax1.grid(True)

            # Forecast Sales by Product
            fig2 = None
            if product_col:
                product_df = self.data[[product_col, sales_col]].dropna()
                top_products = product_df.groupby(product_col).sum().nlargest(5, sales_col)

                delta = top_products[sales_col].mean() * 0.1
                future_product_sales = top_products.copy()
                future_product_sales[sales_col] = future_product_sales[sales_col] + delta

                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.bar(top_products.index, top_products[sales_col], label="Current Sales", alpha=0.7)
                ax2.bar(future_product_sales.index, future_product_sales[sales_col], label="Forecasted Sales", alpha=0.7, color='orange')
                ax2.set_title("🛒 Sales Forecast for Top 5 Products")
                ax2.set_ylabel("Sales")
                ax2.legend()
                ax2.grid(True)

            return fig1, fig2
        except Exception as e:
            print(f"Forecasting error: {e}")
            return None, None

# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="Business Intelligence Assistant", layout="wide")
st.title("AI-Powered Business Intelligence Assistant")

if 'assistant' not in st.session_state:
    st.session_state.assistant = BIAssistant()
assistant = st.session_state.assistant

tab1, tab2, tab3 = st.tabs(["Data Setup", "Ask Questions", "Forecasting"])

# ---- Tab 1: Load and Initialize ----
with tab1:
    st.header("Step 1: Upload Your Business CSV Data")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file and st.button("Load Data"):
        status, preview = assistant.load_data(uploaded_file)
        if "successfully" in status:
            st.success(status)
            st.markdown(preview, unsafe_allow_html=True)
        else:
            st.error(status)

# ---- Tab 2: Ask Questions ----
with tab2:
    st.header("Step 2: Ask Business Questions")
    question = st.text_input("Type your question (e.g., What are the top-selling products?)")
    assistant.stats_summary = assistant._generate_statistical_summaries()

    if st.button("Get Answer"):
        if assistant.stats_summary:
            answer = assistant.ask_question(question)
            st.text_area("Answer", value=answer, height=200)

# ---- Tab 3: Forecast ----
with tab3:
    st.header("Step 3: Forecast Sales")

    if st.button("Generate Forecast"):
        if assistant.data is not None:
            fig1, fig2 = assistant.generate_forecast_plot()
            if fig1:
                st.subheader("1. Time-based Sales Forecast")
                st.pyplot(fig1)
            if fig2:
                st.subheader("2. Product-wise Sales Forecast")
                st.pyplot(fig2)
            if not fig1 and not fig2:
                st.error("Could not generate any forecast.")
        else:
            st.warning("Please load and initialize data.")

# ---- Style ----
st.markdown("""
<style>
.stTextInput input, .stTextArea textarea {
    border-radius: 8px !important;
}
.stButton>button {
    background-color: #4f46e5 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
}
</style>
""", unsafe_allow_html=True)
