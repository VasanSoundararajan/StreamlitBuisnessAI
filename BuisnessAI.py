import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from sklearn.linear_model import LinearRegression
import torch
from openai import OpenAI

import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNING"] = "true"
os.environ["PYTORCH_JIT"] = "0"

class BIAssistant:
    def __init__(self):
        self.data = None
        self.vectorstore = None
        self.qa_chain = None
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=  ""# Replace with a secure variable in production
        )
        self.column_map = {
            'date': None,
            'product': None,
            'region': None,
            'sales': None,
            'customer_age': None
        }

    def load_data(self, uploaded_file):
        try:
            self.data = pd.read_csv(uploaded_file)
            self._detect_columns()
            if self.column_map['date']:
                self.data[self.column_map['date']] = pd.to_datetime(
                    self.data[self.column_map['date']], errors='coerce')
            return "Data loaded successfully!"
        except Exception as e:
            return f"Error loading data: {str(e)}"

    def _detect_columns(self):
        for col in self.data.columns:
            col_lower = col.lower()
            for expected_col in self.column_map.keys():
                if expected_col in col_lower:
                    self.column_map[expected_col] = col

    def _generate_statistical_summaries(self):
        summary = "Business Data Summary:\n"
        product_col = self.column_map['product']
        sales_col = self.column_map['sales']

        if product_col and sales_col:
            try:
                top_products = self.data.groupby(product_col)[sales_col].sum().nlargest(5)
                summary += f"\nTop 5 Products by Sales:\n{top_products.to_string()}\n"
            except:
                summary += "\nProduct analysis unavailable\n"

        date_col = self.column_map['date']
        if date_col and sales_col:
            try:
                monthly_sales = self.data.groupby(pd.Grouper(key=date_col, freq='ME'))[sales_col].sum()
                summary += f"\nLast 3 Months Sales:\n{monthly_sales.tail(3).to_string()}\n"
            except:
                summary += "\nSales trend analysis unavailable\n"

        return summary

    def generate_sales_plot_with_prediction(self, future_months=3):
        date_col = self.column_map['date']
        sales_col = self.column_map['sales']

        if not date_col or not sales_col:
            return None, None, "Required columns not found."

        df = self.data.copy()
        df = df[[date_col, sales_col]].dropna()
        df = df.groupby(pd.Grouper(key=date_col, freq='M')).sum().reset_index()

        df['month_number'] = np.arange(len(df))
        X = df[['month_number']]
        y = df[sales_col]

        model = LinearRegression()
        model.fit(X, y)

        future_X = np.arange(len(df), len(df) + future_months).reshape(-1, 1)
        future_dates = pd.date_range(start=df[date_col].max() + pd.DateOffset(months=1), periods=future_months, freq='M')
        future_preds = model.predict(future_X)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df[date_col], y, label="Historical Sales", marker='o')
        ax.plot(future_dates, future_preds, label="Predicted Sales", marker='x', linestyle='--', color='orange')
        ax.set_title("Monthly Sales with Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        forecast_df = pd.DataFrame({
            'Date': list(df[date_col]) + list(future_dates),
            'Sales': list(y) + list(future_preds),
            'Type': ['Historical'] * len(df) + ['Predicted'] * future_months
        })

        return fig, forecast_df, None

    def initialize_system(self):
        try:
            self.stats_summary = self._generate_statistical_summaries()
            return "System initialized successfully!"
        except Exception as e:
            return f"Initialization failed: {str(e)}"

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
                monthly_sales = self.data.groupby(pd.Grouper(key=date_col, freq='ME'))[sales_col].sum()
                summary += f"\nLast 3 Months Sales:\n{monthly_sales.tail(3).to_string()}\n"
            except:
                summary += "\nSales trend analysis unavailable\n"

        return summary

    def ask_question(self, question):
        try:
            if not self.stats_summary:
                return "System not initialized. Please load data and initialize first."

            system_message = (
                "You are a business intelligence assistant. Use the provided business data "
                "to answer user queries with clear insights.\n\n"
                + self.stats_summary
            )

            completion = self.client.chat.completions.create(
                model="nvidia/llama-3.3-nemotron-super-49b-v1",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question}
                ],
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,
                frequency_penalty=0,
                presence_penalty=0,
                stream=False
            )

            return completion.choices[0].message.content.strip().replace('*', '')

        except Exception as e:
            return f"Error answering question: {str(e)}"

    def generate_plot(self, plot_type):
        try:
            date_col = self.column_map['date']
            sales_col = self.column_map['sales']
            product_col = self.column_map['product']
            fig, ax = plt.subplots(figsize=(10, 6))

            if plot_type == "Sales Trends" and date_col and sales_col:
                sales = self.data.groupby(pd.Grouper(key=date_col, freq='ME'))[sales_col].sum()
                ax.plot(sales.index, sales.values, marker='o')
                ax.set_title("Monthly Sales Trends")
                ax.set_xlabel("Month")
                ax.set_ylabel("Total Sales")
                plt.xticks(rotation=45)
                plt.tight_layout()
                return fig

            elif plot_type == "Product Performance" and product_col and sales_col:
                top_products = self.data.groupby(product_col)[sales_col].sum().nlargest(10)
                top_products.plot(kind='bar', ax=ax)
                ax.set_title("Top Products by Sales")
                ax.set_xlabel("Product")
                ax.set_ylabel("Total Sales")
                plt.tight_layout()
                return fig

        except Exception as e:
            st.error(f"Plot error: {str(e)}")
            return None

# ========== Streamlit Interface ==========
st.set_page_config(page_title="AI BI Assistant", layout="wide")
st.title("AI-Powered Business Intelligence Assistant")

assistant = BIAssistant()

st.sidebar.header("Step 1: Upload CSV Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file:
    status = assistant.load_data(uploaded_file)
    st.sidebar.success(status)

    if st.sidebar.button("Initialize System"):
        init_status = assistant.initialize_system()
        st.sidebar.success(init_status)

    st.sidebar.markdown("---")
    forecast_months = st.sidebar.slider("Forecast Months", 1, 12, 3)
    chart_option = st.sidebar.selectbox("Choose Plot Type", ["Sales Trends", "Product Performance"])

if assistant.data is not None:
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Ask Question", "Visualize & Forecast"])

    with tab1:
        st.subheader("Preview of Uploaded Data")
        st.dataframe(assistant.data.head())

    with tab2:
        st.subheader("Ask a Business Question")

        if assistant.qa_chain is None:
            st.warning("Please initialize the system from the sidebar first.")
        else:
            user_question = st.text_input("Ask a question about your data", placeholder="e.g. What are the top performing products?")
            if st.button("Ask"):
                if user_question.strip():
                    with st.spinner("Analyzing..."):
                        response = assistant.ask_question(user_question)
                    st.markdown("### Assistant's Response")
                    st.text_area("Response", value=response, height=220)
                else:
                    st.warning("Please enter a question.")

            st.markdown("#### Example Prompts")
            examples = [
                "What are the top products by revenue?",
                "Which region has shown the highest growth?",
                "What is the sales trend over the past three months?",
            ]
            for example in examples:
                if st.button(f"Try: {example}"):
                    with st.spinner("Analyzing..."):
                        response = assistant.ask_question(example)
                    st.text_area("Response", value=response, height=220)

    with tab3:
        st.subheader("Generate Plot")
        if st.button("Generate Plot"):
            fig = assistant.generate_plot(chart_option)
            if fig:
                st.pyplot(fig)
            else:
                st.warning("Plot could not be  generated due to missing data.")

        st.subheader("Forecast Sales")
        if st.button("Generate Forecast"):
            fig, forecast_df, error = assistant.generate_sales_plot_with_prediction(forecast_months)
            if error:
                st.error(error)
            else:
                st.pyplot(fig)
                st.dataframe(forecast_df)
                csv = forecast_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Forecast CSV", data=csv, file_name="sales_forecast.csv", mime='text/csv')
