# import plotly.express as px

# def show_explore_page():
#     st.title("📊 Explore Software Engineer Salaries")
#     st.write("### Data from Stack Overflow Developer Survey 2024")

#     data = df["Country"].value_counts().reset_index()
#     data.columns = ["Country", "Count"]

#     fig = px.bar(data, x="Country", y="Count", color="Country",
#                  title="Number of Responses by Country",
#                  labels={"Count": "Number of Developers"})
#     st.plotly_chart(fig)
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime

# # Set page configuration
# st.set_page_config(page_title="Explore Data", layout="wide")

# # Custom CSS for styling
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f5e6cc; /* Warm Beige, matching predict_page */
#         font-family: 'Arial', sans-serif;
#     }
#     .stButton>button {
#         background-color: #2c3e50;
#         color: white;
#         border-radius: 5px;
#         padding: 10px 20px;
#         font-size: 16px;
#     }
#     .stButton>button:hover {
#         background-color: #34495e;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Title and description
# st.title("📊 Explore Stack Overflow Developer Survey 2024")
# st.markdown("### Analyze trends and distributions in the developer survey data.")
# st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}", unsafe_allow_html=True)

# # Load and clean data
# @st.cache_data
# def load_data():
#     df = pd.read_csv("C:/Users/GOHIL RAJENDRASINH/Downloads/stack-overflow-developer-survey-2024/survey_results_public.csv")
#     necessary_columns = ['Age', 'EdLevel', 'YearsCodePro', 'Country', 'Industry', 'LanguageHaveWorkedWith', 
#                         'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith', 'WorkExp', 'ConvertedCompYearly']
#     df = df[necessary_columns].copy()
#     df = df.dropna(subset=necessary_columns)  # Drop rows with missing values in key columns
    
#     # Convert Age to numeric midpoints (as in predict_page)
#     age_mapping = {
#         "Under 18 years old": 14,
#         "18-24 years old": 21,
#         "25-34 years old": 29.5,
#         "35-44 years old": 39.5,
#         "45-54 years old": 49.5,
#         "55-64 years old": 59.5,
#         "65 years or older": 70
#     }
#     df['Age'] = df['Age'].map(age_mapping).astype(float)
    
#     # Convert YearsCodePro and WorkExp to numeric, handling non-numeric values
#     for col in ['YearsCodePro', 'WorkExp']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # Take first value for multi-value columns
#     for col in ['LanguageHaveWorkedWith', 'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith']:
#         df[col] = df[col].apply(lambda x: x.split(';')[0] if isinstance(x, str) else x)
    
#     # Education mapping
#     ed_level_mapping = {
#         "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
#         "Some college/university study without earning a degree": 4,
#         "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
#         "Primary/elementary school": 2,
#         "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8,
#         "Associate degree (A.A., A.S., etc.)": 5,
#         "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
#         "Something else": 1
#     }
#     df['EdLevel'] = df['EdLevel'].map(ed_level_mapping).astype(float)
    
#     return df

# df = load_data()

# # Sidebar for filters
# st.sidebar.header("Filter Data")
# country_filter = st.sidebar.multiselect("Select Countries", options=df['Country'].unique(), default=df['Country'].unique())
# industry_filter = st.sidebar.multiselect("Select Industries", options=df['Industry'].unique(), default=df['Industry'].unique())
# min_salary = st.sidebar.slider("Minimum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=0)
# max_salary = st.sidebar.slider("Maximum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=int(df['ConvertedCompYearly'].max()))

# # Filter data
# filtered_df = df[
#     (df['Country'].isin(country_filter)) &
#     (df['Industry'].isin(industry_filter)) &
#     (df['ConvertedCompYearly'] >= min_salary) &
#     (df['ConvertedCompYearly'] <= max_salary)
# ].copy()

# # Summary Statistics
# st.header("📈 Summary Statistics")
# st.write(filtered_df.describe())

# # Distribution Plots
# st.header("📊 Distributions")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Salary Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='ConvertedCompYearly', bins=30, ax=ax)
#     ax.set_title("Distribution of Annual Salary")
#     ax.set_xlabel("Salary ($)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Age Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='Age', bins=20, ax=ax)
#     ax.set_title("Distribution of Age")
#     ax.set_xlabel("Age (Midpoint)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Categorical Bar Charts
# st.header("🏷️ Categorical Analysis")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Top 10 Countries by Count")
#     top_countries = filtered_df['Country'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_countries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Countries")
#     ax.set_xlabel("Country")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Top 10 Industries by Count")
#     top_industries = filtered_df['Industry'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_industries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Industries")
#     ax.set_xlabel("Industry")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Correlation Heatmap
# st.header("🔍 Correlation Heatmap")
# numeric_df = filtered_df[['Age', 'EdLevel', 'YearsCodePro', 'WorkExp', 'ConvertedCompYearly']].dropna()
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
# ax.set_title("Correlation Between Numeric Features")
# st.pyplot(fig)

# # Interactive Data Table
# st.header("📋 Raw Data Preview")
# st.write(filtered_df.head(10))  # Show first 10 rows of filtered data

# if st.button("Download Filtered Data as CSV"):
#     csv = filtered_df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">Download Filtered Data</a>'
#     st.markdown(href, unsafe_allow_html=True)

# if __name__ == "__main__":
#     show_predict_page()

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import base64

# # Set page configuration
# st.set_page_config(page_title="Explore Data", layout="wide")

# # Custom CSS for styling
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f5e6cc; /* Warm Beige, matching predict_page */
#         font-family: 'Arial', sans-serif;
#     }
#     .stButton>button {
#         background-color: #2c3e50;
#         color: white;
#         border-radius: 5px;
#         padding: 10px 20px;
#         font-size: 16px;
#     }
#     .stButton>button:hover {
#         background-color: #34495e;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Title and description
# st.title("📊 Explore Stack Overflow Developer Survey 2024")
# st.markdown("### Analyze trends and distributions in the developer survey data.")
# st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}", unsafe_allow_html=True)

# # Load and clean data
# @st.cache_data
# def load_data():
#     df = pd.read_csv("C:/Users/GOHIL RAJENDRASINH/Downloads/stack-overflow-developer-survey-2024/survey_results_public.csv")
#     necessary_columns = ['Age', 'EdLevel', 'YearsCodePro', 'Country', 'Industry', 'LanguageHaveWorkedWith', 
#                         'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith', 'WorkExp', 'ConvertedCompYearly']
#     df = df[necessary_columns].copy()
#     df = df.dropna(subset=necessary_columns)  # Drop rows with missing values in key columns
    
#     # Convert Age to numeric midpoints
#     age_mapping = {
#         "Under 18 years old": 14,
#         "18-24 years old": 21,
#         "25-34 years old": 29.5,
#         "35-44 years old": 39.5,
#         "45-54 years old": 49.5,
#         "55-64 years old": 59.5,
#         "65 years or older": 70
#     }
#     df['Age'] = df['Age'].map(age_mapping).astype(float)
    
#     # Convert YearsCodePro and WorkExp to numeric, handling non-numeric values
#     for col in ['YearsCodePro', 'WorkExp']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # Take first value for multi-value columns
#     for col in ['LanguageHaveWorkedWith', 'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith']:
#         df[col] = df[col].apply(lambda x: x.split(';')[0] if isinstance(x, str) else x)
    
#     # Education mapping
#     ed_level_mapping = {
#         "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
#         "Some college/university study without earning a degree": 4,
#         "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
#         "Primary/elementary school": 2,
#         "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8,
#         "Associate degree (A.A., A.S., etc.)": 5,
#         "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
#         "Something else": 1
#     }
#     df['EdLevel'] = df['EdLevel'].map(ed_level_mapping).astype(float)
    
#     return df

# df = load_data()

# # Sidebar for filters
# st.sidebar.header("Filter Data")
# country_filter = st.sidebar.multiselect("Select Countries", options=df['Country'].unique(), default=df['Country'].unique())
# industry_filter = st.sidebar.multiselect("Select Industries", options=df['Industry'].unique(), default=df['Industry'].unique())
# min_salary = st.sidebar.slider("Minimum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=0)
# max_salary = st.sidebar.slider("Maximum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=int(df['ConvertedCompYearly'].max()))

# # Filter data
# filtered_df = df[
#     (df['Country'].isin(country_filter)) &
#     (df['Industry'].isin(industry_filter)) &
#     (df['ConvertedCompYearly'] >= min_salary) &
#     (df['ConvertedCompYearly'] <= max_salary)
# ].copy()

# # Summary Statistics
# st.header("📈 Summary Statistics")
# st.write(filtered_df.describe())

# # Distribution Plots
# st.header("📊 Distributions")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Salary Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='ConvertedCompYearly', bins=30, ax=ax)
#     ax.set_title("Distribution of Annual Salary")
#     ax.set_xlabel("Salary ($)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Age Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='Age', bins=20, ax=ax)
#     ax.set_title("Distribution of Age")
#     ax.set_xlabel("Age (Midpoint)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Categorical Bar Charts
# st.header("🏷️ Categorical Analysis")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Top 10 Countries by Count")
#     top_countries = filtered_df['Country'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_countries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Countries")
#     ax.set_xlabel("Country")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Top 10 Industries by Count")
#     top_industries = filtered_df['Industry'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_industries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Industries")
#     ax.set_xlabel("Industry")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Correlation Heatmap
# st.header("🔍 Correlation Heatmap")
# numeric_df = filtered_df[['Age', 'EdLevel', 'YearsCodePro', 'WorkExp', 'ConvertedCompYearly']].dropna()
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
# ax.set_title("Correlation Between Numeric Features")
# st.pyplot(fig)

# # Interactive Data Table
# st.header("📋 Raw Data Preview")
# st.write(filtered_df.head(10))  # Show first 10 rows of filtered data

# if st.button("Download Filtered Data as CSV"):
#     csv = filtered_df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">Download Filtered Data</a>'
#     st.markdown(href, unsafe_allow_html=True)

# def show_explore_page():
#     # This function is called by app.py
#     pass  # The rest of the code above is executed when this file is run

# if __name__ == "__main__":
#     show_explore_page()


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime

# # Set page configuration
# st.set_page_config(page_title="Explore Data", layout="wide")

# # Custom CSS for styling
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f5e6cc; /* Warm Beige, matching predict_page */
#         font-family: 'Arial', sans-serif;
#     }
#     .stButton>button {
#         background-color: #2c3e50;
#         color: white;
#         border-radius: 5px;
#         padding: 10px 20px;
#         font-size: 16px;
#     }
#     .stButton>button:hover {
#         background-color: #34495e;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Title and description
# st.title("📊 Explore Stack Overflow Developer Survey 2024")
# st.markdown("### Analyze trends and distributions in the developer survey data.")
# st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}", unsafe_allow_html=True)

# # Load and clean data
# @st.cache_data
# def load_data():
#     df = pd.read_csv("C:/Users/GOHIL RAJENDRASINH/Downloads/stack-overflow-developer-survey-2024/survey_results_public.csv")
#     necessary_columns = ['Age', 'EdLevel', 'YearsCodePro', 'Country', 'Industry', 'LanguageHaveWorkedWith', 
#                         'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith', 'WorkExp', 'ConvertedCompYearly']
#     df = df[necessary_columns].copy()
#     df = df.dropna(subset=necessary_columns)  # Drop rows with missing values in key columns
    
#     # Convert Age to numeric midpoints
#     age_mapping = {
#         "Under 18 years old": 14,
#         "18-24 years old": 21,
#         "25-34 years old": 29.5,
#         "35-44 years old": 39.5,
#         "45-54 years old": 49.5,
#         "55-64 years old": 59.5,
#         "65 years or older": 70
#     }
#     df['Age'] = df['Age'].map(age_mapping).astype(float)
    
#     # Convert YearsCodePro and WorkExp to numeric, handling non-numeric values
#     for col in ['YearsCodePro', 'WorkExp']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # Take first value for multi-value columns
#     for col in ['LanguageHaveWorkedWith', 'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith']:
#         df[col] = df[col].apply(lambda x: x.split(';')[0] if isinstance(x, str) else x)
    
#     # Education mapping
#     ed_level_mapping = {
#         "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
#         "Some college/university study without earning a degree": 4,
#         "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
#         "Primary/elementary school": 2,
#         "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8,
#         "Associate degree (A.A., A.S., etc.)": 5,
#         "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
#         "Something else": 1
#     }
#     df['EdLevel'] = df['EdLevel'].map(ed_level_mapping).astype(float)
    
#     return df

# df = load_data()

# # Sidebar for filters
# st.sidebar.header("Filter Data")
# country_filter = st.sidebar.multiselect("Select Countries", options=df['Country'].unique(), default=df['Country'].unique())
# industry_filter = st.sidebar.multiselect("Select Industries", options=df['Industry'].unique(), default=df['Industry'].unique())
# min_salary = st.sidebar.slider("Minimum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=0)
# max_salary = st.sidebar.slider("Maximum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=int(df['ConvertedCompYearly'].max()))

# # Filter data
# filtered_df = df[
#     (df['Country'].isin(country_filter)) &
#     (df['Industry'].isin(industry_filter)) &
#     (df['ConvertedCompYearly'] >= min_salary) &
#     (df['ConvertedCompYearly'] <= max_salary)
# ].copy()

# # Summary Statistics
# st.header("📈 Summary Statistics")
# st.write(filtered_df.describe())

# # Distribution Plots
# st.header("📊 Distributions")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Salary Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='ConvertedCompYearly', bins=30, ax=ax)
#     ax.set_title("Distribution of Annual Salary")
#     ax.set_xlabel("Salary ($)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Age Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='Age', bins=20, ax=ax)
#     ax.set_title("Distribution of Age")
#     ax.set_xlabel("Age (Midpoint)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Categorical Bar Charts
# st.header("🏷️ Categorical Analysis")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Top 10 Countries by Count")
#     top_countries = filtered_df['Country'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_countries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Countries")
#     ax.set_xlabel("Country")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Top 10 Industries by Count")
#     top_industries = filtered_df['Industry'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_industries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Industries")
#     ax.set_xlabel("Industry")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Correlation Heatmap
# st.header("🔍 Correlation Heatmap")
# numeric_df = filtered_df[['Age', 'EdLevel', 'YearsCodePro', 'WorkExp', 'ConvertedCompYearly']].dropna()
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
# ax.set_title("Correlation Between Numeric Features")
# st.pyplot(fig)

# # Interactive Data Table
# st.header("📋 Raw Data Preview")
# st.write(filtered_df.head(10))  # Show first 10 rows of filtered data

# def show_explore_page():
#     pass  # Triggered by Streamlit multi-page, main code runs at module level

# if __name__ == "__main__":
#     show_explore_page()

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime

# # Set page configuration
# st.set_page_config(page_title="Explore Data", layout="wide")

# # Custom CSS for styling
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f5e6cc; /* Warm Beige, matching predict_page */
#         font-family: 'Arial', sans-serif;
#     }
#     .stButton>button {
#         background-color: #2c3e50;
#         color: white;
#         border-radius: 5px;
#         padding: 10px 20px;
#         font-size: 16px;
#     }
#     .stButton>button:hover {
#         background-color: #34495e;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Title and description
# st.title("📊 Explore Stack Overflow Developer Survey 2024")
# st.markdown("### Analyze trends and distributions in the developer survey data.")
# st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}", unsafe_allow_html=True)

# # Load and clean data
# @st.cache_data
# def load_data():
#     df = pd.read_csv("C:/Users/GOHIL RAJENDRASINH/Downloads/stack-overflow-developer-survey-2024/survey_results_public.csv")
#     necessary_columns = ['Age', 'EdLevel', 'YearsCodePro', 'Country', 'Industry', 'LanguageHaveWorkedWith', 
#                         'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith', 'WorkExp', 'ConvertedCompYearly']
#     df = df[necessary_columns].copy()
#     df = df.dropna(subset=necessary_columns)  # Drop rows with missing values in key columns
    
#     # Convert Age to numeric midpoints
#     age_mapping = {
#         "Under 18 years old": 14,
#         "18-24 years old": 21,
#         "25-34 years old": 29.5,
#         "35-44 years old": 39.5,
#         "45-54 years old": 49.5,
#         "55-64 years old": 59.5,
#         "65 years or older": 70
#     }
#     df['Age'] = df['Age'].map(age_mapping).astype(float)
    
#     # Convert YearsCodePro and WorkExp to numeric, handling non-numeric values
#     for col in ['YearsCodePro', 'WorkExp']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # Take first value for multi-value columns
#     for col in ['LanguageHaveWorkedWith', 'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith']:
#         df[col] = df[col].apply(lambda x: x.split(';')[0] if isinstance(x, str) else x)
    
#     # Education mapping
#     ed_level_mapping = {
#         "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
#         "Some college/university study without earning a degree": 4,
#         "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
#         "Primary/elementary school": 2,
#         "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8,
#         "Associate degree (A.A., A.S., etc.)": 5,
#         "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
#         "Something else": 1
#     }
#     df['EdLevel'] = df['EdLevel'].map(ed_level_mapping).astype(float)
    
#     return df

# df = load_data()

# # Sidebar for filters
# st.sidebar.header("Filter Data")
# country_filter = st.sidebar.multiselect("Select Countries", options=df['Country'].unique(), default=df['Country'].unique())
# industry_filter = st.sidebar.multiselect("Select Industries", options=df['Industry'].unique(), default=df['Industry'].unique())
# min_salary = st.sidebar.slider("Minimum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=0)
# max_salary = st.sidebar.slider("Maximum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=int(df['ConvertedCompYearly'].max()))

# # Filter data
# filtered_df = df[
#     (df['Country'].isin(country_filter)) &
#     (df['Industry'].isin(industry_filter)) &
#     (df['ConvertedCompYearly'] >= min_salary) &
#     (df['ConvertedCompYearly'] <= max_salary)
# ].copy()

# # Summary Statistics
# st.header("📈 Summary Statistics")
# st.write(filtered_df.describe())

# # Distribution Plots
# st.header("📊 Distributions")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Salary Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='ConvertedCompYearly', bins=30, ax=ax)
#     ax.set_title("Distribution of Annual Salary")
#     ax.set_xlabel("Salary ($)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Age Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='Age', bins=20, ax=ax)
#     ax.set_title("Distribution of Age")
#     ax.set_xlabel("Age (Midpoint)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Categorical Bar Charts
# st.header("🏷️ Categorical Analysis")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Top 10 Countries by Count")
#     top_countries = filtered_df['Country'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_countries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Countries")
#     ax.set_xlabel("Country")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Top 10 Industries by Count")
#     top_industries = filtered_df['Industry'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_industries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Industries")
#     ax.set_xlabel("Industry")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Correlation Heatmap
# st.header("🔍 Correlation Heatmap")
# numeric_df = filtered_df[['Age', 'EdLevel', 'YearsCodePro', 'WorkExp', 'ConvertedCompYearly']].dropna()
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
# ax.set_title("Correlation Between Numeric Features")
# st.pyplot(fig)

# # Interactive Data Table
# st.header("📋 Raw Data Preview")
# st.write(filtered_df.head(10))  # Show first 10 rows of filtered data

# def show_explore_page():
#     pass  # Triggered by Streamlit multi-page, main code runs at module level

# if __name__ == "__main__":
#     show_explore_page()


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import base64

# # Set page configuration
# st.set_page_config(page_title="Explore Data", layout="wide")

# # Custom CSS for styling
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f5e6cc; /* Warm Beige, matching predict_page */
#         font-family: 'Arial', sans-serif;
#     }
#     .stButton>button {
#         background-color: #2c3e50;
#         color: white;
#         border-radius: 5px;
#         padding: 10px 20px;
#         font-size: 16px;
#     }
#     .stButton>button:hover {
#         background-color: #34495e;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Title and description
# st.title("📊 Explore Stack Overflow Developer Survey 2024")
# st.markdown("### Analyze trends and distributions in the developer survey data.")
# st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}", unsafe_allow_html=True)

# # Load and clean data
# @st.cache_data
# def load_data():
#     df = pd.read_csv("C:/Users/GOHIL RAJENDRASINH/Downloads/stack-overflow-developer-survey-2024/survey_results_public.csv")
#     necessary_columns = ['Age', 'EdLevel', 'YearsCodePro', 'Country', 'Industry', 'LanguageHaveWorkedWith', 
#                         'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith', 'WorkExp', 'ConvertedCompYearly']
#     df = df[necessary_columns].copy()
#     df = df.dropna(subset=necessary_columns)  # Drop rows with missing values in key columns
    
#     # Convert Age to numeric midpoints
#     age_mapping = {
#         "Under 18 years old": 14,
#         "18-24 years old": 21,
#         "25-34 years old": 29.5,
#         "35-44 years old": 39.5,
#         "45-54 years old": 49.5,
#         "55-64 years old": 59.5,
#         "65 years or older": 70
#     }
#     df['Age'] = df['Age'].map(age_mapping).astype(float)
    
#     # Convert YearsCodePro and WorkExp to numeric, handling non-numeric values
#     for col in ['YearsCodePro', 'WorkExp']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # Take first value for multi-value columns
#     for col in ['LanguageHaveWorkedWith', 'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith']:
#         df[col] = df[col].apply(lambda x: x.split(';')[0] if isinstance(x, str) else x)
    
#     # Education mapping
#     ed_level_mapping = {
#         "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
#         "Some college/university study without earning a degree": 4,
#         "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
#         "Primary/elementary school": 2,
#         "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8,
#         "Associate degree (A.A., A.S., etc.)": 5,
#         "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
#         "Something else": 1
#     }
#     df['EdLevel'] = df['EdLevel'].map(ed_level_mapping).astype(float)
    
#     return df

# df = load_data()

# # Sidebar for filters
# st.sidebar.header("Filter Data")
# country_filter = st.sidebar.multiselect("Select Countries", options=df['Country'].unique(), default=df['Country'].unique())
# industry_filter = st.sidebar.multiselect("Select Industries", options=df['Industry'].unique(), default=df['Industry'].unique())
# min_salary = st.sidebar.slider("Minimum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=0)
# max_salary = st.sidebar.slider("Maximum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=int(df['ConvertedCompYearly'].max()))

# # Filter data
# filtered_df = df[
#     (df['Country'].isin(country_filter)) &
#     (df['Industry'].isin(industry_filter)) &
#     (df['ConvertedCompYearly'] >= min_salary) &
#     (df['ConvertedCompYearly'] <= max_salary)
# ].copy()

# # Summary Statistics
# st.header("📈 Summary Statistics")
# st.write(filtered_df.describe())

# # Distribution Plots
# st.header("📊 Distributions")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Salary Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='ConvertedCompYearly', bins=30, ax=ax)
#     ax.set_title("Distribution of Annual Salary")
#     ax.set_xlabel("Salary ($)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Age Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='Age', bins=20, ax=ax)
#     ax.set_title("Distribution of Age")
#     ax.set_xlabel("Age (Midpoint)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Categorical Bar Charts
# st.header("🏷️ Categorical Analysis")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Top 10 Countries by Count")
#     top_countries = filtered_df['Country'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_countries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Countries")
#     ax.set_xlabel("Country")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Top 10 Industries by Count")
#     top_industries = filtered_df['Industry'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_industries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Industries")
#     ax.set_xlabel("Industry")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Correlation Heatmap
# st.header("🔍 Correlation Heatmap")
# numeric_df = filtered_df[['Age', 'EdLevel', 'YearsCodePro', 'WorkExp', 'ConvertedCompYearly']].dropna()
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
# ax.set_title("Correlation Between Numeric Features")
# st.pyplot(fig)

# # Interactive Data Table
# st.header("📋 Raw Data Preview")
# st.write(filtered_df.head(10))  # Show first 10 rows of filtered data

# if st.button("Download Filtered Data as CSV"):
#     csv = filtered_df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">Download Filtered Data</a>'
#     st.markdown(href, unsafe_allow_html=True)

# def show_explore_page():
#     # This function is called by app.py
#     pass  # The rest of the code above is executed when this file is run

# if __name__ == "__main__":
#     show_explore_page()


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime

# # Set page configuration
# st.set_page_config(page_title="Explore Data", layout="wide")

# # Custom CSS for styling
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f5e6cc; /* Warm Beige, matching predict_page */
#         font-family: 'Arial', sans-serif;
#     }
#     .stButton>button {
#         background-color: #2c3e50;
#         color: white;
#         border-radius: 5px;
#         padding: 10px 20px;
#         font-size: 16px;
#     }
#     .stButton>button:hover {
#         background-color: #34495e;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Title and description
# st.title("📊 Explore Stack Overflow Developer Survey 2024")
# st.markdown("### Analyze trends and distributions in the developer survey data.")
# st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}", unsafe_allow_html=True)

# # Load and clean data
# @st.cache_data
# def load_data():
#     df = pd.read_csv("C:/Users/GOHIL RAJENDRASINH/Downloads/stack-overflow-developer-survey-2024/survey_results_public.csv")
#     necessary_columns = ['Age', 'EdLevel', 'YearsCodePro', 'Country', 'Industry', 'LanguageHaveWorkedWith', 
#                         'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith', 'WorkExp', 'ConvertedCompYearly']
#     df = df[necessary_columns].copy()
#     df = df.dropna(subset=necessary_columns)  # Drop rows with missing values in key columns
    
#     # Convert Age to numeric midpoints
#     age_mapping = {
#         "Under 18 years old": 14,
#         "18-24 years old": 21,
#         "25-34 years old": 29.5,
#         "35-44 years old": 39.5,
#         "45-54 years old": 49.5,
#         "55-64 years old": 59.5,
#         "65 years or older": 70
#     }
#     df['Age'] = df['Age'].map(age_mapping).astype(float)
    
#     # Convert YearsCodePro and WorkExp to numeric, handling non-numeric values
#     for col in ['YearsCodePro', 'WorkExp']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # Take first value for multi-value columns
#     for col in ['LanguageHaveWorkedWith', 'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith']:
#         df[col] = df[col].apply(lambda x: x.split(';')[0] if isinstance(x, str) else x)
    
#     # Education mapping
#     ed_level_mapping = {
#         "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
#         "Some college/university study without earning a degree": 4,
#         "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
#         "Primary/elementary school": 2,
#         "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8,
#         "Associate degree (A.A., A.S., etc.)": 5,
#         "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
#         "Something else": 1
#     }
#     df['EdLevel'] = df['EdLevel'].map(ed_level_mapping).astype(float)
    
#     return df

# df = load_data()

# # Sidebar for filters
# st.sidebar.header("Filter Data")
# country_filter = st.sidebar.multiselect("Select Countries", options=df['Country'].unique(), default=df['Country'].unique())
# industry_filter = st.sidebar.multiselect("Select Industries", options=df['Industry'].unique(), default=df['Industry'].unique())
# min_salary = st.sidebar.slider("Minimum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=0)
# max_salary = st.sidebar.slider("Maximum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=int(df['ConvertedCompYearly'].max()))

# # Filter data
# filtered_df = df[
#     (df['Country'].isin(country_filter)) &
#     (df['Industry'].isin(industry_filter)) &
#     (df['ConvertedCompYearly'] >= min_salary) &
#     (df['ConvertedCompYearly'] <= max_salary)
# ].copy()

# # Summary Statistics
# st.header("📈 Summary Statistics")
# st.write(filtered_df.describe())

# # Distribution Plots
# st.header("📊 Distributions")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Salary Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='ConvertedCompYearly', bins=30, ax=ax)
#     ax.set_title("Distribution of Annual Salary")
#     ax.set_xlabel("Salary ($)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Age Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='Age', bins=20, ax=ax)
#     ax.set_title("Distribution of Age")
#     ax.set_xlabel("Age (Midpoint)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Categorical Bar Charts
# st.header("🏷️ Categorical Analysis")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Top 10 Countries by Count")
#     top_countries = filtered_df['Country'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_countries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Countries")
#     ax.set_xlabel("Country")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Top 10 Industries by Count")
#     top_industries = filtered_df['Industry'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_industries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Industries")
#     ax.set_xlabel("Industry")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Correlation Heatmap
# st.header("🔍 Correlation Heatmap")
# numeric_df = filtered_df[['Age', 'EdLevel', 'YearsCodePro', 'WorkExp', 'ConvertedCompYearly']].dropna()
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
# ax.set_title("Correlation Between Numeric Features")
# st.pyplot(fig)

# # Interactive Data Table
# st.header("📋 Raw Data Preview")
# st.write(filtered_df.head(10))  # Show first 10 rows of filtered data


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import base64

# # Set page configuration
# st.set_page_config(page_title="Explore Data", layout="wide")

# # Custom CSS for styling
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f5e6cc; /* Warm Beige, matching predict_page */
#         font-family: 'Arial', sans-serif;
#     }
#     .stButton>button {
#         background-color: #2c3e50;
#         color: white;
#         border-radius: 5px;
#         padding: 10px 20px;
#         font-size: 16px;
#     }
#     .stButton>button:hover {
#         background-color: #34495e;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Title and description
# st.title("📊 Explore Stack Overflow Developer Survey 2024")
# st.markdown("### Analyze trends and distributions in the developer survey data.")
# st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}", unsafe_allow_html=True)

# # Load and clean data
# @st.cache_data
# def load_data():
#     df = pd.read_csv("C:/Users/GOHIL RAJENDRASINH/Downloads/stack-overflow-developer-survey-2024/survey_results_public.csv")
#     necessary_columns = ['Age', 'EdLevel', 'YearsCodePro', 'Country', 'Industry', 'LanguageHaveWorkedWith', 
#                         'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith', 'WorkExp', 'ConvertedCompYearly']
#     df = df[necessary_columns].copy()
#     df = df.dropna(subset=necessary_columns)  # Drop rows with missing values in key columns
    
#     # Convert Age to numeric midpoints
#     age_mapping = {
#         "Under 18 years old": 14,
#         "18-24 years old": 21,
#         "25-34 years old": 29.5,
#         "35-44 years old": 39.5,
#         "45-54 years old": 49.5,
#         "55-64 years old": 59.5,
#         "65 years or older": 70
#     }
#     df['Age'] = df['Age'].map(age_mapping).astype(float)
    
#     # Convert YearsCodePro and WorkExp to numeric, handling non-numeric values
#     for col in ['YearsCodePro', 'WorkExp']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # Take first value for multi-value columns
#     for col in ['LanguageHaveWorkedWith', 'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith']:
#         df[col] = df[col].apply(lambda x: x.split(';')[0] if isinstance(x, str) else x)
    
#     # Education mapping
#     ed_level_mapping = {
#         "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
#         "Some college/university study without earning a degree": 4,
#         "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
#         "Primary/elementary school": 2,
#         "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8,
#         "Associate degree (A.A., A.S., etc.)": 5,
#         "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
#         "Something else": 1
#     }
#     df['EdLevel'] = df['EdLevel'].map(ed_level_mapping).astype(float)
    
#     return df

# df = load_data()

# # Sidebar for filters
# st.sidebar.header("Filter Data")
# country_filter = st.sidebar.multiselect("Select Countries", options=df['Country'].unique(), default=df['Country'].unique())
# industry_filter = st.sidebar.multiselect("Select Industries", options=df['Industry'].unique(), default=df['Industry'].unique())
# min_salary = st.sidebar.slider("Minimum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=0)
# max_salary = st.sidebar.slider("Maximum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=int(df['ConvertedCompYearly'].max()))

# # Filter data
# filtered_df = df[
#     (df['Country'].isin(country_filter)) &
#     (df['Industry'].isin(industry_filter)) &
#     (df['ConvertedCompYearly'] >= min_salary) &
#     (df['ConvertedCompYearly'] <= max_salary)
# ].copy()

# # Summary Statistics
# st.header("📈 Summary Statistics")
# st.write(filtered_df.describe())

# # Distribution Plots
# st.header("📊 Distributions")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Salary Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='ConvertedCompYearly', bins=30, ax=ax)
#     ax.set_title("Distribution of Annual Salary")
#     ax.set_xlabel("Salary ($)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Age Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='Age', bins=20, ax=ax)
#     ax.set_title("Distribution of Age")
#     ax.set_xlabel("Age (Midpoint)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Categorical Bar Charts
# st.header("🏷️ Categorical Analysis")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Top 10 Countries by Count")
#     top_countries = filtered_df['Country'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_countries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Countries")
#     ax.set_xlabel("Country")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Top 10 Industries by Count")
#     top_industries = filtered_df['Industry'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_industries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Industries")
#     ax.set_xlabel("Industry")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Correlation Heatmap
# st.header("🔍 Correlation Heatmap")
# numeric_df = filtered_df[['Age', 'EdLevel', 'YearsCodePro', 'WorkExp', 'ConvertedCompYearly']].dropna()
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
# ax.set_title("Correlation Between Numeric Features")
# st.pyplot(fig)

# # Interactive Data Table
# st.header("📋 Raw Data Preview")
# st.write(filtered_df.head(10))  # Show first 10 rows of filtered data

# def show_explore_page():
#     pass  # This function is called by app.py, but the main code runs at module level


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime

# # Set page configuration
# st.set_page_config(page_title="Explore Data", layout="wide")

# # Custom CSS for styling
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f5e6cc; /* Warm Beige, matching predict_page */
#         font-family: 'Arial', sans-serif;
#     }
#     .stButton>button {
#         background-color: #2c3e50;
#         color: white;
#         border-radius: 5px;
#         padding: 10px 20px;
#         font-size: 16px;
#     }
#     .stButton>button:hover {
#         background-color: #34495e;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Load and clean data
# @st.cache_data
# def load_data():
#     df = pd.read_csv("C:/Users/GOHIL RAJENDRASINH/Downloads/stack-overflow-developer-survey-2024/survey_results_public.csv")
#     necessary_columns = ['Age', 'EdLevel', 'YearsCodePro', 'Country', 'Industry', 'LanguageHaveWorkedWith', 
#                         'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith', 'WorkExp', 'ConvertedCompYearly']
#     df = df[necessary_columns].copy()
#     df = df.dropna(subset=necessary_columns)
    
#     # Convert Age to numeric midpoints
#     age_mapping = {
#         "Under 18 years old": 14,
#         "18-24 years old": 21,
#         "25-34 years old": 29.5,
#         "35-44 years old": 39.5,
#         "45-54 years old": 49.5,
#         "55-64 years old": 59.5,
#         "65 years or older": 70
#     }
#     df['Age'] = df['Age'].map(age_mapping).astype(float)
    
#     # Ensure ConvertedCompYearly is numeric
#     df['ConvertedCompYearly'] = pd.to_numeric(df['ConvertedCompYearly'], errors='coerce')
#     df = df.dropna(subset=['ConvertedCompYearly'])
    
#     # Convert YearsCodePro and WorkExp to numeric
#     for col in ['YearsCodePro', 'WorkExp']:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # Take first value for multi-value columns
#     for col in ['LanguageHaveWorkedWith', 'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith']:
#         df[col] = df[col].apply(lambda x: x.split(';')[0] if isinstance(x, str) else x)
    
#     # Education mapping
#     ed_level_mapping = {
#         "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
#         "Some college/university study without earning a degree": 4,
#         "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
#         "Primary/elementary school": 2,
#         "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8,
#         "Associate degree (A.A., A.S., etc.)": 5,
#         "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
#         "Something else": 1
#     }
#     df['EdLevel'] = df['EdLevel'].map(ed_level_mapping).astype(float)
    
#     return df

# df = load_data()

# # Sidebar for filters
# st.sidebar.header("Filter Data")
# country_filter = st.sidebar.multiselect("Select Countries", options=df['Country'].unique(), default=[])
# industry_filter = st.sidebar.multiselect("Select Industries", options=df['Industry'].unique(), default=[])
# min_salary = st.sidebar.slider("Minimum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=0)
# max_salary = st.sidebar.slider("Maximum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=int(df['ConvertedCompYearly'].max()))

# # Filter data function
# def filter_data():
#     filtered_df = df[
#         (df['Country'].isin(country_filter) if country_filter else True) &
#         (df['Industry'].isin(industry_filter) if industry_filter else True) &
#         (df['ConvertedCompYearly'] >= min_salary) &
#         (df['ConvertedCompYearly'] <= max_salary)
#     ].copy()
#     return filtered_df

# filtered_df = filter_data()

# # Title and description
# st.title("📊 Explore Stack Overflow Developer Survey 2024")
# st.markdown("### Analyze trends and distributions in the developer survey data.")
# st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}", unsafe_allow_html=True)

# # Summary Statistics
# st.header("📈 Summary Statistics")
# st.write(filtered_df.describe())

# # Distribution Plots
# st.header("📊 Distributions")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Salary Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='ConvertedCompYearly', bins=30, ax=ax)
#     ax.set_title("Distribution of Annual Salary")
#     ax.set_xlabel("Salary ($)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Age Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(data=filtered_df, x='Age', bins=20, ax=ax)
#     ax.set_title("Distribution of Age")
#     ax.set_xlabel("Age (Midpoint)")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Categorical Bar Charts
# st.header("🏷️ Categorical Analysis")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Top 10 Countries by Count")
#     top_countries = filtered_df['Country'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_countries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Countries")
#     ax.set_xlabel("Country")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# with col2:
#     st.subheader("Top 10 Industries by Count")
#     top_industries = filtered_df['Industry'].value_counts().head(10)
#     fig, ax = plt.subplots()
#     top_industries.plot(kind='bar', ax=ax)
#     ax.set_title("Top 10 Industries")
#     ax.set_xlabel("Industry")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

# # Correlation Heatmap
# st.header("🔍 Correlation Heatmap")
# numeric_df = filtered_df[['Age', 'EdLevel', 'YearsCodePro', 'WorkExp', 'ConvertedCompYearly']].dropna()
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
# ax.set_title("Correlation Between Numeric Features")
# st.pyplot(fig)

# # Interactive Data Table
# st.header("📋 Raw Data Preview")
# st.write(filtered_df.head(10))  # Show first 10 rows of filtered data

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Explore Data", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5e6cc; /* Warm Beige, matching predict_page */
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #34495e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/GOHIL RAJENDRASINH/Downloads/stack-overflow-developer-survey-2024/survey_results_public.csv")
    necessary_columns = ['Age', 'EdLevel', 'YearsCodePro', 'Country', 'Industry', 'LanguageHaveWorkedWith', 
                        'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith', 'WorkExp', 'ConvertedCompYearly']
    df = df[necessary_columns].copy()
    df = df.dropna(subset=necessary_columns)
    
    # Convert Age to numeric midpoints
    age_mapping = {
        "Under 18 years old": 14,
        "18-24 years old": 21,
        "25-34 years old": 29.5,
        "35-44 years old": 39.5,
        "45-54 years old": 49.5,
        "55-64 years old": 59.5,
        "65 years or older": 70
    }
    df['Age'] = df['Age'].map(age_mapping).astype(float)
    
    # Ensure ConvertedCompYearly is numeric
    df['ConvertedCompYearly'] = pd.to_numeric(df['ConvertedCompYearly'], errors='coerce')
    df = df.dropna(subset=['ConvertedCompYearly'])
    
    # Convert YearsCodePro and WorkExp to numeric
    for col in ['YearsCodePro', 'WorkExp']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Take first value for multi-value columns
    for col in ['LanguageHaveWorkedWith', 'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith']:
        df[col] = df[col].apply(lambda x: x.split(';')[0] if isinstance(x, str) else x)
    
    # Education mapping
    ed_level_mapping = {
        "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
        "Some college/university study without earning a degree": 4,
        "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
        "Primary/elementary school": 2,
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8,
        "Associate degree (A.A., A.S., etc.)": 5,
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
        "Something else": 1
    }
    df['EdLevel'] = df['EdLevel'].map(ed_level_mapping).astype(float)
    
    return df

df = load_data()

# Sidebar for filters
st.sidebar.header("Filter Data")
country_filter = st.sidebar.multiselect("Select Countries", options=df['Country'].unique(), default=[])
industry_filter = st.sidebar.multiselect("Select Industries", options=df['Industry'].unique(), default=[])
min_salary = st.sidebar.slider("Minimum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=0)
max_salary = st.sidebar.slider("Maximum Salary ($)", min_value=0, max_value=int(df['ConvertedCompYearly'].max()), value=int(df['ConvertedCompYearly'].max()))

# Filter data function
def filter_data():
    filtered_df = df[
        (df['Country'].isin(country_filter) if country_filter else True) &
        (df['Industry'].isin(industry_filter) if industry_filter else True) &
        (df['ConvertedCompYearly'] >= min_salary) &
        (df['ConvertedCompYearly'] <= max_salary)
    ].copy()
    return filtered_df

filtered_df = filter_data()

# Title and description
st.title("📊 Explore Stack Overflow Developer Survey 2024")
st.markdown("### Analyze trends and distributions in the developer survey data.")
st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}", unsafe_allow_html=True)

# Summary Statistics
st.header("📈 Summary Statistics")
st.write(filtered_df.describe())

# Distribution Plots
st.header("📊 Distributions")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Salary Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data=filtered_df, x='ConvertedCompYearly', bins=30, ax=ax)
    ax.set_title("Distribution of Annual Salary")
    ax.set_xlabel("Salary ($)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data=filtered_df, x='Age', bins=20, ax=ax)
    ax.set_title("Distribution of Age")
    ax.set_xlabel("Age (Midpoint)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Categorical Bar Charts
st.header("🏷️ Categorical Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Countries by Count")
    top_countries = filtered_df['Country'].value_counts().head(10)
    fig, ax = plt.subplots()
    top_countries.plot(kind='bar', ax=ax)
    ax.set_title("Top 10 Countries")
    ax.set_xlabel("Country")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.subheader("Top 10 Industries by Count")
    top_industries = filtered_df['Industry'].value_counts().head(10)
    fig, ax = plt.subplots()
    top_industries.plot(kind='bar', ax=ax)
    ax.set_title("Top 10 Industries")
    ax.set_xlabel("Industry")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Correlation Heatmap
st.header("🔍 Correlation Heatmap")
numeric_df = filtered_df[['Age', 'EdLevel', 'YearsCodePro', 'WorkExp', 'ConvertedCompYearly']].dropna()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlation Between Numeric Features")
st.pyplot(fig)

# Interactive Data Table
st.header("📋 Raw Data Preview")
st.write(filtered_df.head(10))  # Show first 10 rows of filtered data

def show_explore_page():
    pass  # This function is called by app.py, but the main code runs at module level when imported


# import streamlit as st
# import pandas as pd

# def show_explore_page():
#     st.title("🔍 Explore Developer Salary Data")
    
#     # Load your data (replace with your actual data loading)
#     @st.cache_data
#     def load_data():
#         try:
#             return pd.read_csv("survey_results_public.csv")
#         except:
#             return pd.DataFrame()  # Return empty DataFrame if file not found
    
#     df = load_data()
    
#     if df.empty:
#         st.warning("No data available for exploration")
#         return
    
#     # Filters container
#     with st.container():
#         st.subheader("Filter Data")
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             country_filter = st.multiselect(
#                 "Filter by Country",
#                 options=df['Country'].unique(),
#                 default=[],
#                 key="country_filter"
#             )
            
#         with col2:
#             experience_filter = st.slider(
#                 "Years of Experience",
#                 min_value=int(df['YearsCodePro'].min()),
#                 max_value=int(df['YearsCodePro'].max()),
#                 value=(0, 20),
#                 key="exp_filter"
#             )
            
#         with col3:
#             salary_filter = st.slider(
#                 "Salary Range (USD)",
#                 min_value=int(df['ConvertedCompYearly'].min()),
#                 max_value=int(df['ConvertedCompYearly'].max()),
#                 value=(50000, 200000),
#                 key="salary_filter"
#             )
    
#     # Apply filters
#     filtered_df = df.copy()
#     if country_filter:
#         filtered_df = filtered_df[filtered_df['Country'].isin(country_filter)]
#     filtered_df = filtered_df[
#         (filtered_df['YearsCodePro'] >= experience_filter[0]) & 
#         (filtered_df['YearsCodePro'] <= experience_filter[1]) &
#         (filtered_df['ConvertedCompYearly'] >= salary_filter[0]) & 
#         (filtered_df['ConvertedCompYearly'] <= salary_filter[1])
#     ]
    
#     # Display results
#     st.subheader("Filtered Results")
#     if not filtered_df.empty:
#         st.dataframe(filtered_df, use_container_width=True)
        
#         # Show some statistics
#         st.subheader("Statistics")
#         col1, col2, col3 = st.columns(3)
#         col1.metric("Average Salary", f"${filtered_df['ConvertedCompYearly'].mean():,.0f}")
#         col2.metric("Median Salary", f"${filtered_df['ConvertedCompYearly'].median():,.0f}")
#         col3.metric("Respondents", len(filtered_df))
#     else:
#         st.warning("No results match your filters")