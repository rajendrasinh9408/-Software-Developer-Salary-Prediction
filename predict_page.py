# import streamlit as st
# import pickle
# import numpy as np
# from datetime import datetime

# @st.cache_resource
# def load_model_and_encoders():
#     try:
#         with open('rf_final_model.pkl', 'rb') as f:
#             data = pickle.load(f)
#         if isinstance(data, dict) and 'model' in data and all(k in data for k in ['country', 'industry', 'languagehaveworkedwith', 'platformhaveworkedwith', 'toolstechhaveworkedwith']):
#             st.write("✅ Model and encoders loaded successfully.")
#             return data['model'], {k: data[k] for k in ['country', 'industry', 'languagehaveworkedwith', 'platformhaveworkedwith', 'toolstechhaveworkedwith']}
#         else:
#             st.error("Model file is missing 'model' or required encoder keys.")
#             return None, None
#     except FileNotFoundError as e:
#         st.error(f"File not found: {e}")
#         return None, None
#     except Exception as e:
#         st.error(f"Error loading model/encoders: {e}")
#         return None, None

# def show_predict_page():
#     st.title("💼 Software Developer Salary Prediction")
#     st.write("### Provide your details to estimate your salary:")
#     st.write(f"Last updated: {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}")  # e.g., 06:35 PM IST, August 16, 2025

#     regressor, encoders = load_model_and_encoders()
#     if regressor is None or encoders is None:
#         st.stop()

#     # Debug: Print available classes to verify
#     st.write("Available options loaded:")
#     st.write({"Country": encoders['country'].classes_, 
#               "Industry": encoders['industry'].classes_, 
#               "Language": encoders['languagehaveworkedwith'].classes_, 
#               "Platform": encoders['platformhaveworkedwith'].classes_, 
#               "Tools": encoders['toolstechhaveworkedwith'].classes_})

#     # Get options from encoders
#     country_options = encoders['country'].classes_
#     industry_options = encoders['industry'].classes_
#     language_options = encoders['languagehaveworkedwith'].classes_
#     platform_options = encoders['platformhaveworkedwith'].classes_
#     tools_options = encoders['toolstechhaveworkedwith'].classes_

#     # Education map
#     edu_map = {
#         "Something else": 1,
#         "Primary/elementary school": 2,
#         "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
#         "Some college/university study without earning a degree": 4,
#         "Associate degree (A.A., A.S., etc.)": 5,
#         "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
#         "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
#         "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8
#     }
#     education_options = list(edu_map.keys())

#     # User Inputs
#     age = st.number_input("🎂 Age", min_value=14, max_value=70, value=29, help="Your current age (midpoint of range)")
#     education = st.selectbox("🎓 Education Level", education_options, help="Select your highest education level")
#     years_code_pro = st.number_input("👨‍💻 Years of Professional Coding", min_value=0, max_value=50, value=3, help="Years you've coded professionally")
#     country = st.selectbox("🌍 Country", country_options, help="Your country of residence")
#     industry = st.selectbox("🏢 Industry", industry_options, help="The industry you work in")
#     language = st.selectbox("💻 Programming Language", language_options, help="Primary language you work with")
#     platform = st.selectbox("🖥️ Platform", platform_options, help="Primary platform you work with")
#     tools = st.selectbox("🛠️ Tools & Technologies", tools_options, help="Primary tools/tech you work with")
#     work_exp = st.number_input("📊 Total Work Experience (years)", min_value=0, max_value=50, value=3, help="Total years of work experience")

#     if st.button("Calculate Salary"):
#         try:
#             # Encode inputs
#             edu_enc = edu_map[education]
#             country_enc = encoders['country'].transform([country])[0]
#             industry_enc = encoders['industry'].transform([industry])[0]
#             lang_enc = encoders['languagehaveworkedwith'].transform([language])[0]
#             platform_enc = encoders['platformhaveworkedwith'].transform([platform])[0]
#             tools_enc = encoders['toolstechhaveworkedwith'].transform([tools])[0]

#             # Feature vector (match training order)
#             X = np.array([[age, edu_enc, years_code_pro, country_enc, industry_enc, lang_enc, platform_enc, tools_enc, work_exp]])

#             # Predict
#             salary = regressor.predict(X.astype(float))[0]
#             st.success(f"💰 Estimated Annual Salary: ${salary:,.2f}")
#         except ValueError as ve:
#             st.error(f"Input error: {ve} (Check if selected values match training data)")
#         except Exception as e:
#             st.error(f"Prediction error: {e}")

# if __name__ == "__main__":
#     show_predict_page()



# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# from datetime import datetime
# import base64

# # Page configuration
# st.set_page_config(page_title="Salary Prediction", layout="wide")

# # Custom CSS
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f5e6cc; /* Warm Beige */
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
#     .stSuccess {
#         background-color: #e8f5e9;
#         padding: 10px;
#         border-radius: 5px;
#         color: #2e7d32;
#         font-weight: bold;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# @st.cache_resource
# def load_model_and_encoders():
#     try:
#         with open('rf_final_model.pkl', 'rb') as f:
#             data = pickle.load(f)
#         if isinstance(data, dict) and 'model' in data and all(k in data for k in ['country', 'industry', 'languagehaveworkedwith', 'platformhaveworkedwith', 'toolstechhaveworkedwith']):
#             st.write("✅ Model and encoders loaded successfully.")
#             return data['model'], {k: data[k] for k in ['country', 'industry', 'languagehaveworkedwith', 'platformhaveworkedwith', 'toolstechhaveworkedwith']}
#         else:
#             st.error("Model file is missing 'model' or required encoder keys.")
#             return None, None
#     except FileNotFoundError as e:
#         st.error(f"File not found: {e}")
#         return None, None
#     except Exception as e:
#         st.error(f"Error loading model/encoders: {e}")
#         return None, None

# def show_predict_page():
#     st.title("💼 Software Developer Salary Prediction")
#     st.markdown("### Provide your details to estimate your salary:")
#     st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}", unsafe_allow_html=True)

#     regressor, encoders = load_model_and_encoders()
#     if regressor is None or encoders is None:
#         st.stop()

#     with st.expander("View Available Options"):
#         st.write("Available options loaded:")
#         st.write({"Country": encoders['country'].classes_, 
#                   "Industry": encoders['industry'].classes_, 
#                   "Language": encoders['languagehaveworkedwith'].classes_, 
#                   "Platform": encoders['platformhaveworkedwith'].classes_, 
#                   "Tools": encoders['toolstechhaveworkedwith'].classes_})

#     country_options = encoders['country'].classes_
#     industry_options = encoders['industry'].classes_
#     language_options = encoders['languagehaveworkedwith'].classes_
#     platform_options = encoders['platformhaveworkedwith'].classes_
#     tools_options = encoders['toolstechhaveworkedwith'].classes_

#     edu_map = {
#         "Something else": 1,
#         "Primary/elementary school": 2,
#         "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
#         "Some college/university study without earning a degree": 4,
#         "Associate degree (A.A., A.S., etc.)": 5,
#         "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
#         "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
#         "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8
#     }
#     education_options = list(edu_map.keys())

#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.number_input("🎂 Age", min_value=14, max_value=70, value=29, help="Your current age (midpoint of range)")
#         education = st.selectbox("🎓 Education Level", education_options, help="Select your highest education level")
#         years_code_pro = st.number_input("👨‍💻 Years of Professional Coding", min_value=0, max_value=50, value=3, help="Years you've coded professionally")
#         country = st.selectbox("🌍 Country", country_options, help="Your country of residence")

#     with col2:
#         industry = st.selectbox("🏢 Industry", industry_options, help="The industry you work in")
#         language = st.selectbox("💻 Programming Language", language_options, help="Primary language you work with")
#         platform = st.selectbox("🖥️ Platform", platform_options, help="Primary platform you work with")
#         tools = st.selectbox("🛠️ Tools & Technologies", tools_options, help="Primary tools/tech you work with")
#         work_exp = st.number_input("📊 Total Work Experience (years)", min_value=0, max_value=50, value=3, help="Total years of work experience")

#     if st.button("Calculate Salary", key="predict_button", help="Click to predict your salary"):
#         try:
#             edu_enc = edu_map[education]
#             country_enc = encoders['country'].transform([country])[0]
#             industry_enc = encoders['industry'].transform([industry])[0]
#             lang_enc = encoders['languagehaveworkedwith'].transform([language])[0]
#             platform_enc = encoders['platformhaveworkedwith'].transform([platform])[0]
#             tools_enc = encoders['toolstechhaveworkedwith'].transform([tools])[0]

#             X = np.array([[age, edu_enc, years_code_pro, country_enc, industry_enc, lang_enc, platform_enc, tools_enc, work_exp]])

#             salary = regressor.predict(X.astype(float))[0]
#             st.success(f"💰 **Estimated Annual Salary: ${salary:,.2f}**", icon="💸")

#             prediction_data = pd.DataFrame({
#                 "Age": [age],
#                 "Education_Level": [education],
#                 "Years_Coding_Pro": [years_code_pro],
#                 "Country": [country],
#                 "Industry": [industry],
#                 "Language": [language],
#                 "Platform": [platform],
#                 "Tools": [tools],
#                 "Work_Experience": [work_exp],
#                 "Predicted_Salary": [salary]
#             })
#             csv = prediction_data.to_csv(index=False)
#             b64 = base64.b64encode(csv.encode()).decode()
#             href = f'<a href="data:file/csv;base64,{b64}" download="salary_prediction.csv">Download Prediction as CSV</a>'
#             st.markdown(href, unsafe_allow_html=True)

#             if st.button("Show Salary vs. Age Chart"):
#                 ages = np.array([14, 21, 29.5, 39.5, 49.5, 59.5, 70])
#                 salaries = [regressor.predict(np.array([[a, edu_enc, years_code_pro, country_enc, industry_enc, lang_enc, platform_enc, tools_enc, work_exp]]).astype(float))[0] for a in ages]
#                 chart_data = pd.DataFrame({"Age": ages, "Predicted Salary ($)": salaries})
#                 st.line_chart(chart_data.set_index("Age"))

#         except ValueError as ve:
#             st.error(f"Input error: {ve} (Check if selected values match training data)")
#         except Exception as e:
#             st.error(f"Prediction error: {e}")

# if __name__ == "__main__":
#     show_predict_page()

# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# from datetime import datetime
# import base64

# # Page configuration
# st.set_page_config(page_title="Salary Prediction", layout="wide")

# # Custom CSS
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f5e6cc; /* Warm Beige */
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
#     .stSuccess {
#         background-color: #e8f5e9;
#         padding: 10px;
#         border-radius: 5px;
#         color: #2e7d32;
#         font-weight: bold;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# @st.cache_resource
# def load_model_and_encoders():
#     try:
#         with open('rf_final_model.pkl', 'rb') as f:
#             data = pickle.load(f)
#         if isinstance(data, dict) and 'model' in data and all(k in data for k in ['country', 'industry', 'languagehaveworkedwith', 'platformhaveworkedwith', 'toolstechhaveworkedwith']):
#             st.write("✅ Model and encoders loaded successfully.")
#             return data['model'], {k: data[k] for k in ['country', 'industry', 'languagehaveworkedwith', 'platformhaveworkedwith', 'toolstechhaveworkedwith']}
#         else:
#             st.error("Model file is missing 'model' or required encoder keys.")
#             return None, None
#     except FileNotFoundError as e:
#         st.error(f"File not found: {e}")
#         return None, None
#     except Exception as e:
#         st.error(f"Error loading model/encoders: {e}")
#         return None, None

# def show_predict_page():
#     st.title("💼 Software Developer Salary Prediction")
#     st.markdown("### Provide your details to estimate your salary:")
#     st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}", unsafe_allow_html=True)

#     regressor, encoders = load_model_and_encoders()
#     if regressor is None or encoders is None:
#         st.stop()

#     with st.expander("View Available Options"):
#         st.write("Available options loaded:")
#         st.write({"Country": encoders['country'].classes_, 
#                   "Industry": encoders['industry'].classes_, 
#                   "Language": encoders['languagehaveworkedwith'].classes_, 
#                   "Platform": encoders['platformhaveworkedwith'].classes_, 
#                   "Tools": encoders['toolstechhaveworkedwith'].classes_})

#     country_options = encoders['country'].classes_
#     industry_options = encoders['industry'].classes_
#     language_options = encoders['languagehaveworkedwith'].classes_
#     platform_options = encoders['platformhaveworkedwith'].classes_
#     tools_options = encoders['toolstechhaveworkedwith'].classes_

#     edu_map = {
#         "Something else": 1,
#         "Primary/elementary school": 2,
#         "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
#         "Some college/university study without earning a degree": 4,
#         "Associate degree (A.A., A.S., etc.)": 5,
#         "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
#         "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
#         "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8
#     }
#     education_options = list(edu_map.keys())

#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.number_input("🎂 Age", min_value=14, max_value=70, value=29, help="Your current age (midpoint of range)")
#         education = st.selectbox("🎓 Education Level", education_options, help="Select your highest education level")
#         years_code_pro = st.number_input("👨‍💻 Years of Professional Coding", min_value=0, max_value=50, value=3, help="Years you've coded professionally")
#         country = st.selectbox("🌍 Country", country_options, help="Your country of residence")

#     with col2:
#         industry = st.selectbox("🏢 Industry", industry_options, help="The industry you work in")
#         language = st.selectbox("💻 Programming Language", language_options, help="Primary language you work with")
#         platform = st.selectbox("🖥️ Platform", platform_options, help="Primary platform you work with")
#         tools = st.selectbox("🛠️ Tools & Technologies", tools_options, help="Primary tools/tech you work with")
#         work_exp = st.number_input("📊 Total Work Experience (years)", min_value=0, max_value=50, value=3, help="Total years of work experience")

#     if st.button("Calculate Salary", key="predict_button", help="Click to predict your salary"):
#         try:
#             edu_enc = edu_map[education]
#             country_enc = encoders['country'].transform([country])[0]
#             industry_enc = encoders['industry'].transform([industry])[0]
#             lang_enc = encoders['languagehaveworkedwith'].transform([language])[0]
#             platform_enc = encoders['platformhaveworkedwith'].transform([platform])[0]
#             tools_enc = encoders['toolstechhaveworkedwith'].transform([tools])[0]

#             X = np.array([[age, edu_enc, years_code_pro, country_enc, industry_enc, lang_enc, platform_enc, tools_enc, work_exp]])

#             salary = regressor.predict(X.astype(float))[0]
#             st.success(f"💰 **Estimated Annual Salary: ${salary:,.2f}**", icon="💸")

#             prediction_data = pd.DataFrame({
#                 "Age": [age],
#                 "Education_Level": [education],
#                 "Years_Coding_Pro": [years_code_pro],
#                 "Country": [country],
#                 "Industry": [industry],
#                 "Language": [language],
#                 "Platform": [platform],
#                 "Tools": [tools],
#                 "Work_Experience": [work_exp],
#                 "Predicted_Salary": [salary]
#             })
#             csv = prediction_data.to_csv(index=False)
#             b64 = base64.b64encode(csv.encode()).decode()
#             href = f'<a href="data:file/csv;base64,{b64}" download="salary_prediction.csv">Download Prediction as CSV</a>'
#             st.markdown(href, unsafe_allow_html=True)

#             if st.button("Show Salary vs. Age Chart"):
#                 ages = np.array([14, 21, 29.5, 39.5, 49.5, 59.5, 70])
#                 salaries = [regressor.predict(np.array([[a, edu_enc, years_code_pro, country_enc, industry_enc, lang_enc, platform_enc, tools_enc, work_exp]]).astype(float))[0] for a in ages]
#                 chart_data = pd.DataFrame({"Age": ages, "Predicted Salary ($)": salaries})
#                 st.line_chart(chart_data.set_index("Age"))

#         except ValueError as ve:
#             st.error(f"Input error: {ve} (Check if selected values match training data)")
#         except Exception as e:
#             st.error(f"Prediction error: {e}")

# if __name__ == "__main__":
#     show_predict_page()

                                    ## right one
# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# from datetime import datetime
# import base64

# # Page configuration
# st.set_page_config(page_title="Salary Prediction", layout="wide")

# # Custom CSS
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f5e6cc; /* Warm Beige */
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
#     .stSuccess {
#         background-color: #e8f5e9;
#         padding: 10px;
#         border-radius: 5px;
#         color: #2e7d32;
#         font-weight: bold;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# @st.cache_resource
# def load_model_and_encoders():
#     try:
#         with open('rf_final_model.pkl', 'rb') as f:
#             data = pickle.load(f)
#         if isinstance(data, dict) and 'model' in data and all(k in data for k in ['country', 'industry', 'languagehaveworkedwith', 'platformhaveworkedwith', 'toolstechhaveworkedwith']):
#             st.write("✅ Model and encoders loaded successfully.")
#             return data['model'], {k: data[k] for k in ['country', 'industry', 'languagehaveworkedwith', 'platformhaveworkedwith', 'toolstechhaveworkedwith']}
#         else:
#             st.error("Model file is missing 'model' or required encoder keys.")
#             return None, None
#     except FileNotFoundError as e:
#         st.error(f"File not found: {e}")
#         return None, None
#     except Exception as e:
#         st.error(f"Error loading model/encoders: {e}")
#         return None, None

# def show_predict_page():
#     st.title("💼 Software Developer Salary Prediction")
#     st.markdown("### Provide your details to estimate your salary:")
#     st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}", unsafe_allow_html=True)

#     regressor, encoders = load_model_and_encoders()
#     if regressor is None or encoders is None:
#         st.stop()

#     with st.expander("View Available Options"):
#         st.write("Available options loaded:")
#         st.write({"Country": encoders['country'].classes_, 
#                   "Industry": encoders['industry'].classes_, 
#                   "Language": encoders['languagehaveworkedwith'].classes_, 
#                   "Platform": encoders['platformhaveworkedwith'].classes_, 
#                   "Tools": encoders['toolstechhaveworkedwith'].classes_})

#     country_options = encoders['country'].classes_
#     industry_options = encoders['industry'].classes_
#     language_options = encoders['languagehaveworkedwith'].classes_
#     platform_options = encoders['platformhaveworkedwith'].classes_
#     tools_options = encoders['toolstechhaveworkedwith'].classes_

#     edu_map = {
#         "Something else": 1,
#         "Primary/elementary school": 2,
#         "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
#         "Some college/university study without earning a degree": 4,
#         "Associate degree (A.A., A.S., etc.)": 5,
#         "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
#         "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
#         "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8
#     }
#     education_options = list(edu_map.keys())

#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.number_input("🎂 Age", min_value=14, max_value=70, value=29, help="Your current age (midpoint of range)")
#         education = st.selectbox("🎓 Education Level", education_options, help="Select your highest education level")
#         years_code_pro = st.number_input("👨‍💻 Years of Professional Coding", min_value=0, max_value=50, value=3, help="Years you've coded professionally")
#         country = st.selectbox("🌍 Country", country_options, help="Your country of residence")

#     with col2:
#         industry = st.selectbox("🏢 Industry", industry_options, help="The industry you work in")
#         language = st.selectbox("💻 Programming Language", language_options, help="Primary language you work with")
#         platform = st.selectbox("🖥️ Platform", platform_options, help="Primary platform you work with")
#         tools = st.selectbox("🛠️ Tools & Technologies", tools_options, help="Primary tools/tech you work with")
#         work_exp = st.number_input("📊 Total Work Experience (years)", min_value=0, max_value=50, value=3, help="Total years of work experience")

#     if st.button("Calculate Salary", key="predict_button", help="Click to predict your salary"):
#         try:
#             edu_enc = edu_map[education]
#             country_enc = encoders['country'].transform([country])[0]
#             industry_enc = encoders['industry'].transform([industry])[0]
#             lang_enc = encoders['languagehaveworkedwith'].transform([language])[0]
#             platform_enc = encoders['platformhaveworkedwith'].transform([platform])[0]
#             tools_enc = encoders['toolstechhaveworkedwith'].transform([tools])[0]

#             X = np.array([[age, edu_enc, years_code_pro, country_enc, industry_enc, lang_enc, platform_enc, tools_enc, work_exp]])

#             salary = regressor.predict(X.astype(float))[0]
#             st.success(f"💰 **Estimated Annual Salary: ${salary:,.2f}**", icon="💸")

#             prediction_data = pd.DataFrame({
#                 "Age": [age],
#                 "Education_Level": [education],
#                 "Years_Coding_Pro": [years_code_pro],
#                 "Country": [country],
#                 "Industry": [industry],
#                 "Language": [language],
#                 "Platform": [platform],
#                 "Tools": [tools],
#                 "Work_Experience": [work_exp],
#                 "Predicted_Salary": [salary]
#             })
#             csv = prediction_data.to_csv(index=False)
#             b64 = base64.b64encode(csv.encode()).decode()
#             href = f'<a href="data:file/csv;base64,{b64}" download="salary_prediction.csv">Download Prediction as CSV</a>'
#             st.markdown(href, unsafe_allow_html=True)

#             if st.button("Show Salary vs. Age Chart"):
#                 ages = np.array([14, 21, 29.5, 39.5, 49.5, 59.5, 70])
#                 salaries = [regressor.predict(np.array([[a, edu_enc, years_code_pro, country_enc, industry_enc, lang_enc, platform_enc, tools_enc, work_exp]]).astype(float))[0] for a in ages]
#                 chart_data = pd.DataFrame({"Age": ages, "Predicted Salary ($)": salaries})
#                 st.line_chart(chart_data.set_index("Age"))

#         except ValueError as ve:
#             st.error(f"Input error: {ve} (Check if selected values match training data)")
#         except Exception as e:
#             st.error(f"Prediction error: {e}")

# if __name__ == "__main__":
#     show_predict_page()




# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# from datetime import datetime
# import base64

# # Page configuration
# st.set_page_config(page_title="Salary Prediction", layout="wide")

# # Custom CSS
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f5e6cc; /* Warm Beige */
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
#     .stSuccess {
#         background-color: #e8f5e9;
#         padding: 10px;
#         border-radius: 5px;
#         color: #2e7d32;
#         font-weight: bold;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# @st.cache_resource
# def load_model_and_encoders():
#     try:
#         with open('rf_final_model.pkl', 'rb') as f:
#             data = pickle.load(f)
#         if isinstance(data, dict) and 'model' in data and all(k in data for k in ['country', 'industry', 'languagehaveworkedwith', 'platformhaveworkedwith', 'toolstechhaveworkedwith']):
#             st.write("✅ Model and encoders loaded successfully.")
#             return data['model'], {k: data[k] for k in ['country', 'industry', 'languagehaveworkedwith', 'platformhaveworkedwith', 'toolstechhaveworkedwith']}
#         else:
#             st.error("Model file is missing 'model' or required encoder keys.")
#             return None, None
#     except FileNotFoundError as e:
#         st.error(f"File not found: {e}")
#         return None, None
#     except Exception as e:
#         st.error(f"Error loading model/encoders: {e}")
#         return None, None

# st.title("💼 Software Developer Salary Prediction")
# st.markdown("### Provide your details to estimate your salary:")
# st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}", unsafe_allow_html=True)

# regressor, encoders = load_model_and_encoders()
# if regressor is None or encoders is None:
#     st.stop()

# with st.expander("View Available Options"):
#     st.write("Available options loaded:")
#     st.write({"Country": encoders['country'].classes_, 
#               "Industry": encoders['industry'].classes_, 
#               "Language": encoders['languagehaveworkedwith'].classes_, 
#               "Platform": encoders['platformhaveworkedwith'].classes_, 
#               "Tools": encoders['toolstechhaveworkedwith'].classes_})

# country_options = encoders['country'].classes_
# industry_options = encoders['industry'].classes_
# language_options = encoders['languagehaveworkedwith'].classes_
# platform_options = encoders['platformhaveworkedwith'].classes_
# tools_options = encoders['toolstechhaveworkedwith'].classes_

# edu_map = {
#     "Something else": 1,
#     "Primary/elementary school": 2,
#     "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
#     "Some college/university study without earning a degree": 4,
#     "Associate degree (A.A., A.S., etc.)": 5,
#     "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
#     "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
#     "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8
# }
# education_options = list(edu_map.keys())

# col1, col2 = st.columns(2)

# with col1:
#     age = st.number_input("🎂 Age", min_value=14, max_value=70, value=29, help="Your current age (midpoint of range)")
#     education = st.selectbox("🎓 Education Level", education_options, help="Select your highest education level")
#     years_code_pro = st.number_input("👨‍💻 Years of Professional Coding", min_value=0, max_value=50, value=3, help="Years you've coded professionally")
#     country = st.selectbox("🌍 Country", country_options, help="Your country of residence")

# with col2:
#     industry = st.selectbox("🏢 Industry", industry_options, help="The industry you work in")
#     language = st.selectbox("💻 Programming Language", language_options, help="Primary language you work with")
#     platform = st.selectbox("🖥️ Platform", platform_options, help="Primary platform you work with")
#     tools = st.selectbox("🛠️ Tools & Technologies", tools_options, help="Primary tools/tech you work with")
#     work_exp = st.number_input("📊 Total Work Experience (years)", min_value=0, max_value=50, value=3, help="Total years of work experience")

# if st.button("Calculate Salary", key="predict_button", help="Click to predict your salary"):
#     try:
#         edu_enc = edu_map[education]
#         country_enc = encoders['country'].transform([country])[0]
#         industry_enc = encoders['industry'].transform([industry])[0]
#         lang_enc = encoders['languagehaveworkedwith'].transform([language])[0]
#         platform_enc = encoders['platformhaveworkedwith'].transform([platform])[0]
#         tools_enc = encoders['toolstechhaveworkedwith'].transform([tools])[0]

#         X = np.array([[age, edu_enc, years_code_pro, country_enc, industry_enc, lang_enc, platform_enc, tools_enc, work_exp]])

#         salary = regressor.predict(X.astype(float))[0]
#         st.success(f"💰 **Estimated Annual Salary: ${salary:,.2f}**", icon="💸")

#         prediction_data = pd.DataFrame({
#             "Age": [age],
#             "Education_Level": [education],
#             "Years_Coding_Pro": [years_code_pro],
#             "Country": [country],
#             "Industry": [industry],
#             "Language": [language],
#             "Platform": [platform],
#             "Tools": [tools],
#             "Work_Experience": [work_exp],
#             "Predicted_Salary": [salary]
#         })
#         csv = prediction_data.to_csv(index=False)
#         b64 = base64.b64encode(csv.encode()).decode()
#         href = f'<a href="data:file/csv;base64,{b64}" download="salary_prediction.csv">Download Prediction as CSV</a>'
#         st.markdown(href, unsafe_allow_html=True)

#         if st.button("Show Salary vs. Age Chart"):
#             ages = np.array([14, 21, 29.5, 39.5, 49.5, 59.5, 70])
#             salaries = [regressor.predict(np.array([[a, edu_enc, years_code_pro, country_enc, industry_enc, lang_enc, platform_enc, tools_enc, work_exp]]).astype(float))[0] for a in ages]
#             chart_data = pd.DataFrame({"Age": ages, "Predicted Salary ($)": salaries})
#             st.line_chart(chart_data.set_index("Age"))

#     except ValueError as ve:
#         st.error(f"Input error: {ve} (Check if selected values match training data)")
#     except Exception as e:
#         st.error(f"Prediction error: {e}")

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import base64

# Page configuration
st.set_page_config(page_title="Salary Prediction", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5e6cc; /* Warm Beige */
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
    .stSuccess {
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 5px;
        color: #2e7d32;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model_and_encoders():
    try:
        with open('rf_final_model.pkl', 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict) and 'model' in data and all(k in data for k in ['country', 'industry', 'languagehaveworkedwith', 'platformhaveworkedwith', 'toolstechhaveworkedwith']):
            st.write("✅ Model and encoders loaded successfully.")
            return data['model'], {k: data[k] for k in ['country', 'industry', 'languagehaveworkedwith', 'platformhaveworkedwith', 'toolstechhaveworkedwith']}
        else:
            st.error("Model file is missing 'model' or required encoder keys.")
            return None, None
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model/encoders: {e}")
        return None, None

def show_predict_page():
    st.title("💼 Software Developer Salary Prediction")
    st.markdown("### Provide your details to estimate your salary:")
    st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}", unsafe_allow_html=True)

    regressor, encoders = load_model_and_encoders()
    if regressor is None or encoders is None:
        st.stop()

    with st.expander("View Available Options"):
        st.write("Available options loaded:")
        st.write({"Country": encoders['country'].classes_, 
                  "Industry": encoders['industry'].classes_, 
                  "Language": encoders['languagehaveworkedwith'].classes_, 
                  "Platform": encoders['platformhaveworkedwith'].classes_, 
                  "Tools": encoders['toolstechhaveworkedwith'].classes_})

    country_options = encoders['country'].classes_
    industry_options = encoders['industry'].classes_
    language_options = encoders['languagehaveworkedwith'].classes_
    platform_options = encoders['platformhaveworkedwith'].classes_
    tools_options = encoders['toolstechhaveworkedwith'].classes_

    edu_map = {
        "Something else": 1,
        "Primary/elementary school": 2,
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
        "Some college/university study without earning a degree": 4,
        "Associate degree (A.A., A.S., etc.)": 5,
        "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
        "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8
    }
    education_options = list(edu_map.keys())

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🎂 Age", min_value=14, max_value=70, value=29, help="Your current age (midpoint of range)")
        education = st.selectbox("🎓 Education Level", education_options, help="Select your highest education level")
        years_code_pro = st.number_input("👨‍💻 Years of Professional Coding", min_value=0, max_value=50, value=3, help="Years you've coded professionally")
        country = st.selectbox("🌍 Country", country_options, help="Your country of residence")

    with col2:
        industry = st.selectbox("🏢 Industry", industry_options, help="The industry you work in")
        language = st.selectbox("💻 Programming Language", language_options, help="Primary language you work with")
        platform = st.selectbox("🖥️ Platform", platform_options, help="Primary platform you work with")
        tools = st.selectbox("🛠️ Tools & Technologies", tools_options, help="Primary tools/tech you work with")
        work_exp = st.number_input("📊 Total Work Experience (years)", min_value=0, max_value=50, value=3, help="Total years of work experience")

    if st.button("Calculate Salary", key="predict_button", help="Click to predict your salary"):
        try:
            edu_enc = edu_map[education]
            country_enc = encoders['country'].transform([country])[0]
            industry_enc = encoders['industry'].transform([industry])[0]
            lang_enc = encoders['languagehaveworkedwith'].transform([language])[0]
            platform_enc = encoders['platformhaveworkedwith'].transform([platform])[0]
            tools_enc = encoders['toolstechhaveworkedwith'].transform([tools])[0]

            X = np.array([[age, edu_enc, years_code_pro, country_enc, industry_enc, lang_enc, platform_enc, tools_enc, work_exp]])

            salary = regressor.predict(X.astype(float))[0]
            st.success(f"💰 **Estimated Annual Salary: ${salary:,.2f}**", icon="💸")

            prediction_data = pd.DataFrame({
                "Age": [age],
                "Education_Level": [education],
                "Years_Coding_Pro": [years_code_pro],
                "Country": [country],
                "Industry": [industry],
                "Language": [language],
                "Platform": [platform],
                "Tools": [tools],
                "Work_Experience": [work_exp],
                "Predicted_Salary": [salary]
            })
            csv = prediction_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="salary_prediction.csv">Download Prediction as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

            if st.button("Show Salary vs. Age Chart"):
                ages = np.array([14, 21, 29.5, 39.5, 49.5, 59.5, 70])
                salaries = [regressor.predict(np.array([[a, edu_enc, years_code_pro, country_enc, industry_enc, lang_enc, platform_enc, tools_enc, work_exp]]).astype(float))[0] for a in ages]
                chart_data = pd.DataFrame({"Age": ages, "Predicted Salary ($)": salaries})
                st.line_chart(chart_data.set_index("Age"))

        except ValueError as ve:
            st.error(f"Input error: {ve} (Check if selected values match training data)")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# No if __name__ == "__main__": block needed; the function is called by app.py

# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd
# from datetime import datetime
# import base64

# # Page configuration
# st.set_page_config(page_title="Salary Prediction", layout="wide")

# # Custom CSS
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f5e6cc; /* Warm Beige */
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
#     .stSuccess {
#         background-color: #e8f5e9;
#         padding: 10px;
#         border-radius: 5px;
#         color: #2e7d32;
#         font-weight: bold;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# @st.cache_resource
# def load_model_and_encoders():
#     try:
#         with open('rf_final_model.pkl', 'rb') as f:
#             data = pickle.load(f)
#         if isinstance(data, dict) and 'model' in data and all(k in data for k in ['country', 'industry', 'languagehaveworkedwith', 'platformhaveworkedwith', 'toolstechhaveworkedwith']):
#             st.write("✅ Model and encoders loaded successfully.")
#             return data['model'], {k: data[k] for k in ['country', 'industry', 'languagehaveworkedwith', 'platformhaveworkedwith', 'toolstechhaveworkedwith']}
#         else:
#             st.error("Model file is missing 'model' or required encoder keys.")
#             return None, None
#     except FileNotFoundError as e:
#         st.error(f"File not found: {e}")
#         return None, None
#     except Exception as e:
#         st.error(f"Error loading model/encoders: {e}")
#         return None, None

# st.title("💼 Software Developer Salary Prediction")
# st.markdown("### Provide your details to estimate your salary:")
# st.markdown(f"**Last updated:** {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}", unsafe_allow_html=True)

# regressor, encoders = load_model_and_encoders()
# if regressor is None or encoders is None:
#     st.stop()

# with st.expander("View Available Options"):
#     st.write("Available options loaded:")
#     st.write({"Country": encoders['country'].classes_, 
#               "Industry": encoders['industry'].classes_, 
#               "Language": encoders['languagehaveworkedwith'].classes_, 
#               "Platform": encoders['platformhaveworkedwith'].classes_, 
#               "Tools": encoders['toolstechhaveworkedwith'].classes_})

# country_options = encoders['country'].classes_
# industry_options = encoders['industry'].classes_
# language_options = encoders['languagehaveworkedwith'].classes_
# platform_options = encoders['platformhaveworkedwith'].classes_
# tools_options = encoders['toolstechhaveworkedwith'].classes_

# edu_map = {
#     "Something else": 1,
#     "Primary/elementary school": 2,
#     "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 3,
#     "Some college/university study without earning a degree": 4,
#     "Associate degree (A.A., A.S., etc.)": 5,
#     "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 6,
#     "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 7,
#     "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 8
# }
# education_options = list(edu_map.keys())

# col1, col2 = st.columns(2)

# with col1:
#     age = st.number_input("🎂 Age", min_value=14, max_value=70, value=29, help="Your current age (midpoint of range)")
#     education = st.selectbox("🎓 Education Level", education_options, help="Select your highest education level")
#     years_code_pro = st.number_input("👨‍💻 Years of Professional Coding", min_value=0, max_value=50, value=3, help="Years you've coded professionally")
#     country = st.selectbox("🌍 Country", country_options, help="Your country of residence")

# with col2:
#     industry = st.selectbox("🏢 Industry", industry_options, help="The industry you work in")
#     language = st.selectbox("💻 Programming Language", language_options, help="Primary language you work with")
#     platform = st.selectbox("🖥️ Platform", platform_options, help="Primary platform you work with")
#     tools = st.selectbox("🛠️ Tools & Technologies", tools_options, help="Primary tools/tech you work with")
#     work_exp = st.number_input("📊 Total Work Experience (years)", min_value=0, max_value=50, value=3, help="Total years of work experience")

# if st.button("Calculate Salary", key="predict_button", help="Click to predict your salary"):
#     try:
#         edu_enc = edu_map[education]
#         country_enc = encoders['country'].transform([country])[0]
#         industry_enc = encoders['industry'].transform([industry])[0]
#         lang_enc = encoders['languagehaveworkedwith'].transform([language])[0]
#         platform_enc = encoders['platformhaveworkedwith'].transform([platform])[0]
#         tools_enc = encoders['toolstechhaveworkedwith'].transform([tools])[0]

#         X = np.array([[age, edu_enc, years_code_pro, country_enc, industry_enc, lang_enc, platform_enc, tools_enc, work_exp]])

#         salary = regressor.predict(X.astype(float))[0]
#         st.success(f"💰 **Estimated Annual Salary: ${salary:,.2f}**", icon="💸")

#         prediction_data = pd.DataFrame({
#             "Age": [age],
#             "Education_Level": [education],
#             "Years_Coding_Pro": [years_code_pro],
#             "Country": [country],
#             "Industry": [industry],
#             "Language": [language],
#             "Platform": [platform],
#             "Tools": [tools],
#             "Work_Experience": [work_exp],
#             "Predicted_Salary": [salary]
#         })
#         csv = prediction_data.to_csv(index=False)
#         b64 = base64.b64encode(csv.encode()).decode()
#         href = f'<a href="data:file/csv;base64,{b64}" download="salary_prediction.csv">Download Prediction as CSV</a>'
#         st.markdown(href, unsafe_allow_html=True)

#         if st.button("Show Salary vs. Age Chart"):
#             ages = np.array([14, 21, 29.5, 39.5, 49.5, 59.5, 70])
#             salaries = [regressor.predict(np.array([[a, edu_enc, years_code_pro, country_enc, industry_enc, lang_enc, platform_enc, tools_enc, work_exp]]).astype(float))[0] for a in ages]
#             chart_data = pd.DataFrame({"Age": ages, "Predicted Salary ($)": salaries})
#             st.line_chart(chart_data.set_index("Age"))

#     except ValueError as ve:
#         st.error(f"Input error: {ve} (Check if selected values match training data)")
#     except Exception as e:
#         st.error(f"Prediction error: {e}")

# def show_predict_page():
#     pass  # This function is called by app.py, but the main code runs at module level when imported

