# import streamlit as st
# from predict_page import show_predict_page
# from explore_page import show_explore_page

# page=st.sidebar.selectbox("Explor or Predict",("Predict","Explore"))

# if page=="Predict":
#     show_predict_page()
# else:
#     show_explore_page()

# app.py
# import streamlit as st

# page = st.sidebar.selectbox("Choose a page", ["Predict", "Explore"])
# if page == "Predict":
#     import predict_page
#     predict_page.show_predict_page()
# elif page == "Explore":
#     import explore_page
#     explore_page.show_predict_page()  # Fix the function name here

# import streamlit as st

# page = st.sidebar.selectbox("Choose a page", ["Predict", "Explore"])
# if page == "Predict":
#     import predict_page
#     predict_page.show_predict_page()
# elif page == "Explore":
#     import explore_page
#     explore_page.show_explore_page()  # Corrected to match the function name


# import streamlit as st

# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Predict", "Explore"])

# import streamlit as st

# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Predict", "Explore"])
# No need to call functions manually; Streamlit handles pages in the pages/ directory

# import streamlit as st

# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Predict", "Explore"])
# # No need to call functions manually; Streamlit handles pages in the pages/ directory  

# import streamlit as st

# page = st.sidebar.selectbox("Choose a page", ["Predict", "Explore"])
# if page == "Predict":
#     import predict_page
#     predict_page.show_predict_page()
# elif page == "Explore":
#     import explore_page
#     explore_page.show_explore_page()  # Corrected function name


# import streamlit as st

# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Predict", "Explore"])
# Streamlit automatically loads pages from the pages/ directory


import streamlit as st

page = st.sidebar.selectbox("Choose a page", ["Predict", "Explore"])
if page == "Predict":
    import predict_page
    predict_page.show_predict_page()  # Call the function after import
elif page == "Explore":
    import explore_page
    explore_page.show_explore_page()  # Corrected function name


# import streamlit as st
# from predict_page import show_predict_page
# from explore_page import show_explore_page

# # Configure page settings
# st.set_page_config(
#     page_title="Developer Salary Predictor",
#     page_icon="💻",
#     layout="wide"
# )

# # Custom CSS to fix navigation and layout issues
# st.markdown("""
#     <style>
#         /* Fix navigation sticking to top */
#         .stApp {
#             margin-top: 0rem;
#             padding-top: 0rem;
#         }
        
#         /* Fix filter container sizing */
#         .st-emotion-cache-1v0mbdj {
#             width: 100% !important;
#         }
        
#         /* Fix blank page issues */
#         .block-container {
#             padding-top: 1rem;
#             padding-bottom: 1rem;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Navigation sidebar
# page = st.sidebar.selectbox("Navigate", ["Predict Salary", "Explore Data"])

# if page == "Predict Salary":
#     show_predict_page()
# else:
#     show_explore_page()