import pandas as pd
import pickle
import streamlit as st
import streamlit.components.v1 as components
import model_methods

# Set Page Config Options
st.set_page_config(page_title='Logistic Regression Assignment',
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items=None)
st.sidebar.markdown("[Training File on Colab](https://colab.research.google.com/drive/1mnR8oaHDPMKl34-Qy2W0Q_q1r3ypSAXu#scrollTo=VDiobCeJoP6U)", unsafe_allow_html=True)
st.sidebar.markdown("[Github Project Link](https://github.com/thanecoder/linear_regression_assignment)", unsafe_allow_html=True)

st.title('Logistic Regression Assignment')
assignment_question = """Logistic Regression Assignment
Task: Deploy this assignment in any cloud platform.(Try to look for free cloud platform)

Assignment: Submit assignment’s deployable link only.
"""
st.subheader(assignment_question)

profile_report_container = st.container()
profile_report_container.subheader("Pandas Profile Report")
# profile_report_container.write("Please wait while the the report is loading....")
with open('profile_report.sav', 'rb') as f:
    profile_report = pickle.load(f)
components.html(profile_report, height=600, width=800, scrolling=True)

# Form for Input Parameters
with st.sidebar:
    with st.form("my_form"):
        st.write("Input Parameters")
        CRIM = st.number_input('Insert CRIM', min_value=0.0, max_value=100.0, step=0.5)
        ZN = st.number_input('Insert ZN', min_value=0.0, max_value=100.0, step=0.5)
        INDUS = st.number_input('Insert INDUS', min_value=0.0, max_value=100.0, step=0.5)
        CHAS = st.number_input('Insert CHAS', min_value=0.0, max_value=100.0, step=0.5)
        NOX = st.number_input('Insert NOX', min_value=0.0, max_value=100.0, step=0.5)
        RM = st.number_input('Insert RM', min_value=0.0, max_value=100.0, step=0.5)
        AGE = st.number_input('Insert AGE', min_value=0.0, max_value=100.0, step=0.5)
        DIS = st.number_input('Insert DIS', min_value=0.0, max_value=100.0, step=0.5)
        RAD = st.number_input('Insert RAD', min_value=0.0, max_value=100.0, step=0.5)
        TAX = st.number_input('Insert TAX', min_value=0.0, max_value=1000.0, step=0.5)
        PTRATIO = st.number_input('Insert PTRATIO', min_value=0.0, max_value=100.0, step=0.5)
        B = st.number_input('Insert B', min_value=0.0, max_value=1000.0, step=0.5)
        LSTAT = st.number_input('Insert LSTAT', min_value=0.0, max_value=100.0, step=0.5)

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            prediction = model_methods.predict([CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT])

df = pd.DataFrame(
    [[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]],
    columns=(["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])
)
input_container = st.container()
input_container.subheader("Input Parameters")
input_container.table(df)

output_container = st.container()
output_container.subheader("Prediction")
try:
    if prediction is None:
        output_container.write("Please submit Input Parameters")
    else:
        output_container.write(prediction)
except:
    output_container.write("Please submit Input Parameters")
