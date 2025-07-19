
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample dataset
data = pd.DataFrame({
    'YearsExperience': [1, 2, 3, 4, 5, 6, 7],
    'Education': ['Bachelors', 'Masters', 'PhD', 'Bachelors', 'Masters', 'PhD', 'Bachelors'],
    'JobTitle': ['Developer', 'Data Scientist', 'Manager', 'Developer', 'Manager', 'Data Scientist', 'Developer'],
    'Company': ['Google', 'Amazon', 'Microsoft', 'Amazon', 'Google', 'Meta', 'Meta'],
    'Location': ['New York', 'San Francisco', 'Seattle', 'Austin', 'Seattle', 'New York', 'Austin'],
    'Salary': [60000, 80000, 120000, 65000, 115000, 130000, 70000]
})

# Train model
X = data.drop('Salary', axis=1)
y = data['Salary']
categorical_features = ['Education', 'JobTitle', 'Company', 'Location']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), categorical_features),
    ('num', 'passthrough', ['YearsExperience'])
])
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
model.fit(X, y)

# ------------------ Streamlit UI ------------------

# Sidebar: User Info
st.sidebar.title("📝 User Info")
user_name = st.sidebar.text_input("Name")
user_email = st.sidebar.text_input("Email")
user_address = st.sidebar.text_area("Address")

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    st.header("📥 Input")
    experience = st.slider("Years of Experience", 0, 20, 2)
    education = st.selectbox("Education Level", data['Education'].unique())
    job_title = st.selectbox("Job Title", data['JobTitle'].unique())
    company = st.selectbox("Company", data['Company'].unique())
    location = st.selectbox("Location", data['Location'].unique())

with col2:
    st.header("📤 Output")
    if st.button("Predict Salary"):
        input_df = pd.DataFrame({
            'YearsExperience': [experience],
            'Education': [education],
            'JobTitle': [job_title],
            'Company': [company],
            'Location': [location]
        })
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Salary: ₹{int(prediction):,}")

    # Pie chart
    st.subheader("📊 Salary Breakdown")
    pie_data = data.groupby('JobTitle')['Salary'].mean()
    fig, ax = plt.subplots()
    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
    st.pyplot(fig)

# Developer Contact Section
st.markdown("---")
st.subheader("👨‍💻 Developer Contact")
st.markdown("""
**Name:** Shan Shareef 
**Email:** `shan@example.com`  
**GitHub:** [github.com/SHAN2348](https://github.com/SHAN2348)  
**Location:** India
""")

# Upload image
st.markdown("---")
st.subheader("📸 Upload Company Logo")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Logo", use_column_width=True)
