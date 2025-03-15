import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# โหลดข้อมูลและโมเดล
df = pd.read_csv(r"D:\Intellligence\Loan_Applications_Dataset.csv")
model = load_model(r"D:\Intellligence\loan_approval_model.h5")

# One-Hot Encoding สำหรับ loan_purpose (แก้ไข sparse เป็น sparse_output)
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_purpose = encoder.fit_transform(df[['loan_purpose']])

# เตรียม StandardScaler
scaler = StandardScaler()
scaler.fit(df[['age', 'income', 'loan_amount', 'loan_term', 'credit_score']])

# ฟังก์ชันเตรียมข้อมูล
def preprocess_input(age, income, loan_amount, loan_term, loan_purpose, credit_score):
    # One-Hot Encoding
    purpose_encoded = encoder.transform([[loan_purpose]])
    purpose_df = pd.DataFrame(purpose_encoded, columns=encoder.get_feature_names_out(['loan_purpose']))

    # รวมข้อมูล input
    input_data = pd.DataFrame([[age, income, loan_amount, loan_term, credit_score]], 
                              columns=["age", "income", "loan_amount", "loan_term", "credit_score"])
    
    input_data = pd.concat([input_data, purpose_df], axis=1)

    # **เลือกเฉพาะคอลัมน์ที่โมเดลต้องการ**
    input_scaled = scaler.transform(input_data.iloc[:, :5])  # ปรับสเกลเฉพาะตัวเลข
    input_scaled = np.hstack((input_scaled, input_data.iloc[:, -1].values.reshape(-1, 1)))  # รวมกับ 1 คอลัมน์ของ One-Hot Encoding

    return input_scaled  # **ต้องได้ (1, 6) เท่านั้น**


# หน้าเว็บ 1: อธิบายแนวทางการพัฒนา
def show_explanation():
    st.title("การพัฒนาโมเดล Neural Network สำหรับการอนุมัติเงินกู้")
    st.write("""
    ### การเตรียมข้อมูล
    - โหลดข้อมูลจากไฟล์ CSV และตรวจสอบค่าที่หายไปและค่าผิดปกติ.
    - ทำความสะอาดข้อมูลและแปลงข้อมูลให้เหมาะสมสำหรับการฝึกโมเดล.
    ### ทฤษฎีของ Neural Network
    - Neural Network เป็นอัลกอริทึมที่เลียนแบบการทำงานของสมองมนุษย์.
    - ประกอบด้วย layers ต่าง ๆ ที่เชื่อมต่อกันและเรียนรู้จากข้อมูล.
    ### ขั้นตอนการพัฒนาโมเดล
    - ออกแบบโครงสร้างโมเดลและฝึกโมเดลด้วยชุดข้อมูล.
    - ปรับ hyperparameters เพื่อให้ได้โมเดลที่ดีที่สุด.
    ### โมเดลนี้ใช้ค่า PREDICTION ที่เกิดจากการคำนวณ
    - ข้อมูล input ที่ผู้ใช้ป้อนเข้าไป
    - น้ำหนักและไบแอส ของโมเดลที่ได้จากการฝึก
    - สถาปัตยกรรมของโมเดล (จำนวน layers, neurons, ฟังก์ชัน activation)
    - ข้อมูล training set ที่ใช้ฝึกโมเดล
    ### ค่า prediction เป็นค่าความน่าจะเป็นที่โมเดลทำนายว่าผู้กู้จะได้รับการอนุมัติเงินกู้หรือไม่ โดยใช้เกณฑ์ 0.5 เป็นจุดตัดสินใจ (threshold) ในการจำแนกคลาส
    """)
# หน้าเว็บ 2: Demo การทำงานของโมเดล
def show_demo():
    st.title("Demo การทำงานของโมเดล")
    
    # สร้างฟอร์มสำหรับผู้ใช้ป้อนข้อมูล
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income ($)", min_value=0, value=50000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=100000)
    loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    loan_purpose = st.selectbox("Loan Purpose", df["loan_purpose"].unique())
    
    # เกณฑ์ตัดสินใจ (threshold) = 0.5
    threshold = 0.5
    
    # เมื่อผู้ใช้กดปุ่ม "ทำนาย"
    if st.button("ทำนาย"):
        input_data = preprocess_input(age, income, loan_amount, loan_term, loan_purpose, credit_score)
        prediction = model.predict(input_data)[0][0]
        result = "✅ อนุมัติ" if prediction > threshold else "❌ ไม่อนุมัติ"
        st.subheader(f"ผลการทำนาย: {result}")
        st.write(f"ค่าความน่าจะเป็น: {prediction:.4f}")
def show_demo2():
    st.title("Demo การทำงานของโมเดลของ Machine Learning")
def show_explanation2():
    st
# กำหนดหน้าเว็บ
pages = {
    "อธิบายแนวทางการพัฒนาโมเดล Neural Network": show_explanation,
    "อธิบายแนวทางการพัฒนาโมเดล Machine Learning": show_explanation2,
    "Demo การทำงานของโมเดล Neural Network": show_demo,
    "Demo การทำงานของโมเดล Machine Learning":show_demo2

}

# สร้าง sidebar สำหรับเลือกหน้า
selected_page = st.sidebar.selectbox("เลือกหน้า", list(pages.keys()))
pages[selected_page]()
