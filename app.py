import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# โหลดข้อมูลและโมเดล
df = pd.read_csv(r"Loan_Applications_Dataset.csv")
model = load_model(r"loan_approval_model.h5")

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
    ### ผมนำข้อมูล DATASET ระบบเงินกู้ มาจากการ Generate ของ GPT
    ผมนำค่าที่ prediction ได้เป็นค่าความน่าจะเป็นที่โมเดลทำนายว่าผู้กู้จะได้รับการอนุมัติเงินกู้หรือไม่ โดยใช้เกณฑ์ 0.5 เป็นจุดตัดสินใจ (threshold) ในการจำแนกคลาส
    ### โมเดลนี้ใช้ค่า PREDICTION ที่เกิดจากการคำนวณ
    - ข้อมูล input ที่ผู้ใช้ป้อนเข้าไป
    - น้ำหนักและไบแอส ของโมเดลที่ได้จากการฝึก
    - สถาปัตยกรรมของโมเดล (จำนวน layers, neurons, ฟังก์ชัน activation)
    - ข้อมูล training set ที่ใช้ฝึกโมเดล
                 """)
    st.markdown("""
### 📊 ตัวอย่าง Feature ของ Dataset

| Column        | Description                                      |
|---------------|--------------------------------------------------|
| customer_id   | รหัสลูกค้า                                       |
| age           | อายุ                                             |
| income        | รายได้ต่อเดือน (บาท)                             |
| loan_amount   | จำนวนเงินกู้ที่ขอ (บาท)                          |
| loan_term     | ระยะเวลากู้ (เดือน)                              |
| loan_purpose  | วัตถุประสงค์การกู้ เช่น บ้าน, รถ, การศึกษา       |
| credit_score  | คะแนนเครดิต (0-1000)                            |
| approved      | ได้รับอนุมัติหรือไม่ (Yes/No)                    |
""")
    st.write("""   ### การเตรียมข้อมูล
    - โหลดข้อมูลจากไฟล์ CSV และตรวจสอบค่าที่หายไปและค่าผิดปกติ.
    - ทำความสะอาดข้อมูลและแปลงข้อมูลให้เหมาะสมสำหรับการฝึกโมเดล.
    ### ทฤษฎีของ Neural Network
    - Neural Network เป็นอัลกอริทึมที่เลียนแบบการทำงานของสมองมนุษย์.
    - ประกอบด้วย layers ต่าง ๆ ที่เชื่อมต่อกันและเรียนรู้จากข้อมูล.
    ### ขั้นตอนการพัฒนาโมเดล
    - ออกแบบโครงสร้างโมเดลและฝึกโมเดลด้วยชุดข้อมูล.
    - ทำความสะอาดข้อมูล
    - แก้ไขค่าที่หายไป
    - แปลง categorical features เป็น numerical
    - เลือก features และ target
    - แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
    - ปรับขนาดข้อมูล
    - สร้างโมเดล
    - คอมไพล์โมเดล
    - ประเมินโมเดล
    - ปรับ hyperparameters เพื่อให้ได้โมเดลที่ดีที่สุด""")
    st.image("Component/PIC1.png", caption="นำไฟล์Datasetเข้ามาลงในColab และ อ่าน", use_container_width=True)
    st.image("Component/PIC2.png", caption="ทำความสะอาดข้อมูลโดยการแก้ไขข้อผิดพลาดและแปลงค่าเป็น BINARY", use_container_width=True)
    st.image("Component/PIC3.png", caption="แก้ไขค่าที่หายไป", use_container_width=True)
    st.image("Component/PIC4.png", caption="แปลง categorical features เป็น numerical", use_container_width=True)
    st.image("Component/PIC5.png", caption="เลือก features และ target", use_container_width=True)
    st.image("Component/PIC6.png", caption="แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ", use_container_width=True)
    st.image("Component/PIC7.png", caption="ปรับขนาดข้อมูล", use_container_width=True)
    st.image("Component/PIC8.png", caption="สร้างโมเดล", use_container_width=True)
    st.image("Component/PIC9.png", caption="คอมไพล์โมเดล", use_container_width=True)
    st.image("Component/PIC11.png", caption="ประเมินโมเดล", use_container_width=True)
    

# หน้าเว็บ 2: Demo การทำงานของโมเดล
def show_demo():
    st.title("Demo การทำงานของโมเดล Neural Network")
    
    # สร้างฟอร์มสำหรับผู้ใช้ป้อนข้อมูล
    st.header("กรุณาป้อนข้อมูล")
    col1, col2 = st.columns(2)  # แบ่งหน้าจอเป็น 2 คอลัมน์
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Income ($)", min_value=0, value=50000)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=100000)
    
    with col2:
        loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        loan_purpose = st.selectbox("Loan Purpose", df["loan_purpose"].unique())
    
    # เกณฑ์ตัดสินใจ (threshold) = 0.5
    threshold = 0.5
    
    # เมื่อผู้ใช้กดปุ่ม "ทำนาย"
    if st.button("ทำนาย", key="predict_loan"):  # เพิ่ม key ที่ไม่ซ้ำกัน
        input_data = preprocess_input(age, income, loan_amount, loan_term, loan_purpose, credit_score)
        prediction = model.predict(input_data)[0][0]
        result = "✅ อนุมัติ" if prediction > threshold else "❌ ไม่อนุมัติ"
        st.subheader(f"ผลการทำนาย: {result}")
        st.write(f"ค่าความน่าจะเป็น: {prediction:.4f}")


# หน้าเว็บ 3: อธิบายแนวทางการพัฒนาโมเดล Machine Learning
def show_explanation2():
    st.title("การพัฒนาโมเดล Machine Learning สำหรับผู้ที่มีสิทธิเป็นเบาหวาน")
    st.write("""
    ### ผมนำข้อมูล DATASET ของผู้มีสิทธิเป็นเบาหวานมาจากการ Generate ของ GPT
    ผมนำข้อมูลที่ผ่านการ พัตนาโมเดลทั้ง 3 แบบ    
    - Support Vector Machine
    - Random Forest
    - Logistic Regression
    มาเช็คว่าถ้ามีค่าต่างเท่านี้จะเป็นเบาหวาน
    """)
    st.markdown("""
### 📊 ตาราง Feature Description

| Feature Name   | Description                                      | Data Type | Example Value |
|----------------|--------------------------------------------------|-----------|---------------|
| **customer_id**| รหัสลูกค้า                                       | Integer   | 12345         |
| **age**        | อายุ                                             | Integer   | 35            |
| **income**     | รายได้ต่อเดือน (บาท)                             | Float     | 50000.0       |
| **loan_amount**| จำนวนเงินกู้ที่ขอ (บาท)                          | Float     | 200000.0      |
| **loan_term**  | ระยะเวลากู้ (เดือน)                              | Integer   | 24            |
| **loan_purpose**| วัตถุประสงค์การกู้ เช่น บ้าน, รถ, การศึกษา       | String    | "บ้าน"        |
| **credit_score**| คะแนนเครดิต (0-1000)                            | Integer   | 750           |
| **approved**   | ได้รับอนุมัติหรือไม่ (Yes/No)                    | String    | "Yes"         |
""")
    st.write("""
    ### การเตรียมข้อมูล
    - โหลดข้อมูลจากไฟล์ CSV และตรวจสอบค่าที่หายไปและค่าผิดปกติ.
    - ทำความสะอาดข้อมูลและแปลงข้อมูลให้เหมาะสมสำหรับการฝึกโมเดล.
    ### ทฤษฎีของ Machine Learning
    - Machine Learning เป็นอัลกอริทึมที่เรียนรู้จากข้อมูลและทำนายผลลัพธ์.
    - ประกอบด้วยขั้นตอนการฝึกโมเดลและการทดสอบโมเดล.
    ### ขั้นตอนการพัฒนาโมเดล
    - ออกแบบโครงสร้างโมเดลและฝึกโมเดลด้วยชุดข้อมูล.
    - ทำความสะอาดข้อมูล (จัดการกับค่าที่หายไป)
    - แบ่งข้อมูลเป็น features (X) และ target (y)
    - แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
    - สร้างฟังก์ชันสำหรับการประเมินโมเดล
    - โดยใช้Algorithm 3 แบบ
    - Support Vector Machine
    - Random Forest
    - Logistic Regression
       """)
    st.image("Component/PIC2_1.png", caption="นำข้อมูลมาทำความสะอาด", use_container_width=True)
    st.image("Component/PIC2_2.png", caption="แบ่งข้อมูลเป็น features (X) และ target (y)", use_container_width=True)
    st.image("Component/PIC2_3.png", caption="แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ", use_container_width=True)
    st.image("Component/PIC2_4.png", caption="สร้างฟังก์ชันสำหรับการประเมินโมเดล", use_container_width=True)
    st.image("Component/PIC2_5.png", caption="ทดสอบ Logistic Regression", use_container_width=True)
    st.image("Component/PIC2_6.png", caption="ทดสอบ Support Vector Machine", use_container_width=True)
    st.image("Component/PIC2_7.png", caption="ทดสอบ Random Forest", use_container_width=True)
    

# หน้าเว็บ 4: Demo การทำงานของโมเดล Machine Learning
def show_demo2():
    st.title("Demo การทำงานของโมเดล Machine Learning สำหรับผู้เป็นเบาหวาน")
    
    # โหลดโมเดล Machine Learning
    log_reg_model = joblib.load("log_reg.pkl")
    rf_model = joblib.load("rf.pkl")
    svm_model = joblib.load("svm.pkl")
    
    # เลือกอัลกอริทึม
    algorithm = st.selectbox("เลือกอัลกอริทึม", ["Logistic Regression", "Random Forest", "Support Vector Machine"])
    
    # สร้างฟอร์มสำหรับผู้ใช้ป้อนข้อมูล
    st.header("กรุณาป้อนข้อมูล")
    col1, col2 = st.columns(2)  # แบ่งหน้าจอเป็น 2 คอลัมน์
    
    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        Glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    
    with col2:
        Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        Age = st.number_input("Age", min_value=0, max_value=120, value=30)
    
    # เกณฑ์ตัดสินใจ (threshold) = 0.5
    threshold = 0.5
    
    # เมื่อผู้ใช้กดปุ่ม "ทำนาย"
    if st.button("ทำนาย", key="predict_diabetes"):
        input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],
                                 columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
        
        # เลือกโมเดลตามอัลกอริทึมที่ผู้ใช้เลือก
        if algorithm == "Logistic Regression":
            model = log_reg_model
        elif algorithm == "Random Forest":
            model = rf_model
        elif algorithm == "Support Vector Machine":
            model = svm_model
        
        # ทำนายผล
        prediction = model.predict(input_data)[0]
        result = "✅ เป็นเบาหวาน" if prediction > threshold else "❌ ไม่เป็นเบาหวาน"
        st.subheader(f"ผลการทำนาย: {result}")
        st.write(f"ค่าความน่าจะเป็น: {prediction:.4f}")
        st.write(f"อัลกอริทึมที่ใช้: {algorithm}")
# กำหนดหน้าเว็บ
pages = {
    "อธิบายแนวทางการพัฒนาโมเดล Neural Network": show_explanation,
    "Demo การทำงานของโมเดล Neural Network": show_demo,
    "อธิบายแนวทางการพัฒนาโมเดล Machine Learning": show_explanation2,
    "Demo การทำงานของโมเดล Machine Learning": show_demo2,
}

# สร้างเมนูแบบกด
tab1, tab2 ,tab3 ,tab4= st.tabs(["อธิบายแนวทางการพัฒนาโมเดล NM", "Demo การทำงานของโมเดล NM","อธิบายแนวทางการพัฒนาโมเดล ML","Demo การทำงานของโมเดล ML"])

with tab1:
    show_explanation()
with tab2:
    show_demo()
with tab3:
    show_explanation2()
with tab4:
    show_demo2()