# About Dataset
Dữ liệu này chứa thông tin sức khỏe chi tiết của 2,149 bệnh nhân, mỗi bệnh nhân được xác định duy nhất bằng mã ID từ 4751 đến 6900. Dữ liệu bao gồm các thông tin nhân khẩu học, yếu tố lối sống, tiền sử bệnh lý, các chỉ số lâm sàng, đánh giá nhận thức và chức năng, triệu chứng, và chẩn đoán bệnh Alzheimer. Dữ liệu này lý tưởng cho các nhà nghiên cứu và các nhà khoa học dữ liệu muốn khám phá các yếu tố liên quan đến bệnh Alzheimer, phát triển các mô hình dự đoán, và thực hiện các phân tích thống kê.


## Table of Contents
- [About Dataset](#about-dataset)
  - [Table of Contents](#table-of-contents)
  - [Patient Information](#patient-information)
    - [Patient ID](#patient-id)
  - [Demographic Details](#demographic-details)
  - [Lifestyle Factors](#lifestyle-factors)
  - [Medical History](#medical-history)
  - [Clinical Measurements](#clinical-measurements)
  - [Cognitive and Functional Assessments](#cognitive-and-functional-assessments)
  - [Symptoms](#symptoms)
  - [Diagnosis Information](#diagnosis-information)
  - [Confidential Information](#confidential-information)

## Patient Information
### Patient ID
- `PatientID`: Mã định danh duy nhất được gán cho mỗi bệnh nhân (từ 4751 đến 6900).

## Demographic Details
- **Age**: Tuổi của bệnh nhân, dao động từ 60 đến 90 tuổi.
- **Gender**: Giới tính của bệnh nhân, với:
  - 0 là Nam
  - 1 là Nữ
- **Ethnicity**: Dân tộc của bệnh nhân, được mã hóa như sau:
  - 0: Người da trắng (Caucasian)
  - 1: Người da đen (African American)
  - 2: Người châu Á (Asian)
  - 3: Dân tộc khác (Other)
- **EducationLevel**: Trình độ học vấn của bệnh nhân, được mã hóa như sau:
  - 0: Không có học vấn
  - 1: Trung học
  - 2: Cử nhân
  - 3: Trình độ cao hơn

## Lifestyle Factors
- **BMI**: Chỉ số khối cơ thể (Body Mass Index) của bệnh nhân, dao động từ 15 đến 40.
- **Smoking**: Tình trạng hút thuốc, với:
  - 0 là Không
  - 1 là Có
- **AlcoholConsumption**: Lượng rượu tiêu thụ hàng tuần tính theo đơn vị, dao động từ 0 đến 20.
- **PhysicalActivity**: Số giờ hoạt động thể chất hàng tuần, dao động từ 0 đến 10 giờ.
- **DietQuality**: Điểm số chất lượng chế độ ăn uống, dao động từ 0 đến 10.
- **SleepQuality**: Điểm số chất lượng giấc ngủ, dao động từ 4 đến 10.

## Medical History
- **FamilyHistoryAlzheimers**: Tiền sử gia đình mắc bệnh Alzheimer, với:
  - 0 là Không
  - 1 là Có
- **CardiovascularDisease**: Tiền sử bệnh tim mạch, với:
  - 0 là Không
  - 1 là Có
- **Diabetes**: Tiền sử tiểu đường, với:
  - 0 là Không
  - 1 là Có
- **Depression**: Tiền sử trầm cảm, với:
  - 0 là Không
  - 1 là Có
- **HeadInjury**: Tiền sử chấn thương đầu, với:
  - 0 là Không
  - 1 là Có
- **Hypertension**: Tiền sử tăng huyết áp, với:
  - 0 là Không
  - 1 là Có

## Clinical Measurements
- **SystolicBP**: Huyết áp tâm thu, dao động từ 90 đến 180 mmHg.
- **DiastolicBP**: Huyết áp tâm trương, dao động từ 60 đến 120 mmHg.
- **CholesterolTotal**: Mức cholesterol tổng, dao động từ 150 đến 300 mg/dL.
- **CholesterolLDL**: Mức cholesterol lipoprotein tỷ trọng thấp (LDL), dao động từ 50 đến 200 mg/dL.
- **CholesterolHDL**: Mức cholesterol lipoprotein tỷ trọng cao (HDL), dao động từ 20 đến 100 mg/dL.
- **CholesterolTriglycerides**: Mức triglyceride, dao động từ 50 đến 400 mg/dL.

## Cognitive and Functional Assessments
- **MMSE**: Điểm số Mini-Mental State Examination, dao động từ 0 đến 30. Điểm thấp hơn cho thấy sự suy giảm nhận thức.
- **FunctionalAssessment**: Điểm đánh giá chức năng, dao động từ 0 đến 10. Điểm thấp hơn cho thấy sự suy giảm chức năng lớn hơn.
- **MemoryComplaints**: Sự hiện diện của các phàn nàn về trí nhớ, với:
  - 0 là Không
  - 1 là Có
- **BehavioralProblems**: Sự hiện diện của các vấn đề hành vi, với:
  - 0 là Không
  - 1 là Có
- **ADL**: Điểm số các hoạt động sinh hoạt hàng ngày, dao động từ 0 đến 10. Điểm thấp hơn cho thấy sự suy giảm chức năng sinh hoạt hàng ngày lớn hơn.

## Symptoms
- **Confusion**: Sự hiện diện của sự bối rối, với:
  - 0 là Không
  - 1 là Có
- **Disorientation**: Sự hiện diện của sự mất phương hướng, với:
  - 0 là Không
  - 1 là Có
- **PersonalityChanges**: Sự hiện diện của thay đổi nhân cách, với:
  - 0 là Không
  - 1 là Có
- **DifficultyCompletingTasks**: Sự hiện diện của khó khăn trong việc hoàn thành các nhiệm vụ, với:
  - 0 là Không
  - 1 là Có
- **Forgetfulness**: Sự hiện diện của trí nhớ kém, với:
  - 0 là Không
  - 1 là Có

## Diagnosis Information
- **Diagnosis**: Tình trạng chẩn đoán bệnh Alzheimer, với:
  - 0 là Không
  - 1 là Có

## Confidential Information
- **DoctorInCharge**: Cột này chứa thông tin bảo mật về bác sĩ phụ trách, với giá trị là "XXXConfid" cho tất cả bệnh nhân.
