import pandas as pd
import random 

def generator_ehr_data(num_records=200):
    diagnosis = ["Diabetes", "Hypertension", "Asthma", "Coronary Artery Disease", "Healthy"]
    medication = {
        "Diabetes": ["Metformin", "Insulin"],
        "Hypertension": ["Lisinopril", "Amlodipine"],
        "Asthma": ["Albuterol", "Fluticasone"],
        "Coronary Artery Disease": ["Aspirin", "Atorvastatin"],
        "Healthy": ["None"]
    }

    data = []
    for i in range(1,num_records+1):
        diagnoses = random.choice(diagnosis)
        record = {
            "patient_id":i,
            "diagnosis":diagnoses,
            "medication" : random.choice(medication[diagnoses]),
            "blood_pressure": f"{random.randint(90,160)}/{random.randint(60,100)}",
            "heart_rate" : random.randint(60,110),
            "glucose_level" : random.randint(70,250)

        }

        data.append(record)

    return pd.DataFrame(data)

if __name__== "__main__":
    df = generator_ehr_data(200)
    df.to_csv("data/raw/patient_ehr.csv", index=False)
    print("Synthetic ejr data saved at data/raw/patient_ehr.csv")