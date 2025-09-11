import pandas as pd
import random

def generate_clinical_notes(num_records):
    diagnosis = {
         "Diabetes": ["high blood sugar", "increased thirst", "frequent urination"],
        "Hypertension": ["chest pain", "headache", "shortness of breath"],
        "Asthma": ["wheezing", "coughing", "difficulty breathing"],
        "Coronary Artery Disease": ["chest tightness", "fatigue", "shortness of breath"],
        "Healthy": ["routine check-up", "no significant complaints"]
    }

    medications = {
         "Diabetes": ["Metformin", "Insulin"],
        "Hypertension": ["Lisinopril", "Amlodipine"],
        "Asthma": ["Albuterol", "Fluticasone"],
        "Coronary Artery Disease": ["Aspirin", "Atorvastatin"],
        "Healthy": ["None"]
    }

    notes = []
    
    for i in range(num_records + 1):
        diag = random.choice(list(diagnosis.keys()))
        symptom = random.choice(diagnosis[diag])
        med = random.choice(diagnosis[diag])

        note = (
            f"Patient is a {random.randint(20,80)}-year-old "
            f"{random.choice(["Male","Female"])} with a history of {diag}. "
            f"Presenting symptoms include {symptom}. "
            f"Currently prescribed {med}. "
            f"Follow-up recommended in {random.choice(['2 week','1 month','3 month'])}"
        )

        notes.append({"patient_id":i, 
                      "note_text":note})
        
    return pd.DataFrame(notes)

if __name__ == "__main__":

    df = generate_clinical_notes(200)
    df.to_csv("data/raw/clinical_notes.csv", index=False)
    print("Synthetic clinical notes are saved")